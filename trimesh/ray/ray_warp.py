"""
Ray queries using the Warp package with the
API wrapped to match our native raytracer.

Warp provides GPU-accelerated (and CPU-fallback) mesh
queries including ray casting and closest-point via BVH.

By default Warp runs kernels on ``"cuda:0"`` when a GPU is
present, and falls back to the CPU otherwise.  Users can
override this globally *before* any mesh is created by
calling ``warp.set_device("cpu")`` (or setting
``CUDA_VISIBLE_DEVICES=""``), which is useful for debugging
or for machines without a GPU.
"""

import numpy as np
import warp as wp

from .. import caching, intersections, util
from ..constants import log_time
from .ray_util import contains_points

# match the Embree backend constants so the iterative
# ray-offset logic has the same numerical behavior
# the factor of geometry.scale to offset a ray from a triangle
# to reliably not hit its origin triangle
_ray_offset_factor = 1e-4
# we want to clip our offset to a sane distance
_ray_offset_floor = 1e-8


def _get_device():
    """Return Warp's preferred device (``"cuda:0"`` when a GPU is
    present, ``"cpu"`` otherwise).  Respects any prior call to
    ``warp.set_device()``."""
    try:
        return wp.get_preferred_device()
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# Warp kernels (must be at module level for codegen / inspect.getsourcelines)
# ---------------------------------------------------------------------------


@wp.kernel
def _kernel_ray_first_hit(
    mesh: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    offset: wp.vec3,
    scale: float,
    max_t: float,
    out_face: wp.array(dtype=int),
):
    """Cast one ray per thread and record the first triangle hit (-1 for miss).

    The scaling transform (subtract *offset*, multiply by *scale*) is
    applied on-GPU to avoid expensive CPU-side NumPy arithmetic.
    """
    tid = wp.tid()
    o = (origins[tid] - offset) * scale
    query = wp.mesh_query_ray(mesh, o, directions[tid], max_t)
    if query.result:
        out_face[tid] = query.face


@wp.kernel
def _kernel_closest_point(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    offset: wp.vec3,
    scale: float,
    max_dist: float,
    out_closest: wp.array(dtype=wp.vec3),
    out_dist: wp.array(dtype=float),
    out_face: wp.array(dtype=int),
    out_sign: wp.array(dtype=float),
):
    """For each query point, find the closest point on the mesh surface.

    The scaling transform is applied on-GPU; results are returned in
    the scaled coordinate frame (the caller unscales on readback).
    """
    tid = wp.tid()
    q = (points[tid] - offset) * scale
    query = wp.mesh_query_point_sign_normal(mesh, q, max_dist)
    if query.result:
        p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        out_closest[tid] = p
        out_dist[tid] = wp.length(p - q)
        out_face[tid] = query.face
        out_sign[tid] = query.sign


# ---------------------------------------------------------------------------
# Internal wrapper that manages a Warp mesh on-device
# ---------------------------------------------------------------------------


class _WarpWrap:
    """
    A light wrapper for Warp Mesh objects which allows queries
    to be scaled to help with precision issues, as well as
    handling dtype conversion between trimesh (float64) and
    Warp (float32).
    """

    def __init__(self, vertices, faces, scale):
        self.device = _get_device()

        verts_f64 = np.array(vertices, dtype=np.float64)
        self.origin = verts_f64.min(axis=0)
        self.scale = float(scale)
        scaled = (verts_f64 - self.origin) * self.scale

        wp_points = wp.array(scaled.astype(np.float32), dtype=wp.vec3, device=self.device)
        wp_indices = wp.array(
            faces.view(np.ndarray).astype(np.int32).ravel(),
            dtype=wp.int32,
            device=self.device,
        )
        self.mesh = wp.Mesh(points=wp_points, indices=wp_indices)

        # store origin and scale as float32 for the GPU kernels
        self._wp_offset = wp.vec3(
            float(self.origin[0]),
            float(self.origin[1]),
            float(self.origin[2]),
        )
        self._wp_scale = float(self.scale)

        # derive max query distance from the scaled mesh extents
        # so we don't introduce arbitrary numeric constants
        extent = scaled.max(axis=0) - scaled.min(axis=0)
        self._max_t = float(np.linalg.norm(extent)) * 10.0

    def _upload_vec3(self, arr):
        """Convert a float64 numpy array to a float32 Warp vec3 array on device.

        No CPU-side scaling — the kernels handle the transform on-GPU.
        """
        return wp.array(
            np.ascontiguousarray(arr, dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )

    def run(self, origins, directions):
        """
        Run a first-hit ray query and return face indices.

        Parameters
        ----------
        origins : (n, 3) float
          Ray origin points (world space, float64).
        directions : (n, 3) float
          Ray direction vectors (should already be unitised).

        Returns
        ----------
        face_idx : (n,) int
          Triangle index per ray, or -1 for a miss.
        """
        n = len(origins)
        if n == 0:
            return np.array([], dtype=np.int64)

        wp_origins = self._upload_vec3(origins)
        wp_dirs = self._upload_vec3(directions)

        out_face = wp.full(n, value=-1, dtype=int, device=self.device)

        wp.launch(
            _kernel_ray_first_hit,
            dim=n,
            inputs=[
                self.mesh.id,
                wp_origins,
                wp_dirs,
                self._wp_offset,
                self._wp_scale,
                self._max_t,
            ],
            outputs=[out_face],
            device=self.device,
        )

        return out_face.numpy()

    def closest_point(self, points):
        """
        Find the closest point on the mesh for each query point.

        Parameters
        ----------
        points : (n, 3) float
          Query points in *world space* (float64).

        Returns
        ----------
        closest : (n, 3) float64
          Closest points on mesh surface (world space).
        distance : (n,) float64
          Euclidean distances.
        face_id : (n,) int
          Triangle indices.
        sign : (n,) float
          Sign indicator (<0 inside, >=0 outside).
        """
        n = len(points)
        if n == 0:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
            )

        wp_pts = self._upload_vec3(points)

        out_closest = wp.zeros(n, dtype=wp.vec3, device=self.device)
        out_dist = wp.full(n, value=self._max_t, dtype=float, device=self.device)
        out_face = wp.full(n, value=-1, dtype=int, device=self.device)
        out_sign = wp.zeros(n, dtype=float, device=self.device)

        wp.launch(
            _kernel_closest_point,
            dim=n,
            inputs=[
                self.mesh.id,
                wp_pts,
                self._wp_offset,
                self._wp_scale,
                self._max_t,
            ],
            outputs=[out_closest, out_dist, out_face, out_sign],
            device=self.device,
        )

        # unscale results back to world space
        closest_scaled = out_closest.numpy().astype(np.float64)
        closest_world = closest_scaled / self.scale + self.origin

        distance_scaled = out_dist.numpy().astype(np.float64)
        distance_world = distance_scaled / self.scale

        return closest_world, distance_world, out_face.numpy(), out_sign.numpy()


# ---------------------------------------------------------------------------
# Public API: RayMeshIntersector (matches ray_pyembree interface exactly)
# ---------------------------------------------------------------------------


class RayMeshIntersector:
    """
    Ray-mesh intersection queries using Warp.

    This has the same API as the Embree and NumPy triangle backends
    so that it can be used as a drop-in replacement.
    """

    def __init__(self, geometry, scale_to_box=True):
        """
        Do ray-mesh queries.

        Parameters
        ----------
        geometry : Trimesh object
          Mesh to do ray tests on.
        scale_to_box : bool
          If True, will scale mesh to an approximate unit cube
          to avoid precision problems with very large or small meshes.
        """
        self.mesh = geometry
        self._scale_to_box = scale_to_box
        self._cache = caching.Cache(id_function=self.mesh.__hash__)

    @property
    def _scale(self):
        """Scaling factor for precision."""
        if self._scale_to_box:
            scale = 100.0 / self.mesh.scale
        else:
            scale = 1.0
        return scale

    @caching.cache_decorator
    def _scene(self):
        """A cached version of the Warp scene."""
        return _WarpWrap(
            vertices=self.mesh.vertices, faces=self.mesh.faces, scale=self._scale
        )

    def intersects_location(self, ray_origins, ray_directions, multiple_hits=True):
        """
        Return the location of where a ray hits a surface.

        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays.
        ray_directions : (n, 3) float
          Direction (vector) of rays.

        Returns
        ----------
        locations : (m, 3) float
          Intersection points.
        index_ray : (m,) int
          Indexes of ray.
        index_tri : (m,) int
          Indexes of mesh.faces.
        """
        (index_tri, index_ray, locations) = self.intersects_id(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=multiple_hits,
            return_locations=True,
        )
        return locations, index_ray, index_tri

    @log_time
    def intersects_id(
        self,
        ray_origins,
        ray_directions,
        multiple_hits=True,
        max_hits=20,
        return_locations=False,
    ):
        """
        Find the triangles hit by a list of rays, including
        optionally multiple hits along a single ray.

        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays.
        ray_directions : (n, 3) float
          Direction (vector) of rays.
        multiple_hits : bool
          If True will return every hit along the ray.
          If False will only return first hit.
        max_hits : int
          Maximum number of hits per ray.
        return_locations : bool
          Should we return hit locations or not.

        Returns
        ----------
        index_tri : (m,) int
          Indexes of mesh.faces.
        index_ray : (m,) int
          Indexes of ray.
        locations : (m, 3) float
          Intersection points (only if return_locations).
        """
        ray_origins = np.array(ray_origins, dtype=np.float64)
        ray_directions = np.array(ray_directions, dtype=np.float64)
        if ray_origins.shape != ray_directions.shape:
            raise ValueError("Ray origin and direction don't match!")
        ray_directions = util.unitize(ray_directions)

        # collect results across multiple depth passes
        result_triangle = []
        result_ray_idx = []
        result_locations = []

        # mask for rays that are still active
        current = np.ones(len(ray_origins), dtype=bool)

        if multiple_hits or return_locations:
            distance = np.clip(
                _ray_offset_factor * self._scale, _ray_offset_floor, np.inf
            )
            ray_offsets = ray_directions * distance
            # grab the planes from triangles for precise location
            plane_origins = self.mesh.triangles[:, 0, :]
            plane_normals = self.mesh.face_normals

        # use a for loop rather than a while to ensure this exits
        for _ in range(max_hits):
            query = self._scene.run(ray_origins[current], ray_directions[current])
            hit = query != -1
            hit_triangle = query[hit]

            current_index = np.nonzero(current)[0]
            current_index_no_hit = current_index[np.logical_not(hit)]
            current_index_hit = current_index[hit]
            current[current_index_no_hit] = False

            result_triangle.append(hit_triangle)
            result_ray_idx.append(current_index_hit)

            if (not multiple_hits and not return_locations) or not hit.any():
                break

            # find the location of where the ray hit the triangle plane
            new_origins, valid = intersections.planes_lines(
                plane_origins=plane_origins[hit_triangle],
                plane_normals=plane_normals[hit_triangle],
                line_origins=ray_origins[current],
                line_directions=ray_directions[current],
            )

            if not valid.all():
                result_ray_idx.append(result_ray_idx.pop()[valid])
                result_triangle.append(result_triangle.pop()[valid])
                current[current_index_hit[np.logical_not(valid)]] = False

            result_locations.extend(new_origins)

            if multiple_hits:
                ray_origins[current] = new_origins + ray_offsets[current]
            else:
                break

        index_tri = (
            np.hstack(result_triangle)
            if result_triangle
            else np.array([], dtype=np.int64)
        )
        index_ray = (
            np.hstack(result_ray_idx) if result_ray_idx else np.array([], dtype=np.int64)
        )

        if return_locations:
            locations = (
                np.zeros((0, 3), float)
                if len(result_locations) == 0
                else np.array(result_locations)
            )
            return index_tri, index_ray, locations

        return index_tri, index_ray

    @log_time
    def intersects_first(self, ray_origins, ray_directions):
        """
        Find the index of the first triangle a ray hits.

        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays.
        ray_directions : (n, 3) float
          Direction (vector) of rays.

        Returns
        ----------
        triangle_index : (n,) int
          Index of triangle ray hit, or -1 if not hit.
        """
        ray_origins = np.array(ray_origins, dtype=np.float64)
        ray_directions = np.array(ray_directions, dtype=np.float64)
        if ray_origins.shape != ray_directions.shape:
            raise ValueError("Ray origin and direction don't match!")
        ray_directions = util.unitize(ray_directions)

        return self._scene.run(ray_origins, ray_directions)

    def intersects_any(self, ray_origins, ray_directions):
        """
        Check if a list of rays hits the surface.

        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays.
        ray_directions : (n, 3) float
          Direction (vector) of rays.

        Returns
        ----------
        hit : (n,) bool
          Did each ray hit the surface.
        """
        first = self.intersects_first(
            ray_origins=ray_origins, ray_directions=ray_directions
        )
        return first != -1

    def contains_points(self, points):
        """
        Check if a mesh contains a list of points, using ray tests.

        If the point is on the surface of the mesh, behavior is undefined.

        Parameters
        ----------
        points : (n, 3) float
          Points in space.

        Returns
        ----------
        contains : (n,) bool
          Whether point is inside mesh or not.
        """
        return contains_points(self, points)

    def closest_point(self, points):
        """
        Given a list of points find the closest point on any
        triangle of the mesh.

        Parameters
        ----------
        points : (m, 3) float
          Points in space.

        Returns
        ----------
        closest : (m, 3) float
          Closest point on triangles for each point.
        distance : (m,) float
          Distance to mesh.
        triangle_id : (m,) int
          Index of triangle containing closest point.
        """
        points = np.asanyarray(points, dtype=np.float64)
        if not util.is_shape(points, (-1, 3)):
            raise ValueError("points must be (n,3)!")

        closest, distance, triangle_id, _sign = self._scene.closest_point(points)
        return closest, distance, triangle_id

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_cache", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache = caching.Cache(id_function=self.mesh.__hash__)

    def __deepcopy__(self, *args):
        return self.__copy__()

    def __copy__(self, *args):
        return RayMeshIntersector(geometry=self.mesh, scale_to_box=self._scale_to_box)
