"""
Tests for Warp GPU-accelerated proximity queries.

These tests verify that the Warp backend for closest-point
and signed-distance queries produces results consistent with
the CPU implementation.
"""

try:
    from . import generic as g
except BaseException:
    import generic as g


def _sphere(use_warp=False):
    """Create a unit sphere as a Trimesh with the given backend."""
    return g.trimesh.creation.icosphere(subdivisions=4, use_warp=use_warp)


class WarpProximityTest(g.unittest.TestCase):
    def setUp(self):
        """Skip all tests if Warp is not available."""
        if not g.trimesh.ray.has_warp:
            self.skipTest("Warp not available")

    def test_closest_point_matches_cpu(self):
        """
        The Warp closest-point result should closely match the
        CPU (r-tree + barycentric) implementation.
        """
        points = g.trimesh.sample.sample_surface_sphere(200) * 2.0

        # CPU path
        close_cpu, dist_cpu, _tid_cpu = g.trimesh.proximity.closest_point(
            _sphere(use_warp=False), points
        )
        # Warp path
        sphere_warp = _sphere(use_warp=True)
        close_warp, dist_warp, tid_warp = g.trimesh.proximity.closest_point(
            sphere_warp, points
        )

        # distances should agree within float32 tolerance
        g.np.testing.assert_allclose(dist_warp, dist_cpu, atol=0.02)
        # closest points should be very similar
        g.np.testing.assert_allclose(close_warp, close_cpu, atol=0.02)
        # triangle IDs might differ at edges/vertices but all should be valid
        assert (tid_warp >= 0).all()
        assert (tid_warp < len(sphere_warp.faces)).all()

    def test_closest_point_distance(self):
        """
        Points on a radius-2 sphere queried against a radius-1
        sphere should have distance ~1.0.
        """
        points = g.trimesh.sample.sample_surface_sphere(100) * 2.0

        _closest, distance, _tid = g.trimesh.proximity.closest_point(
            _sphere(use_warp=True), points
        )
        g.np.testing.assert_allclose(distance, 1.0, atol=0.02)

    def test_closest_point_on_surface(self):
        """
        Closest points returned should lie on the mesh surface
        (approximately at radius 1.0 for a unit sphere).
        """
        points = g.trimesh.sample.sample_surface_sphere(100) * 3.0

        closest, _distance, _tid = g.trimesh.proximity.closest_point(
            _sphere(use_warp=True), points
        )
        surface_radii = g.np.linalg.norm(closest, axis=1)
        g.np.testing.assert_allclose(surface_radii, 1.0, atol=0.02)

    def test_signed_distance(self):
        """
        Signed distance should be positive inside and negative outside.
        """
        mesh = g.trimesh.creation.icosphere(radius=1.0, use_warp=True)

        inside = g.np.array([[0, 0, 0], [0.1, 0.2, 0.3]])
        outside = g.np.array([[0, 0, 5], [3, 0, 0]])
        points = g.np.vstack([inside, outside])

        sd = g.trimesh.proximity.signed_distance(mesh, points)

        # inside points should have positive signed distance
        assert (sd[:2] > 0).all(), f"Inside should be positive: {sd[:2]}"
        # outside points should have negative signed distance
        assert (sd[2:] < 0).all(), f"Outside should be negative: {sd[2:]}"

    def test_signed_distance_matches_cpu(self):
        """
        Warp-backed signed distance should match the CPU implementation.
        """
        points = g.np.vstack(
            (
                g.trimesh.sample.sample_surface_sphere(100) * 0.5,
                g.trimesh.sample.sample_surface_sphere(100) * 2.0,
            )
        )

        sd_cpu = g.trimesh.proximity.signed_distance(
            g.trimesh.creation.icosphere(radius=1.0, use_warp=False), points
        )
        sd_warp = g.trimesh.proximity.signed_distance(
            g.trimesh.creation.icosphere(radius=1.0, use_warp=True), points
        )
        g.np.testing.assert_allclose(sd_warp, sd_cpu, atol=0.02)

    def test_closest_point_interior(self):
        """
        For a point inside a mesh, the closest-point distance
        should match the distance to the nearest surface point.
        """
        mesh = g.trimesh.creation.icosphere(radius=1.0, use_warp=True)
        points = g.np.array([[0.5, 0, 0]], dtype=float)

        _closest, distance, _tid = g.trimesh.proximity.closest_point(mesh, points)
        g.np.testing.assert_allclose(distance[0], 0.5, atol=0.05)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
