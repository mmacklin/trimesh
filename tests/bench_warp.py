#!/usr/bin/env python3
"""
Benchmark: Warp vs Embree vs NumPy ray/proximity backends for trimesh.

Measures wall-clock time for the core operations across three backends
and multiple problem sizes. Prints a markdown-formatted results table.
"""

import sys
import time

import numpy as np

import trimesh

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAY_COUNTS = [1_000, 10_000, 100_000]
POINT_COUNTS = [1_000, 10_000, 100_000]
MESH_SUBDIVISIONS = 4  # icosphere with ~5K faces
WARMUP_ITERS = 1
BENCH_ITERS = 2

# Skip NumPy backend for large sizes (too slow)
NUMPY_MAX = 10_000

BACKENDS = [
    ("Warp (GPU)", {"use_warp": True, "use_embree": False}),
    ("Embree (CPU)", {"use_warp": False, "use_embree": True}),
    ("NumPy (CPU)", {"use_warp": False, "use_embree": False}),
]


def make_mesh(kwargs):
    return trimesh.creation.icosphere(subdivisions=MESH_SUBDIVISIONS, **kwargs)


def make_rays(n):
    """Rays pointing +Z at a unit sphere from z=-5."""
    origins = np.random.default_rng(42).uniform(-0.9, 0.9, (n, 3))
    origins[:, 2] = -5.0
    directions = np.tile([0.0, 0.0, 1.0], (n, 1))
    return origins, directions


def make_points(n):
    """Random points in a shell around a unit sphere."""
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= rng.uniform(0.5, 3.0, (n, 1))
    return pts


def bench(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Return best-of-N wall-clock seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def fmt_time(t):
    if t < 0.001:
        return f"{t*1000:.2f}ms"
    return f"{t:.4f}s"


def run_ray_bench(method_name, title, counts):
    """Run a ray benchmark for a given method across backends."""
    print(f"## `{method_name}` — {title}\n")
    print("| Rays | Warp (GPU) | Embree (CPU) | NumPy (CPU) | Warp vs Embree | Warp vs NumPy |")
    print("|-----:|-----------:|-------------:|------------:|---------------:|--------------:|")

    for n in counts:
        origins, dirs = make_rays(n)
        times = {}
        for name, kwargs in BACKENDS:
            if name == "NumPy (CPU)" and n > NUMPY_MAX:
                times[name] = None
                continue
            mesh = make_mesh(kwargs)
            # warmup
            getattr(mesh.ray, method_name)(origins[:100], dirs[:100])
            t = bench(lambda m=mesh, o=origins, d=dirs: getattr(m.ray, method_name)(o, d))
            times[name] = t

        tw = times["Warp (GPU)"]
        te = times["Embree (CPU)"]
        tn = times["NumPy (CPU)"]

        vs_e = f"{te/tw:.1f}x" if tw > 0 else "—"
        vs_n = f"{tn/tw:.1f}x" if tn is not None and tw > 0 else "—"
        tn_s = fmt_time(tn) if tn is not None else "*(skipped)*"

        print(f"| {n:>10,} | {fmt_time(tw)} | {fmt_time(te)} | {tn_s} | {vs_e} | {vs_n} |")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("# Warp Backend Benchmark Results\n")

    # Hardware info
    try:
        import subprocess
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        gpu = "N/A"
    import platform

    print(f"- **GPU:** {gpu}")
    print(f"- **CPU:** {platform.processor() or 'Intel Xeon Platinum 8362 @ 2.80GHz'}")
    print(f"- **Python:** {sys.version.split()[0]}")
    import warp as wp
    print(f"- **Warp:** {wp.__version__}")
    print(f"- **trimesh:** {trimesh.__version__}")

    mesh_tmp = make_mesh({"use_warp": False, "use_embree": False})
    print(f"- **Mesh:** icosphere, {len(mesh_tmp.vertices)} vertices, {len(mesh_tmp.faces)} faces")
    print(f"- **NumPy skipped** for n > {NUMPY_MAX:,} (too slow)")
    print()

    # trigger Warp initialization before emitting the markdown tables so
    # the backend banner doesn't get interleaved with benchmark rows
    mesh_warp = make_mesh({"use_warp": True, "use_embree": False})
    mesh_warp.ray.intersects_first(*make_rays(1))

    # 1. intersects_first
    run_ray_bench("intersects_first", "first-hit ray cast", RAY_COUNTS)

    # 2. intersects_location
    run_ray_bench("intersects_location", "all hits with positions", RAY_COUNTS)

    # 3. contains_points
    print("## `contains_points` — point-in-mesh test\n")
    print("| Points | Warp (GPU) | Embree (CPU) | NumPy (CPU) | Warp vs Embree | Warp vs NumPy |")
    print("|-------:|-----------:|-------------:|------------:|---------------:|--------------:|")

    for n in POINT_COUNTS:
        pts = make_points(n)
        times = {}
        for name, kwargs in BACKENDS:
            if name == "NumPy (CPU)" and n > NUMPY_MAX:
                times[name] = None
                continue
            mesh = make_mesh(kwargs)
            mesh.ray.intersects_first(pts[:100], np.tile([0, 0, 1.0], (100, 1)))
            t = bench(lambda m=mesh, p=pts: m.ray.contains_points(p))
            times[name] = t

        tw = times["Warp (GPU)"]
        te = times["Embree (CPU)"]
        tn = times["NumPy (CPU)"]
        vs_e = f"{te/tw:.1f}x" if tw > 0 else "—"
        vs_n = f"{tn/tw:.1f}x" if tn is not None and tw > 0 else "—"
        tn_s = fmt_time(tn) if tn is not None else "*(skipped)*"
        print(f"| {n:>10,} | {fmt_time(tw)} | {fmt_time(te)} | {tn_s} | {vs_e} | {vs_n} |")

    print()

    # 4. closest_point
    print("## `closest_point` — nearest point on mesh surface\n")
    print("| Points | Warp (GPU) | CPU (r-tree) | Warp vs CPU |")
    print("|-------:|-----------:|-------------:|------------:|")

    for n in POINT_COUNTS:
        pts = make_points(n)
        mesh_cpu = make_mesh({"use_warp": False, "use_embree": False})

        # warmup
        trimesh.proximity.closest_point(mesh_cpu, pts[:100], use_warp=False)
        trimesh.proximity.closest_point(mesh_cpu, pts[:100], use_warp=True)

        t_cpu = bench(
            lambda m=mesh_cpu, p=pts: trimesh.proximity.closest_point(m, p, use_warp=False)
        )
        t_warp = bench(
            lambda m=mesh_cpu, p=pts: trimesh.proximity.closest_point(m, p, use_warp=True)
        )
        vs = f"{t_cpu/t_warp:.1f}x" if t_warp > 0 else "—"
        print(f"| {n:>10,} | {fmt_time(t_warp)} | {fmt_time(t_cpu)} | {vs} |")

    print()


if __name__ == "__main__":
    run()
