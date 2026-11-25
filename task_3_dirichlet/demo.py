"""Benchmark the compiled C solver and plot its numerical solution."""

import hashlib
import io
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DIRECTORY = Path(__file__).resolve().parent
SOLVER = DIRECTORY / ("dirichlet.exe" if os.name == "nt" else "dirichlet")
THREADS = (1, 2, 4)
BLOCK = 16
TOLERANCE = 1e-8
BOUNDARY = 20


def benchmark(nodes):
    samples = {p: [] for p in THREADS}
    reference = None
    for repeat in range(3):
        # Rotate the run order so each thread count goes first once.
        for p in THREADS[repeat:] + THREADS[:repeat]:
            result = subprocess.run(
                [str(SOLVER), str(nodes), str(BLOCK), str(TOLERANCE), str(BOUNDARY)],
                env={**os.environ, "OMP_NUM_THREADS": str(p), "OMP_DYNAMIC": "FALSE",
                     "OMP_THREAD_LIMIT": str(p)},
                capture_output=True, check=True,
            )
            stats = dict(item.split("=") for item in result.stderr.decode().split())
            samples[p].append(float(stats["solve_seconds"]))
            signature = (hashlib.sha256(result.stdout).digest(), stats["sweeps"])
            if reference is None:
                reference = signature
                field = np.loadtxt(io.BytesIO(result.stdout))
            elif signature != reference:
                raise ValueError(f"Temperature field or sweep count changed for {p} threads")

    x = np.linspace(0, 1, nodes)
    exact = BOUNDARY + np.outer(x * (1 - x), x * (1 - x))
    error = np.max(np.abs(field - exact))
    if field.shape != exact.shape or not np.allclose(field, exact, atol=5e-5, rtol=0):
        raise ValueError(f"Analytical check failed: max error {error:g}")
    times = [median(samples[p]) for p in THREADS]
    results = []
    for p, seconds in zip(THREADS, times):
        speedup = times[0] / seconds
        results.append({
            "nodes_per_side": nodes,
            "threads": p,
            "sweeps": int(reference[1]),
            "solve_seconds": samples[p],
            "median_seconds": seconds,
            "speedup": speedup,
            "efficiency_percent": 100 * speedup / p,
        })
        print(f"{nodes:5} {p:7} {seconds:9.4f} {speedup:8.2f} {100*speedup/p:10.1f}")
    return field, results, error


def main():
    fig, (plate, chart) = plt.subplots(1, 2, figsize=(11, 5.6))
    fig.subplots_adjust(top=.79, bottom=.23, left=.07, right=.97, wspace=.36)
    print("nodes threads  median_s  speedup efficiency_%", flush=True)
    measurements = []
    for nodes in (65, 129):
        field, results, error = benchmark(nodes)
        measurements.extend(results)
        chart.plot(THREADS, [r["speedup"] for r in results], marker="o", linewidth=2,
                   label=f"{nodes} × {nodes}")
    report = {
        "platform": platform.platform(),
        "block_size": BLOCK,
        "tolerance": TOLERANCE,
        "boundary_temperature": BOUNDARY,
        "measurements": measurements,
    }
    (DIRECTORY / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n",
                                               encoding="utf-8")
    mesh = plate.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap="inferno")
    fig.colorbar(mesh, ax=plate, format="%.3f", shrink=.85)
    plate.set(xlabel="x", ylabel="y", title=f"Температура · {nodes} × {nodes}")
    chart.axhline(1, linestyle="--", color="gray", linewidth=1)
    chart.set(xlabel="Потоки OpenMP", ylabel="Ускорение  T₁ / Tₚ",
              title="Ускорение относительно 1 потока", xticks=THREADS, ylim=(0, None))
    chart.grid(axis="y", alpha=.25)
    chart.legend()
    fig.suptitle("Задача Дирихле · блочный метод Гаусса–Зейделя", fontsize=17)
    fig.text(.07, .095, f"В центре: {field[nodes//2, nodes//2]:.6f}   ·   "
             f"Точное значение: 20.0625\nМаксимальная ошибка по сетке: {error:.2e}")
    fig.text(.56, .095, f"{platform.system()} · медиана трёх запусков\n"
             f"Блок {BLOCK} × {BLOCK} · порог {TOLERANCE:g} · время без вывода")
    fig.savefig(DIRECTORY / "demo.png", dpi=150)


if __name__ == "__main__":
    main()
