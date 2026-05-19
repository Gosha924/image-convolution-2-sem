import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_results():
    results_path = Path("benchmark_results/benchmark_results.json")
    if not results_path.exists():
        raise FileNotFoundError("Результаты не найдены. Сначала запустите run_benchmark.py")
    with open(results_path, "r") as f:
        return json.load(f)


def plot_results(results):
    out_dir = Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)
    all_sizes = sorted(set(r["size"] for r in results))
    size_labels = [str(s) for s in all_sizes]

    grouped = defaultdict(
        lambda: {"sizes": [], "my_mean": [], "my_std": [], "cv_mean": [], "cv_std": []}
    )
    for r in results:
        key = (r["kernel"], r["edge_mode"], r["image_type"])
        grouped[key]["sizes"].append(r["size"])
        grouped[key]["my_mean"].append(r["my_mean_ms"])
        grouped[key]["my_std"].append(r["my_std_ms"])
        grouped[key]["cv_mean"].append(r["cv_mean_ms"])
        grouped[key]["cv_std"].append(r["cv_std_ms"])
    for key in grouped:
        indices = np.argsort(grouped[key]["sizes"])
        for field in ["sizes", "my_mean", "my_std", "cv_mean", "cv_std"]:
            grouped[key][field] = [grouped[key][field][i] for i in indices]

    n_plots = len(grouped)
    cols = 2
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, (key, data) in zip(axes, grouped.items()):
        kernel, edge, img_type = key
        sizes = data["sizes"]
        ax.errorbar(
            sizes,
            data["my_mean"],
            yerr=data["my_std"],
            label="Educational (Python)",
            marker="o",
            capsize=5,
            linestyle="-",
            linewidth=2,
        )
        ax.errorbar(
            sizes,
            data["cv_mean"],
            yerr=data["cv_std"],
            label="OpenCV (C++)",
            marker="s",
            capsize=5,
            linestyle="--",
            linewidth=2,
        )
        ax.set_xticks(all_sizes)
        ax.set_xticklabels(size_labels)
        ax.set_xlabel("Image size (pixels)")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{kernel}\n{edge} | {img_type}")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.6)

    for i in range(len(axes) - n_plots):
        axes[-i - 1].set_visible(False)

    plt.suptitle("Performance comparison: Educational vs OpenCV", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_result.png", dpi=150)
    plt.show()
    print(f"График сохранён в {out_dir / 'benchmark_result.png'}")


def print_summary_table(results):
    speeds = {}
    for r in results:
        key = (r["kernel"], r["edge_mode"], r["image_type"], r["size"])
        speeds[key] = r["speedup"]

    print("\nТаблица ускорений (Educational / OpenCV)")
    sizes = sorted(set(r["size"] for r in results))
    combos = sorted(set((r["kernel"], r["edge_mode"], r["image_type"]) for r in results))

    header = ["Combination"] + [f"{s}x{s}" for s in sizes]
    print(f"{header[0]:<40}", " ".join(f"{h:>10}" for h in header[1:]))
    print("-" * 90)

    for combo in combos:
        kernel, edge, img_type = combo
        label = f"{kernel}, {edge}, {img_type}"
        row = [label]
        for s in sizes:
            sp = speeds.get((kernel, edge, img_type, s), None)
            row.append(f"{sp:>8.1f}x" if sp else "N/A")
        print(f"{row[0]:<40}", " ".join(row[1:]))


def main():
    results = load_results()
    if results:
        plot_results(results)
        print_summary_table(results)
    else:
        print("Нет данных для визуализации.")


if __name__ == "__main__":
    main()
