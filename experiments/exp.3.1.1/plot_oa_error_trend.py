import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_mean_error_by_iteration(csv_path: str) -> tuple[list[int], list[float]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not reader.fieldnames:
            raise ValueError(f"CSV header is missing: {csv_path}")

        iteration_columns = [name for name in reader.fieldnames if name != "image"]
        if not iteration_columns:
            raise ValueError(f"No iteration columns found in: {csv_path}")

        values_by_iteration: dict[int, list[float]] = {int(col): [] for col in iteration_columns}
        for row in reader:
            for col in iteration_columns:
                value = row.get(col, "")
                if value is None or value == "":
                    continue
                values_by_iteration[int(col)].append(float(value))

    iterations = sorted(values_by_iteration)
    mean_errors = []
    for iteration in iterations:
        values = values_by_iteration[iteration]
        if not values:
            raise ValueError(f"No values found for iteration {iteration} in: {csv_path}")
        mean_errors.append(sum(values) / len(values))

    return iterations, mean_errors


def plot_mean_error_trends(
    gibbs_iterations: list[int],
    gibbs_mean_errors: list[float],
    icm_iterations: list[int],
    icm_mean_errors: list[float],
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(
        gibbs_iterations,
        gibbs_mean_errors,
        marker="o",
        color="steelblue",
        linewidth=2,
        label="IV-A",
    )
    ax.plot(
        icm_iterations,
        icm_mean_errors,
        marker="s",
        color="tomato",
        linewidth=2,
        label="IV-B",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average loss among test images")
    ax.grid(True, alpha=0.35)
    all_iterations = sorted(set(gibbs_iterations + icm_iterations))
    ax.set_xticks(all_iterations)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Gibbs and ICM average error trends on one graph."
    )
    parser.add_argument(
        "gibbs_csv",
        nargs="?",
        default=os.path.join("outputs", "estimation_results_gibbs", "oa_error_trend.csv"),
        help="Gibbs OA CSV path. Default: outputs/estimation_results_gibbs/oa_error_trend.csv",
    )
    parser.add_argument(
        "icm_csv",
        nargs="?",
        default=os.path.join("outputs", "estimation_results_icm", "oa_error_trend.csv"),
        help="ICM OA CSV path. Default: outputs/estimation_results_icm/oa_error_trend.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output PNG path. Default: outputs/oa_error_trend_comparison.png",
    )
    args = parser.parse_args()

    gibbs_csv = os.path.abspath(args.gibbs_csv)
    icm_csv = os.path.abspath(args.icm_csv)
    if not os.path.exists(gibbs_csv):
        raise FileNotFoundError(f"Gibbs CSV not found: {gibbs_csv}")
    if not os.path.exists(icm_csv):
        raise FileNotFoundError(f"ICM CSV not found: {icm_csv}")

    output_path = args.output
    if output_path is None:
        output_path = os.path.join("outputs", "oa_error_trend_comparison.png")
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    gibbs_iterations, gibbs_mean_errors = load_mean_error_by_iteration(gibbs_csv)
    icm_iterations, icm_mean_errors = load_mean_error_by_iteration(icm_csv)
    plot_mean_error_trends(
        gibbs_iterations,
        gibbs_mean_errors,
        icm_iterations,
        icm_mean_errors,
        output_path,
    )

    print(f"Gibbs input: {gibbs_csv}")
    print(f"ICM input:   {icm_csv}")
    print(f"Output: {output_path}")
    print("Gibbs mean error by iteration:")
    for iteration, mean_error in zip(gibbs_iterations, gibbs_mean_errors):
        print(f"  iter={iteration:>3}: mean_error={mean_error:.6f}")
    print("ICM mean error by iteration:")
    for iteration, mean_error in zip(icm_iterations, icm_mean_errors):
        print(f"  iter={iteration:>3}: mean_error={mean_error:.6f}")


if __name__ == "__main__":
    main()