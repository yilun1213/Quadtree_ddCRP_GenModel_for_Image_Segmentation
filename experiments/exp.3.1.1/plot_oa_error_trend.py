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


def plot_mean_error_trend(iterations: list[int], mean_errors: list[float], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(iterations, mean_errors, marker="o", color="steelblue", linewidth=2)
    ax.set_xlabel("Gibbs sampling iteration")
    ax.set_ylabel("Mean error over all images")
    ax.set_title("Average Estimation Error Trend")
    ax.grid(True, alpha=0.35)
    ax.set_xticks(iterations)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the average error trend from oa_error_trend.csv."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=os.path.join("outputs", "estimation_results_icm", "oa_error_trend.csv"),
        help="Input CSV path. Default: outputs/estimation_results_icm/oa_error_trend.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output PNG path. Default: next to the CSV as mean_oa_error_trend.png",
    )
    args = parser.parse_args()

    input_csv = os.path.abspath(args.input_csv)
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(input_csv), "mean_oa_error_trend.png")
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    iterations, mean_errors = load_mean_error_by_iteration(input_csv)
    plot_mean_error_trend(iterations, mean_errors, output_path)

    print(f"Input:  {input_csv}")
    print(f"Output: {output_path}")
    print("Mean error by iteration:")
    for iteration, mean_error in zip(iterations, mean_errors):
        print(f"  iter={iteration:>3}: mean_error={mean_error:.6f}")


if __name__ == "__main__":
    main()