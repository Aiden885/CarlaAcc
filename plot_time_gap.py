import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_PATH = Path("speed_data_integrated.csv")
CSV_OUT = Path("time_gap_series.csv")
PLOT_OUT = Path("time_gap_plot.png")


def load_time_mode_samples():
    """Return list of (cycle_idx, desired_gap, actual_gap) taken in TIME mode."""
    samples = []
    with LOG_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Check if in TIME mode
                control_mode = row["Control_Mode"].strip()
                if control_mode != "TIME":
                    continue  # keep only TIME-mode cycles

                # Get desired time gap
                desired_gap = float(row["G2_Setting"])

                # Calculate actual time gap from distance and speed
                # actual_time_gap = distance / speed
                actual_distance = float(row["Actual_Distance(m)"])
                ego_speed_kmh = float(row["Ego_Speed(km/h)"])
                ego_speed_ms = ego_speed_kmh / 3.6

                # Skip if speed is too low (avoid division by zero)
                if ego_speed_ms < 0.1:
                    continue

                actual_gap = actual_distance / ego_speed_ms

            except (KeyError, ValueError, ZeroDivisionError) as e:
                continue

            # Filter out unreasonably large values (e.g., > 200 seconds)
            # This indicates invalid data
            if abs(actual_gap) > 200.0:
                continue  # skip invalid samples

            samples.append((len(samples), desired_gap, actual_gap))
    return samples


def main():
    if not LOG_PATH.exists():
        raise FileNotFoundError(
            f"{LOG_PATH} not found. Run the ACC scenario once to generate it."
        )

    samples = load_time_mode_samples()
    if not samples:
        print("No TIME-mode samples found in log. Nothing to plot.")
        return

    # Save the series for any other analysis
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cycle_index", "desired_time_gap_s", "actual_time_gap_s"])
        writer.writerows(samples)

    # Build plot
    cycle_idx = [s[0] for s in samples]
    desired = [s[1] for s in samples]
    actual = [s[2] for s in samples]

    plt.figure(figsize=(10, 4))
    plt.plot(cycle_idx, desired, label="Desired Time Gap (s)", linewidth=2)
    plt.plot(cycle_idx, actual, label="Actual Time Gap (s)", linewidth=1)
    plt.xlabel("Cycle Index (TIME mode only)")
    plt.ylabel("Time Gap (s)")
    plt.title("Desired vs Actual Time Gap")
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=150)

    print(f"Wrote {len(samples)} samples to {CSV_OUT}")
    print(f"Plot saved to {PLOT_OUT}")


if __name__ == "__main__":
    main()
