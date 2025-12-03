import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

measurements_dir = Path(__file__).resolve().parent.parent
csv_path = measurements_dir / "accuracy.csv"
df = pd.read_csv(csv_path)
df["time"] = pd.to_datetime(df["time"])

print(f"Total datapoints: {len(df)}")
print("\nDataframe:")
print(df)

# Plot 1 (Accuracy scores over time)
fig1, ax1 = plt.subplots(figsize=(12, 6))
fig1.suptitle("Accuracy Metric Results Analysis", fontsize=16, fontweight="bold")

if len(df) > 1:
    ax1.plot(
        df["time"], df["score"], marker="o", linestyle="-", linewidth=2, markersize=8
    )
    ax1.set_xlabel("Time", fontsize=11)
else:
    ax1.scatter(df["time"], df["score"], s=200, edgecolors="black")
    ax1.set_xlabel("Time", fontsize=11)

ax1.set_ylabel("Accuracy Score", fontsize=11)
ax1.set_title("Accuracy Over Time", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

fig1_path = measurements_dir / "accuracy_over_time.png"

plt.tight_layout()
plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
print("Saved plot 1 to: " + str(fig1_path))
plt.close(fig1)

# Plot 2 (Summary statistics)
fig2 = plt.figure(figsize=(8, 6))
fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax2 = fig2.add_subplot(111)
ax2.axis("off")


stats_text = f"""
Summary Statistics:
{'='*40}

Total Measurements: {len(df)}


Accuracy Score:
Mean: {df['score'].mean():.4f}
Median: {df['score'].median():.4f}
Std Dev: {df['score'].std():.4f}
Min: {df['score'].min():.4f}
Max: {df['score'].max():.4f}


Time Range:
First: {df['time'].min().strftime('%Y-%m-%d %H:%M:%S')}
Last: {df['time'].max().strftime('%Y-%m-%d %H:%M:%S')}
"""

ax2.text(
    0.5,
    0.5,
    stats_text,
    fontsize=12,
    family="monospace",
    verticalalignment="center",
    horizontalalignment="center",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)

fig2_path = measurements_dir / "accuracy_summary.png"

fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
print("Saved plot 2 to: " + str(fig2_path))
plt.close(fig2)
