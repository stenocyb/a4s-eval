import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../../data/measures/accuracy.csv")
df["time"] = pd.to_datetime(df["time"])

print(f"Total datapoints: {len(df)}")
print("\nDataframe:")
print(df)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle("Accuracy Metric Results Analysis", fontsize=16, fontweight="bold")

# plot 1: accuracy scores over time
ax1 = axes[0]
if len(df) > 1:
    ax1.plot(
        df["time"], df["score"], marker="o", linestyle="-", linewidth=2, markersize=8
    )
    ax1.set_xlabel("Time", fontsize=11)
else:
    ax1.scatter(df["time"], df["score"], s=200, c="blue", alpha=0.6, edgecolors="black")
    ax1.set_xlabel("Time", fontsize=11)

ax1.set_ylabel("Accuracy Score", fontsize=11)
ax1.set_title("Accuracy Over Time", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

# plot 2: summary statistics
ax2 = axes[1]
ax2.axis("off")

stats_text = f"""
Summary Statistics:
{'='*40}

Total Measurements: {len(df)}

Accuracy Score:
  Mean:    {df['score'].mean():.4f}
  Median:  {df['score'].median():.4f}
  Std Dev: {df['score'].std():.4f}
  Min:     {df['score'].min():.4f}
  Max:     {df['score'].max():.4f}

Time Range:
  First:   {df['time'].min().strftime('%Y-%m-%d %H:%M:%S')}
  Last:    {df['time'].max().strftime('%Y-%m-%d %H:%M:%S')}
"""

ax2.text(
    0.1,
    0.5,
    stats_text,
    fontsize=11,
    family="monospace",
    verticalalignment="center",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)

plt.tight_layout()
plt.savefig("../../data/measures/accuracy_analysis.png", dpi=300, bbox_inches="tight")
print("\nPlot saved to: tests/data/measures/accuracy_analysis.png")
plt.show()
