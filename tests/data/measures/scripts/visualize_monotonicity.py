import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

measurements_dir = Path(__file__).resolve().parent.parent
csv_path = measurements_dir / "monotonicity.csv"
df = pd.read_csv(csv_path)
df["time"] = pd.to_datetime(df["time"])

# plot 1 (monotonicity scores over time)
fig1, ax1 = plt.subplots(figsize=(12, 6))
fig1.suptitle("Monotonicity Metric Results Analysis", fontsize=16, fontweight="bold")

if len(df) > 1:
    ax1.plot(
        df["time"], df["score"], marker="o", linestyle="-", linewidth=2, markersize=8
    )
    ax1.set_xlabel("Time", fontsize=11)
else:
    ax1.scatter(df["time"], df["score"], s=200, edgecolors="black")
    ax1.set_xlabel("Time", fontsize=11)

ax1.set_ylabel("Monotonicity Score", fontsize=11)
ax1.set_title("Monotonicity Over Time", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

fig1_path = measurements_dir / "monotonicity_over_time.png"

plt.tight_layout()
plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
print("Saved plot 1 to: " + str(fig1_path))
plt.close(fig1)

# plot 2 (summary statistics)
fig2 = plt.figure(figsize=(8, 6))
fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax2 = fig2.add_subplot(111)
ax2.axis("off")

stats_text = f"""
Summary Statistics:
{'='*40}

Total Measurements: {len(df)}


Monotonicity Score:
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

fig2_path = measurements_dir / "monotonicity_summary.png"

fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
print("Saved plot 2 to: " + str(fig2_path))
plt.close(fig2)

# plot 3 (distribution histogram)
fig3, ax3 = plt.subplots(figsize=(10, 6))
fig3.suptitle("Monotonicity Score Distribution", fontsize=16, fontweight="bold")

ax3.hist(df["score"], bins=15, edgecolor="black", alpha=0.7, color="skyblue")
ax3.axvline(
    df["score"].mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {df['score'].mean():.4f}",
)
ax3.axvline(
    df["score"].median(),
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Median: {df['score'].median():.4f}",
)
ax3.set_xlabel("Monotonicity Score", fontsize=11)
ax3.set_ylabel("Frequency", fontsize=11)
ax3.set_title("Score Distribution", fontsize=12, fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3, axis="y")

fig3_path = measurements_dir / "monotonicity_distribution.png"

plt.tight_layout()
plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
print("Saved plot 3 to: " + str(fig3_path))
plt.close(fig3)

# plot 4 (rolling mean and score)
fig4, (ax4a) = plt.subplots(1, 1, figsize=(12, 10))
fig4.suptitle(
    "Monotonicity Score Trends and Volatility", fontsize=16, fontweight="bold"
)

window = min(5, len(df))
df["rolling_mean"] = df["score"].rolling(window=window, center=True).mean()
df["rolling_std"] = df["score"].rolling(window=window, center=True).std()

ax4a.plot(
    df["time"],
    df["score"],
    marker="o",
    linestyle="-",
    linewidth=1.5,
    markersize=6,
    alpha=0.6,
    label="Score",
    color="steelblue",
)
ax4a.plot(
    df["time"],
    df["rolling_mean"],
    linewidth=2.5,
    label=f"Rolling Mean (window={window})",
    color="darkred",
)
ax4a.fill_between(
    df["time"],
    df["rolling_mean"] - df["rolling_std"],
    df["rolling_mean"] + df["rolling_std"],
    alpha=0.2,
    color="red",
    label="±1 Std Dev",
)
ax4a.axhline(
    df["score"].mean(),
    color="green",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label="Overall Mean",
)
ax4a.set_ylabel("Monotonicity Score", fontsize=11)
ax4a.set_title("Score with Rolling Statistics", fontsize=12, fontweight="bold")
ax4a.legend(loc="best")
ax4a.grid(True, alpha=0.3)
ax4a.set_ylim([0, 1])
plt.setp(ax4a.xaxis.get_majorticklabels(), rotation=45, ha="right")

fig4_path = measurements_dir / "monotonicity_trends.png"

plt.tight_layout()
plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
print("Saved plot 4 to: " + str(fig4_path))
plt.close(fig4)
