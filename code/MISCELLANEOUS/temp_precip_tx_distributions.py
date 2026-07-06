"""
Quick distribution plots for TX30, TX35, TX40, P_above_10, P_above_20, TN0, TNm5, TNm10
Source: 3emp_3obs regression-ready data (log_lab_prod_to)
"""

import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

FILE = (
    r"C:\Users\Paroli Diego\CMCC Dropbox\Diego Paroli\orbis"
    r"\data\6.regression_ready_data\log_lab_prod_to"
    r"\3emp_3obs__CTRL=lag(operating_p_l_ebit_)-lag(total_assets)"
    r"-lag(fixed_assets)-lag(number_of_employees)-date_of_incorporation__final_data.parquet"
)

COLS = ["TX30", "TX35", "TX40", "P_10_to_20", "P_above_20", "TN0", "TNm5", "TNm10"]

# ---------------------------------------------------------------------------
# LOAD (columns only — avoids reading 7 GB into memory)
# ---------------------------------------------------------------------------

print("Reading columns from parquet...")
df = pq.read_table(FILE, columns=COLS).to_pandas()
print(f"  Loaded {len(df):,} rows")

df["P_above_10"] = df["P_10_to_20"] + df["P_above_20"]

# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

VARS = [
    # row 1 — hot days
    ("TX30",       "TX30 — days with T$_{max}$ ≥ 30 °C",  "#e07b39"),
    ("TX35",       "TX35 — days with T$_{max}$ ≥ 35 °C",  "#d35400"),
    ("TX40",       "TX40 — days with T$_{max}$ ≥ 40 °C",  "#c0392b"),
    # row 2 — precipitation (axes[5] will be hidden)
    ("P_above_10", "P≥10 mm — days with precip ≥ 10 mm",  "#2980b9"),
    ("P_above_20", "P≥20 mm — days with precip ≥ 20 mm",  "#1a5276"),
    # row 3 — cold nights
    ("TN0",        "TN0 — days with T$_{min}$ < 0 °C",    "#5dade2"),
    ("TNm5",       "TNm5 — days with T$_{min}$ < −5 °C",  "#1f618d"),
    ("TNm10",      "TNm10 — days with T$_{min}$ < −10 °C","#7d3c98"),
]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
axes[5].set_visible(False)  # empty slot between precip row and TN row

PLOT_SLOTS = [0, 1, 2, 3, 4, 6, 7, 8]  # skip slot 5

for slot, (col, title, color) in zip(PLOT_SLOTS, VARS):
    ax = axes[slot]
    series = df[col].dropna()

    bins = np.linspace(series.min(), series.max(), 60)

    ax.hist(series, bins=bins, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)

    ax.axvline(series.mean(),   color="black",  lw=1.2, ls="--", label=f"mean={series.mean():.1f}")
    ax.axvline(series.median(), color="dimgrey", lw=1.0, ls=":",  label=f"median={series.median():.1f}")

    ax.set_yscale("log")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("days / year", fontsize=9)
    ax.set_ylabel("count (log)", fontsize=9)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax.text(0.98, 0.95, f"n={len(series):,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="grey")

fig.suptitle("Distribution of precipitation & temperature threshold-day variables\n3emp–3obs regression-ready sample",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()

OUT = r"C:\Users\Paroli Diego\Documents\climate-indicators-grid\code\MISCELLANEOUS\temp_precip_tx_distributions.png"
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved to {OUT}")
plt.show()