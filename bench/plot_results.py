#!/usr/bin/env python3
# /// script
# dependencies = [
#   "altair",
#   "pandas",
# ]
# ///
"""Generate results.html altair plot from results.tsv."""
from pathlib import Path
import altair as alt
import pandas as pd

RESULTS_TSV = Path(__file__).parent.parent / "results.tsv"
OUTPUT_HTML = Path(__file__).parent.parent / "results.html"

df = pd.read_csv(RESULTS_TSV, sep="\t")
df = df.reset_index().rename(columns={"index": "experiment"})

# Assign sequential index to "keep" experiments for baseline tracking
df["experiment_label"] = df["commit"].str[:7] + ": " + df["description"].str[:40]

# Color by status
color_map = {"keep": "#2ca02c", "discard": "#d62728", "crash": "#7f7f7f"}
df["color"] = df["status"].map(color_map).fillna("#1f77b4")

base = alt.Chart(df).encode(
    x=alt.X("experiment:Q", title="Experiment #"),
    tooltip=["commit", "status", "description", "wall_time_ms", "peak_memory_mb", "fitness"],
)

fitness_chart = base.mark_line(point=True).encode(
    y=alt.Y("fitness:Q", title="Fitness (lower=better)", scale=alt.Scale(zero=False)),
    color=alt.Color("status:N", scale=alt.Scale(
        domain=["keep", "discard", "crash"],
        range=["#2ca02c", "#d62728", "#7f7f7f"]
    )),
).properties(title="Composite Fitness Over Experiments", width=800, height=250)

wall_time_chart = base.mark_line(point=True, color="#1f77b4").encode(
    y=alt.Y("wall_time_ms:Q", title="Wall Time (ms)", scale=alt.Scale(zero=False)),
).properties(title="Wall Time (ms)", width=800, height=200)

memory_chart = base.mark_line(point=True, color="#ff7f0e").encode(
    y=alt.Y("peak_memory_mb:Q", title="Peak Memory (MB)", scale=alt.Scale(zero=False)),
).properties(title="Peak Memory (MB)", width=800, height=200)

chart = alt.vconcat(fitness_chart, wall_time_chart, memory_chart).resolve_scale(x="shared")
chart.save(str(OUTPUT_HTML))
print(f"Saved {OUTPUT_HTML}")
