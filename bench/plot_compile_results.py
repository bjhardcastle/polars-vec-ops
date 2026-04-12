#!/usr/bin/env python3
# /// script
# dependencies = [
#   "altair",
#   "pandas",
# ]
# ///
"""Generate compile_results.html altair plot from compile_results.tsv."""
from pathlib import Path
import altair as alt
import pandas as pd

RESULTS_TSV = Path(__file__).parent.parent / "compile_results.tsv"
OUTPUT_HTML = Path(__file__).parent.parent / "compile_results.html"

df = pd.read_csv(RESULTS_TSV, sep="\t")
df = df.reset_index().rename(columns={"index": "experiment"})

df["experiment_label"] = df["commit"].str[:7] + ": " + df["description"].str[:50]

color_map = {"keep": "#2ca02c", "discard": "#d62728", "crash": "#7f7f7f"}
df["color"] = df["status"].map(color_map).fillna("#1f77b4")

base = alt.Chart(df).encode(
    x=alt.X("experiment:Q", title="Experiment #"),
    tooltip=["commit", "status", "description", "clean_compile_s", "incremental_compile_s", "wall_time_ms", "fitness"],
)

fitness_chart = base.mark_line(point=True).encode(
    y=alt.Y("fitness:Q", title="Fitness (lower=better)", scale=alt.Scale(zero=False)),
    color=alt.Color("status:N", scale=alt.Scale(
        domain=["keep", "discard", "crash"],
        range=["#2ca02c", "#d62728", "#7f7f7f"]
    )),
).properties(title="Composite Compile-Time Fitness Over Experiments", width=800, height=250)

clean_chart = base.mark_line(point=True, color="#1f77b4").encode(
    y=alt.Y("clean_compile_s:Q", title="Clean Build (s)", scale=alt.Scale(zero=False)),
).properties(title="Clean Build Time (s)", width=800, height=200)

incremental_chart = base.mark_line(point=True, color="#ff7f0e").encode(
    y=alt.Y("incremental_compile_s:Q", title="Incremental Build (s)", scale=alt.Scale(zero=False)),
).properties(title="Incremental Build Time (s)", width=800, height=200)

wall_time_chart = base.mark_line(point=True, color="#9467bd").encode(
    y=alt.Y("wall_time_ms:Q", title="Runtime wall_time (ms)", scale=alt.Scale(zero=False)),
).properties(title="Runtime Wall Time (ms, regression guard)", width=800, height=200)

chart = alt.vconcat(fitness_chart, clean_chart, incremental_chart, wall_time_chart).resolve_scale(x="shared")
chart.save(str(OUTPUT_HTML))
print(f"Saved {OUTPUT_HTML}")
