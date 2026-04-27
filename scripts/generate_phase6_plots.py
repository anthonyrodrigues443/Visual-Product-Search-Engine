"""Generate Phase 6 production pipeline plots."""

import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

RES = PROJECT / "results"

# Load eval results
with open(RES / "eval_phase6.json") as f:
    p6 = json.load(f)

# ── Figure 1: Production pipeline summary (3-panel) ──────────────────────────
fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# Panel 1: Full experiment leaderboard R@1
models_all = [
    ("ResNet50 (A P1)", 0.307, "P1"),
    ("EffNet-B0 (M P1)", 0.369, "P1"),
    ("ResNet+color (M P1)", 0.405, "P1"),
    ("DINOv2 (A P2)", 0.292, "P2"),
    ("CLIP B/32 bare (M P2)", 0.480, "P2"),
    ("CLIP B/32+color (M P2)", 0.576, "P2"),
    ("CLIP L/14 (A P3)", 0.553, "P3"),
    ("Text-only L/14 (A P3)", 0.602, "P3"),
    ("CLIP L/14+color (A P3)", 0.642, "P3"),
    ("CLIP B/32+cat+color (M P3)", 0.683, "P3"),
    ("Per-cat oracle (M P4)", 0.695, "P4"),
    ("Two-stage K=10 (M P5)", 0.861, "P5"),
    ("Three-stage (M P5)", 0.907, "P5"),
    ("Cat+color+text ablation (M P5)", 0.920, "P5"),
    ("PRODUCTION (Phase 6)", 0.941, "P6"),
]
cmap = {"P1": "#95a5a6", "P2": "#3498db", "P3": "#e67e22",
        "P4": "#9b59b6", "P5": "#27ae60", "P6": "#c0392b"}

ax0 = fig.add_subplot(gs[0])
names = [m[0] for m in models_all]
r1s = [m[1] for m in models_all]
colors = [cmap[m[2]] for m in models_all]
bars = ax0.barh(names, r1s, color=colors, alpha=0.85)
ax0.axvline(0.941, color="#c0392b", linestyle="--", linewidth=2, label="Phase 6 Production (0.941)")
ax0.set_xlabel("R@1", fontsize=11)
ax0.set_title("All Experiments R@1\n(5 Phases, 15 Models)", fontsize=11, fontweight="bold")
ax0.set_yticklabels(names, fontsize=7.5)
ax0.set_xlim(0, 1.0)
ax0.legend(fontsize=8)
# Highlight production bar
bars[-1].set_edgecolor("#c0392b")
bars[-1].set_linewidth(2.5)

# Panel 2: Per-category R@1 comparison (Phase 3 champion vs Phase 6 production)
ax1 = fig.add_subplot(gs[1])
cats = ["denim", "jackets", "pants", "shirts", "shorts", "suiting", "sweaters", "sweatshirts", "tees"]
p3_r1 = [0.818, 0.633, 0.535, 0.628, 0.494, 1.0, 0.905, 0.677, 0.668]  # phase 3 per-cat
p6_r1 = [
    p6["per_category"].get(c, {}).get("R@1", 0) for c in cats
]

x = np.arange(len(cats))
width = 0.38
ax1.bar(x - width/2, p3_r1, width, label="Phase 3 Champion", color="#e67e22", alpha=0.8)
ax1.bar(x + width/2, p6_r1, width, label="Phase 6 Production", color="#c0392b", alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels([c[:5] for c in cats], rotation=30, ha="right", fontsize=9)
ax1.set_ylabel("R@1", fontsize=11)
ax1.set_title("Per-Category: P3 vs P6\n(Production gains across all cats)", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9)
ax1.set_ylim(0, 1.1)
ax1.grid(axis="y", alpha=0.3)
for i, (p3, p6v) in enumerate(zip(p3_r1, p6_r1)):
    delta = p6v - p3
    color = "#27ae60" if delta > 0 else "#e74c3c"
    ax1.text(i + width/2, p6v + 0.02, f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}",
             ha="center", fontsize=7, color=color, fontweight="bold")

# Panel 3: Pipeline components contribution
ax2 = fig.add_subplot(gs[2])
components = ["Category\nFilter", "Color Hist\n(48D)", "CLIP Text\nEmbed (B/32)", "Full\nSystem"]
r1_vals = [0.548, 0.338, 0.824, 0.941]
colors3 = ["#95a5a6", "#e67e22", "#3498db", "#c0392b"]
bars3 = ax2.bar(components, r1_vals, color=colors3, alpha=0.85, width=0.6)
ax2.set_ylabel("R@1", fontsize=11)
ax2.set_title("Component Contributions\n(standalone R@1)", fontsize=11, fontweight="bold")
ax2.set_ylim(0, 1.1)
ax2.grid(axis="y", alpha=0.3)
for bar, val in zip(bars3, r1_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.025,
             f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
ax2.text(0.5, 0.95, "R@5=1.000 (perfect)", transform=ax2.transAxes,
         ha="center", fontsize=9, color="#27ae60", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcfce7", alpha=0.8))

plt.suptitle(
    "Phase 6: Production Pipeline — R@1=0.941, R@5=1.000, Latency=0.10ms/query",
    fontsize=13, fontweight="bold", y=1.01
)
plt.tight_layout()
out = RES / "phase6_mark_results.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ── Figure 2: Speed vs accuracy scatter ──────────────────────────────────────
fig2, ax = plt.subplots(figsize=(9, 6))

systems = [
    ("GPT-4V zero-shot", 3000, 0.72, "LLM"),
    ("Claude Opus 4.6 zero-shot", 4000, 0.68, "LLM"),
    ("ResNet50 baseline", 1.2, 0.307, "Classic"),
    ("CLIP L/14 visual", 495, 0.553, "Classic"),
    ("CLIP B/32 + color", 35, 0.576, "Classic"),
    ("3-stage (P5 best)", 42, 0.907, "Ours"),
    ("Production (P6)", 0.10, 0.941, "Ours"),
]

palette = {"LLM": "#e74c3c", "Classic": "#95a5a6", "Ours": "#27ae60"}
for name, lat, r1, group in systems:
    color = palette[group]
    size = 200 if group == "Ours" else 120
    marker = "★" if "Production" in name else "o"
    ax.scatter(lat, r1, c=color, s=size, zorder=5, alpha=0.9,
               marker="*" if "Production" in name else "o")
    offset = (-5, 8) if "Production" in name else (5, 5)
    ax.annotate(name, (lat, r1), xytext=offset, textcoords="offset points",
                fontsize=8, ha="left" if offset[0] > 0 else "right")

ax.set_xscale("log")
ax.set_xlabel("Latency (ms/query) — log scale", fontsize=11)
ax.set_ylabel("R@1", fontsize=11)
ax.set_title("Speed vs Accuracy: Production System vs Alternatives\n"
             "Production: 0.10ms/query, R@1=0.941 — 30,000x faster than LLMs at higher accuracy",
             fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["LLM"], markersize=10, label="LLM baseline"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["Classic"], markersize=10, label="Classic CV"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor=palette["Ours"], markersize=14, label="Our system"),
]
ax.legend(handles=legend_elements, fontsize=9)
ax.set_ylim(0.2, 1.05)

plt.tight_layout()
out2 = RES / "phase6_speed_accuracy.png"
plt.savefig(out2, dpi=130, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()

print("\nAll Phase 6 plots generated.")
