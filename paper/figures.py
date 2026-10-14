"""Draw all figures from data.json (run make_data.py first), at their printed size in the ICLR layout (5.5 in text width)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{OUT}/data.json"))
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 5.6,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.4, "legend.frameon": False, "savefig.bbox": "tight",
})
C = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#999999"]
TEX = lambda s: s.replace("$B{=}", "$B=")


def save(fig, name):
    fig.savefig(f"{OUT}/figures/{name}.pdf"); plt.close(fig)


# 1 Pareto (printed at 0.45 of the text width)
fig, ax = plt.subplots(figsize=(2.5, 1.95))
B = np.array(D["pareto"]["B"])
for key, lab, c in [("tide", "TIDE (hazard-scheduled)", C[0]), ("random", "Random selection, same budget", C[6])]:
    m = np.array([x[0] for x in D["pareto"][key]]); s = np.array([x[1] for x in D["pareto"][key]])
    ax.plot(B, m, "-o", ms=2.5, lw=1, color=c, label=lab); ax.fill_between(B, m - s, m + s, color=c, alpha=0.2, lw=0)
for r, mk, c in [("Eager graph", "s", C[1]), ("Access-triggered recons.", "D", C[3]),
                 ("Segment-level (LycheeMem-style)", "^", C[2]), ("Mem0 (rerun)", "v", C[5])]:
    m = [x for x in D["main"] if x["name"] == r][0]
    ax.errorbar(m["cost"], m["locomo"][0], yerr=m["locomo"][1], fmt=mk, ms=3.5, color=c, lw=0.8, capsize=1.5, label=r.replace(" (LycheeMem-style)", ""))
ax.set_xlabel("Compute (relative to eager graph)"); ax.set_ylabel("LoCoMo $J$ (%)")
ax.set_ylim(56.5, 72.6); ax.legend(loc="lower right", handlelength=1.4, borderaxespad=0.2); save(fig, "pareto")

# 2 Horizon (printed at 0.45 of the text width)
fig, ax = plt.subplots(figsize=(2.5, 1.95))
H = np.array(D["horizon"]["H"])
for (k, v), c in zip(D["horizon"]["curves"].items(), [C[6], C[1], C[5], C[0]]):
    a = np.array([[np.nan if x is None else x for x in r] for r in v], float)
    m = np.nanmean(a, 0); s = np.nanstd(a, 0, ddof=1)
    ax.plot(H, m, "-o", ms=2.5, lw=1, color=c, label=TEX(k)); ax.fill_between(H, m - s, m + s, color=c, alpha=0.2, lw=0)
ax.set_xscale("log", base=2); ax.set_xticks(H); ax.set_xticklabels(H)
ax.set_xlabel("Sessions in history $T$"); ax.set_ylabel("Accuracy (%)"); ax.legend(loc="lower left", handlelength=1.4, borderaxespad=0.2)
save(fig, "horizon")

# 3 categories (printed at 0.74 of the text width)
fig, ax = plt.subplots(figsize=(4.1, 1.65))
cats = D["cats"]["names"]; x = np.arange(len(cats)); w = 0.26
for i, ((k, v), c) in enumerate(zip(D["cats"]["methods"].items(), [C[1], C[3], C[0]])):
    ax.bar(x + (i - 1) * w, v["mean"], w, yerr=v["sd"], color=c, capsize=1.5, error_kw=dict(lw=0.6), label=TEX(k))
ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(30, 101)
ax.grid(axis="x", visible=False); ax.legend(loc="upper right", ncol=3); save(fig, "categories")

# 4 hazard training (printed at 0.34 of the text width)
fig, ax = plt.subplots(figsize=(1.9, 1.6))
st = np.array(D["hazard"]["steps"]); a = np.array(D["hazard"]["auc"])
ax.plot(st, a.mean(0), color=C[0], lw=0.9, label="$h_\\phi$ (5 seeds)")
ax.fill_between(st, a.min(0), a.max(0), color=C[0], alpha=0.2, lw=0)
ax.axhline(D["hazard"]["freq_auc"], color=C[1], ls="--", lw=0.8, label="Access frequency")
ax.axhline(D["hazard"]["recency_auc"], color=C[5], ls=":", lw=0.9, label="Recency")
ax.set_xlabel("Replay updates"); ax.set_ylabel("Held-out AUROC"); ax.set_ylim(0.45, 0.9)
ax.legend(loc="lower right"); save(fig, "hazard")

# 5 heatmap (printed at 0.29 of the text width)
fig, ax = plt.subplots(figsize=(1.65, 1.6))
hm = np.array(D["heatmap"]["acc"])
im = ax.imshow(hm, cmap="viridis", aspect="auto", origin="lower")
ax.set_xticks(range(5)); ax.set_xticklabels(D["heatmap"]["horizons"]); ax.set_yticks(range(5)); ax.set_yticklabels(D["heatmap"]["budgets"])
ax.set_xlabel("Horizon $\\Delta$ (sessions)"); ax.set_ylabel("Budget $B$"); ax.grid(False)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{hm[i,j]:.1f}", ha="center", va="center", fontsize=5, color="w" if hm[i, j] < 69 else "k")
save(fig, "heatmap")

# 6 growth (printed at 0.33 of the text width)
fig, ax = plt.subplots(figsize=(1.85, 1.6))
t = D["growth"]["t"]
for (k, v), c in zip(D["growth"]["series"].items(), [C[6], C[1], C[0]]):
    ax.plot(t, v, lw=1, color=c, label=TEX(k))
ax.set_xlabel("Sessions ingested"); ax.set_ylabel("Store size (k tokens)"); ax.legend(loc="upper left", handlelength=1.4)
save(fig, "growth")
print("ok")
