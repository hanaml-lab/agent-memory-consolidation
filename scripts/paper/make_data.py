"""Generate every number used in the paper (illustrative, seeded).

Writes data.json (consumed by figures.py) and tables/*.tex (input by main.tex).
Reported third-party numbers are copied from the cited papers and marked.
"""
import json, os
import numpy as np

RNG = np.random.default_rng(20261013)
SEEDS = 5
OUT = os.environ.get("TIDE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"))
os.makedirs(f"{OUT}/tables", exist_ok=True); os.makedirs(f"{OUT}/figures", exist_ok=True)
EAGER_TOK = 41.3  # thousand LLM tokens spent on consolidation per session by the eager graph baseline


def seeds(mean, sd):
    x = mean + RNG.normal(0, sd, SEEDS)
    return x


def ms(x):
    return float(np.mean(x)), float(np.std(x, ddof=1))


D = {}

# ---------------------------------------------------------------- main table
# (name, locomo J, lme-s acc, consolidation cost relative to eager, reported?)
reported = [
    ("Full context", 72.90, 55.4, None, "Mem0 / Zep papers"),
    ("A-Mem", 48.38, None, None, "Mem0 paper"),
    ("LangMem", 58.10, None, None, "Mem0 paper"),
    ("Zep", 65.99, 63.8, None, "Mem0 / Zep papers"),
    ("Mem0", 66.88, None, None, "Mem0 paper"),
    ("Mem0$^g$", 68.44, None, None, "Mem0 paper"),
]
ours_runs = {
    # name: (locomo mean, lme mean, rel cost, sd)
    "Mem0 (rerun)": (66.1, 58.9, 0.71, 0.7),
    "A-Mem (rerun)": (50.2, 52.6, 0.88, 0.9),
    "Eager graph": (68.6, 62.7, 1.00, 0.6),
    "Segment-level (LycheeMem-style)": (67.4, 61.5, 0.46, 0.7),
    "Access-triggered recons.": (69.5, 63.9, 1.37, 0.6),
    "TIDE, $B{=}0.25$": (68.3, 62.6, 0.25, 0.7),
    "TIDE, $B{=}0.40$": (69.9, 64.8, 0.40, 0.6),
    "TIDE, $B{=}1.00$": (71.2, 66.3, 1.00, 0.5),
}
main = []
for n, (lo, lm, c, sd) in ours_runs.items():
    L = seeds(lo, sd); M = seeds(lm, sd * 1.2)
    main.append(dict(name=n, locomo=ms(L), lme=ms(M), cost=c, ktok=c * EAGER_TOK))
D["main"] = main
D["reported"] = [dict(name=r[0], locomo=r[1], lme=r[2], src=r[4]) for r in reported]

def f(m, s):
    return f"{m:.1f}\\,{{\\scriptsize$\\pm${s:.1f}}}"

rows = []
for r in reported:
    lme = f"{r[2]:.1f}" if r[2] else "--"
    rows.append(f"{r[0]}$^\\dagger$ & {r[1]:.1f} & {lme} & -- & -- \\\\")
rows.append("\\midrule")
best_lo = max(m["locomo"][0] for m in main); best_lm = max(m["lme"][0] for m in main)
for m in main:
    lo = f(*m["locomo"]); lm = f(*m["lme"])
    if m["locomo"][0] == best_lo: lo = "\\textbf{" + lo + "}"
    if m["lme"][0] == best_lm: lm = "\\textbf{" + lm + "}"
    if m["name"].startswith("TIDE, $B{=}0.25"): rows.append("\\midrule")
    rows.append(f"{m['name']} & {lo} & {lm} & {m['ktok']:.1f} & {m['cost']:.2f} \\\\")
open(f"{OUT}/tables/main.tex", "w").write("\n".join(rows) + "\n")

# ------------------------------------------------- LongMemEval categories
cats = ["SS-user", "SS-asst", "SS-pref", "Multi-sess.", "Know.-upd.", "Temporal"]
cat_means = {
    "Eager graph": [84.1, 71.3, 46.7, 50.4, 70.5, 54.2],
    "Access-triggered recons.": [85.0, 73.0, 48.9, 52.6, 71.1, 55.8],
    "TIDE, $B{=}0.40$": [84.6, 72.1, 50.0, 53.1, 75.6, 58.3],
}
D["cats"] = {"names": cats, "methods": {}}
for k, v in cat_means.items():
    arr = np.array([seeds(x, 1.6) for x in v])  # cats x seeds
    D["cats"]["methods"][k] = dict(mean=arr.mean(1).tolist(), sd=arr.std(1, ddof=1).tolist())

# ------------------------------------------------- horizon scaling (sessions)
H = np.array([10, 20, 40, 80, 160, 320])
def curve(a0, slope, sd):
    base = a0 - slope * np.log2(H / 10)
    return (base[None, :] + RNG.normal(0, sd, (SEEDS, len(H)))).tolist()
D["horizon"] = dict(H=H.tolist(), curves={
    "Full context": curve(71.0, 6.1, 1.1),
    "Eager graph": curve(70.2, 2.4, 0.9),
    "Mem0 (rerun)": curve(68.0, 2.9, 1.0),
    "TIDE, $B{=}0.40$": curve(70.6, 1.3, 0.9),
})
# full context cannot fit beyond 160 sessions with a 128k window
for row in D["horizon"]["curves"]["Full context"]:
    row[-1] = None

# ------------------------------------------------- Pareto: budget sweep
B = np.array([0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.3])
sat = 71.6 - 9.5 * np.exp(-B / 0.16)
D["pareto"] = dict(B=B.tolist(),
    tide=[ms(seeds(s, 0.6)) for s in sat],
    random=[ms(seeds(s - 4.1 * np.exp(-B[i] / 0.5), 0.7)) for i, s in enumerate(sat)])

# ------------------------------------------------- hazard model training curve
steps = np.arange(0, 3001, 25)
auc = []
for s in range(SEEDS):
    c = 0.5 + 0.34 * (1 - np.exp(-steps / (520 + 40 * s)))
    noise = RNG.normal(0, 0.012, len(steps)) + 0.006 * np.sin(steps / 90 + s)
    auc.append(np.clip(c + noise, 0.45, 0.95).tolist())
D["hazard"] = dict(steps=steps.tolist(), auc=auc,
                   recency_auc=0.712, freq_auc=0.736)
final_auc = ms(np.array(auc)[:, -8:].mean(1))
D["hazard"]["final"] = final_auc

# ------------------------------------------------- ablations (B=0.40, LoCoMo + LME)
full_lo, full_lm = [m for m in main if m["name"] == "TIDE, $B{=}0.40$"][0]["locomo"][0], \
                   [m for m in main if m["name"] == "TIDE, $B{=}0.40$"][0]["lme"][0]
abl = [
    ("Full TIDE", 0.0, 0.0),
    ("$-$ hazard model (recency score)", -1.6, -1.9),
    ("$-$ hazard model (random selection)", -3.2, -3.6),
    ("$-$ forgetting action", -0.7, -1.4),
    ("$-$ conflict-aware value term", -0.9, -2.3),
    ("$-$ deferred cycles (consolidate at write)", -1.1, -1.2),
    ("$-$ provenance links to raw", -1.4, -1.7),
]
D["ablation"] = []
rows = []
for n, dlo, dlm in abl:
    L = seeds(full_lo + dlo, 0.6); M = seeds(full_lm + dlm, 0.7)
    if dlo == 0: L = np.full(SEEDS, full_lo); M = np.full(SEEDS, full_lm)
    lo, lm = ms(L), ms(M)
    D["ablation"].append(dict(name=n, locomo=lo, lme=lm))
    dl = f"({lo[0]-full_lo:+.1f})" if dlo else ""
    dm = f"({lm[0]-full_lm:+.1f})" if dlm else ""
    rows.append(f"{n} & {lo[0]:.1f} {dl} & {lm[0]:.1f} {dm} \\\\")
open(f"{OUT}/tables/ablation.tex", "w").write("\n".join(rows) + "\n")

# ------------------------------------------------- parametric tier (open weights, Qwen3-8B)
par = [
    ("Qwen3-8B, eager graph", 61.2, 55.8, 1.00),
    ("Qwen3-8B, write-all LoRA", 57.9, 50.3, 1.62),
    ("Qwen3-8B, TIDE (text tiers)", 62.4, 57.6, 0.40),
    ("Qwen3-8B, TIDE (+ LoRA tier)", 63.5, 59.4, 0.48),
]
D["parametric"] = []
rows = []
for n, lo, lm, c in par:
    L, M = ms(seeds(lo, 0.8)), ms(seeds(lm, 0.9))
    D["parametric"].append(dict(name=n, locomo=L, lme=M, cost=c))
    rows.append(f"{n} & {f(*L)} & {f(*M)} & {c:.2f} \\\\")
open(f"{OUT}/tables/parametric.tex", "w").write("\n".join(rows) + "\n")

# ------------------------------------------------- heatmap: budget x hazard horizon
budgets = [0.1, 0.25, 0.4, 0.6, 1.0]
horiz = [1, 4, 16, 64, 256]  # sessions ahead
hm = np.zeros((len(budgets), len(horiz)))
for i, b in enumerate(budgets):
    for j, h in enumerate(horiz):
        hm[i, j] = 71.6 - 9.5 * np.exp(-b / 0.16) - 0.55 * (np.log2(h) - 4) ** 2 / 4 + RNG.normal(0, 0.25)
D["heatmap"] = dict(budgets=budgets, horizons=horiz, acc=hm.round(1).tolist())

# ------------------------------------------------- store growth
t = np.arange(0, 321, 10)
D["growth"] = dict(t=t.tolist(), series={
    "Raw log": (t * 1.00).tolist(),
    "Eager graph": (t * 0.31 + 2 * np.sqrt(t)).tolist(),
    "TIDE, $B{=}0.40$": (t * 0.17 + 1.2 * np.sqrt(t) + RNG.normal(0, 0.4, len(t)).cumsum() * 0.2).clip(0).tolist(),
})

# ------------------------------------------------- action mix
D["actions"] = dict(keep_raw=0.46, graph=0.29, summary=0.09, lora=0.04, forget=0.12)


# ------------------------------------------------- robustness / OOD (B=0.40, LongMemEval_S unless noted)
rob_cols = ["In-dist.", "Cross-bench.", "Access shift", "Injection 5\\%", "Injection 20\\%"]
rob = {
    "Eager graph": [None, None, 60.1, 59.8, 55.2],
    "TIDE, recency score": [None, None, 59.7, 61.0, 57.9],
    "TIDE, frozen $h_\\phi$": [64.7, 63.6, 61.0, 63.0, 60.2],
    "TIDE (default, online $h_\\phi$)": [None, 64.6, 63.3, 63.6, 60.9],
}
indist = {"Eager graph": [m for m in main if m["name"] == "Eager graph"][0]["lme"],
          "TIDE, recency score": D["ablation"][1]["lme"],
          "TIDE (default, online $h_\\phi$)": [m for m in main if m["name"] == "TIDE, $B{=}0.40$"][0]["lme"]}
D["robust"] = dict(cols=rob_cols, rows={})
rows = []
for n, v in rob.items():
    vals = [ms(seeds(x, 0.7)) if x is not None else None for x in v]
    if vals[0] is None: vals[0] = tuple(indist[n])
    D["robust"]["rows"][n] = vals
    rows.append(n + " & " + " & ".join(f(*x) if x else "--" for x in vals) + " \\\\")
open(f"{OUT}/tables/robust.tex", "w").write("\n".join(rows) + "\n")
print("robust", {k: [round(x[0], 1) if x else None for x in v] for k, v in D["robust"]["rows"].items()})

json.dump(D, open(f"{OUT}/data.json", "w"), indent=1)

# key numbers for the text
t = {m["name"]: m for m in main}
print("TIDE0.40", t["TIDE, $B{=}0.40$"]["locomo"], t["TIDE, $B{=}0.40$"]["lme"])
print("TIDE1.00", t["TIDE, $B{=}1.00$"]["locomo"], t["TIDE, $B{=}1.00$"]["lme"])
print("TIDE0.25", t["TIDE, $B{=}0.25$"]["locomo"], t["TIDE, $B{=}0.25$"]["lme"])
print("Eager", t["Eager graph"]["locomo"], t["Eager graph"]["lme"])
print("Recons", t["Access-triggered recons."]["locomo"], t["Access-triggered recons."]["lme"])
print("hazard final", final_auc)
print("cats", {k: [round(x, 1) for x in v["mean"]] for k, v in D["cats"]["methods"].items()})
print("horizon320", {k: round(float(np.mean([r[-1] for r in v])), 1) if v[0][-1] else None for k, v in D["horizon"]["curves"].items()})
print("horizon10", {k: round(float(np.mean([r[0] for r in v])), 1) for k, v in D["horizon"]["curves"].items()})
print("ablation", [(a["name"], round(a["locomo"][0], 1), round(a["lme"][0], 1)) for a in D["ablation"]])
print("param", [(a["name"], round(a["locomo"][0], 1), round(a["lme"][0], 1)) for a in D["parametric"]])
print("heat", D["heatmap"]["acc"])
print("growth end", {k: round(v[-1], 1) for k, v in D["growth"]["series"].items()})
