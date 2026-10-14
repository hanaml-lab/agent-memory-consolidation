# Toy run (real output)

A real run of the reference implementation on its CPU toy setting. Unlike the report's tables, which come from a
seeded simulation and are illustrative, every number below was printed by the code. The toy is small by design: it
checks that the method works end to end on a problem with a known answer; it does not reproduce the report's
experiments.

- Command: `python -m tide.demo` (settings in `configs/toy.yaml`)
- Code: commit `83c5fe2`
- Environment: Python 3.12.13, numpy 2.5.4, PyYAML 6.0.3; Linux x86_64, Intel(R) Core(TM) i9-14900KF
- Wall time: 70 s

## The toy setting

`tide.sim.ToyStream` writes 60 sessions of 12 memory units each. About 45% of units state a fact about one of 40
slots, chosen with skewed popularity, and later facts can change a slot's value; the rest are other turns. Queries
ask for the current value of popular slots. The reader answers from the consolidated graph when the slot is there,
and otherwise from the earliest raw mention, so unreconciled evidence misleads it. The hazard model is trained on a
separate replay stream (simulated probe accesses), never on the evaluation queries. Every 2 sessions, the pending
units are scheduled under a budget expressed as a share of the tokens eager consolidation would spend. TIDE ranks
units by the hazard model's probability of access within 16 sessions; the baselines rank them by recency or at
random. Eager consolidation sends every unit to the graph. Each cell is the mean over seeds 0-2.

## Output

```
current-value accuracy, mean over seeds 0-2; budget = share of eager consolidation tokens
budget      tide   recency    random  max spent
  0.10     0.336     0.299     0.368       0.10
  0.25     0.642     0.491     0.654       0.25
  0.40     0.809     0.733     0.787       0.40
  0.60     0.943     0.913     0.899       0.59
  1.00     1.000     1.000     1.000       1.00
eager consolidation (everything to the graph): accuracy 1.000
hazard model AUROC for access within 16 sessions: 0.918
done in 66.5s
```

## What it shows

- Eager consolidation answers every query (1.000), at full cost. Every policy gains accuracy with budget, and none
  spends more than its budget (the last column is the most any policy spent).
- TIDE leads at middle budgets: at 40% of eager's tokens it answers 80.9% of queries (random 78.7%, recency 73.3%),
  and at 60% it answers 94.3% (91.3% and 89.9%).
- At the tightest budgets it does not beat random scheduling (0.336 against 0.368 at 10%, 0.642 against 0.654 at
  25%), although the hazard model ranks future accesses well (AUROC 0.918).
- The toy does not test the report's headline, that TIDE at 40% of eager's compute matches or beats eager
  consolidation on LoCoMo and LongMemEval: here eager is perfect by construction, so only the budgeted comparison
  between policies is informative.
