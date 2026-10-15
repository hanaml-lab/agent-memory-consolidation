# TIDE: hazard-scheduled, budget-constrained memory consolidation

[![ci](https://github.com/hanaml-lab/agent-memory-consolidation/actions/workflows/ci.yml/badge.svg)](https://github.com/hanaml-lab/agent-memory-consolidation/actions/workflows/ci.yml)

Reference implementation of *TIDE: Hazard-Scheduled, Budget-Constrained Memory Consolidation for Long-Horizon LLM Agents* (Agarwal, hanaML technical report, 2026).

Long-term memory systems for LLM agents (Mem0, A-Mem, Zep/Graphiti, Letta) consolidate on a fixed trigger: every turn, every session, or every retrieval. TIDE treats consolidation as a budget-constrained allocation problem. It has three parts:

- **Hazard model** `h_phi(k | x_u)`: a discrete-time survival model of when a memory unit will next be accessed. It is trained on the censored log-likelihood using self-supervised probe accesses (`tide.hazard`).
- **Value of consolidation**: access utility + relational utility + a conflict term for superseded beliefs - storage cost (`tide.voc`).
- **Deferred scheduler**: a greedy multiple-choice knapsack over `{keep, summary, graph, lora, forget}` that runs every `w` sessions under a budget `B` relative to eager consolidation (`tide.scheduler`).

The release contains the scheduler, the hazard model, a minimal tiered store with a temporal slot graph, and a toy stream that runs end to end on CPU. It does not include the harness that calls hosted LLMs for LoCoMo and LongMemEval; `configs/locomo_gpt4omini.yaml` records the settings used there.

## Install

```bash
pip install -e ".[dev]"          # numpy, pyyaml; pytest and ruff
pip install -e ".[paper]"        # + matplotlib, to regenerate the report's figures
```

## Quickstart (CPU, under a minute)

```bash
python -m tide.demo              # or tide-demo; settings in configs/toy.yaml
pytest -q
```

## Toy run (real output)

The toy stream (`tide.sim`) writes 60 sessions of memory units; some state facts about slots whose values change, and queries ask for the current value of popular slots. Every 2 sessions the pending units are scheduled under a budget, a share of the tokens eager consolidation would spend. TIDE ranks units by the hazard model's probability of access; the baselines rank them by recency or at random. Output of `python -m tide.demo` on a CPU (Python 3.12):

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

TIDE leads at middle budgets (80.9% of queries at 40% of eager's tokens, against 78.7% for random and 73.3% for recency), but at the tightest budgets it does not beat random scheduling. The toy does not test the report's headline against eager consolidation: here eager is perfect by construction. `paper/toy_run.md` records the environment and what each line shows.

## Scheduling your own memory

`examples/schedule_memory.py` reads memory units (session, text, entities, tokens, the slot each asserts, and its probability of access from any model, such as `tide.HazardModel`), schedules a consolidation cycle every few sessions under a budget, and prints each unit's action.

```bash
python examples/schedule_memory.py examples/data/memory_units.jsonl --budget 0.4
```

```
cycle ending at session 3: spent 0.290 of eager 0.770 (k tokens)
    0 graph    p=0.62 links=0  I live in Berlin.
    1 graph    p=0.71 links=1  I work at Quill as an analyst.
    2 keep     p=0.03 links=0  Lovely weather today, we walked by the river.
    3 summary  p=0.44 links=1  My manager is Ana.
    4 summary  p=0.38 links=0  I train at the Atlas gym on Mondays.
    5 keep     p=0.27 links=2  Ana asked me to review the Quill budget.
    6 keep     p=0.02 links=0  Thinking about pizza for dinner.
cycle ending at session 7: spent 0.204 of eager 0.570 (k tokens)
    7 graph    p=0.66 links=1  I moved to Lisbon last week.  [conflicts with 'Berlin']
    8 summary  p=0.52 links=0  My new manager is Ben.
    9 keep     p=0.21 links=4  The Lisbon office is near the station.
   10 keep     p=0.04 links=0  Watched a film about sailing.
   11 summary  p=0.58 links=3  Still at Quill, now leading the analytics team.

spent 37% of eager consolidation; 3 graph, 5 keep, 4 summary
consolidated beliefs: user.lives_in = Lisbon, user.works_at = Quill
```

From Python, one cycle is:

```python
from tide import TieredStore, schedule_cycle
store = TieredStore()
for u in pending:
    store.add_raw(u)
actions, spent, eager = schedule_cycle(pending, p_access, {u.uid: store.kappa(u) for u in pending},
                                       {u.uid: store.chi(u) for u in pending}, budget=0.4, now=session)
```

## Results (from the report)

These numbers come from the report's seeded simulation and are illustrative. GPT-4o-mini reader, mean over 5 seeds:

| Method | LoCoMo J | LongMemEval_S | Consolidation tokens / session | Relative cost |
|---|---|---|---|---|
| Mem0 (rerun) | 66.3 | 59.1 | 29.3k | 0.71 |
| Eager graph | 68.6 | 62.4 | 41.3k | 1.00 |
| Access-triggered reconsolidation | 69.7 | 64.5 | 56.6k | 1.37 |
| **TIDE, B = 0.40** | **69.7** | **65.1** | **16.5k** | **0.40** |
| **TIDE, B = 1.00** | **71.4** | **66.2** | 41.3k | 1.00 |

`bash scripts/reproduce_paper.sh` regenerates the tables and figures that the report's seeded simulation produces. On every push, CI reruns `paper/make_data.py` and checks the tables and results it writes against `paper/results.sha256` (`python paper/check_results.py`).

## Layout

```
src/tide/         hazard.py  voc.py  scheduler.py  store.py  sim.py  types.py  demo.py
configs/          toy.yaml  locomo_gpt4omini.yaml
examples/         schedule_memory.py, data/memory_units.jsonl
scripts/          run_toy.py  reproduce_paper.sh
paper/            make_data.py  figures.py  check_results.py  results.sha256  toy_run.md
tests/            test_hazard.py  test_scheduler.py  test_end_to_end.py  test_demo.py  test_examples.py
```

## Citation

```bibtex
@techreport{agarwal2026tide,
  title       = {{TIDE}: Hazard-Scheduled, Budget-Constrained Memory Consolidation for Long-Horizon {LLM} Agents},
  author      = {Agarwal, Manav},
  institution = {hanaML},
  year        = {2026}
}
```

## License

Apache-2.0. See `LICENSE`.
