# TIDE: Hazard-Scheduled, Budget-Constrained Memory Consolidation

Reference implementation for the technical report *TIDE: Hazard-Scheduled, Budget-Constrained Memory Consolidation for Long-Horizon LLM Agents* (Manav Agarwal, hanaML, 2026).

Long-term memory systems for LLM agents (Mem0, A-Mem, Zep/Graphiti, Letta) consolidate on a fixed trigger: every turn, every session, or every retrieval. TIDE treats consolidation as a budget-constrained allocation problem. It has three parts:

- **Hazard model** `h_phi(k | x_u)`: a discrete-time survival model of when a memory unit will next be accessed. It is trained on the censored log-likelihood using self-supervised probe accesses (`tide.hazard`).
- **Value of consolidation**: access utility + relational utility + a conflict term for superseded beliefs - storage cost (`tide.voc`).
- **Deferred scheduler**: a greedy multiple-choice knapsack over `{keep, summary, graph, lora, forget}` that runs every `w` sessions under a budget `B` relative to eager consolidation (`tide.scheduler`).

The release contains the scheduler, the hazard model, a minimal tiered store with a temporal slot graph, and a toy stream that runs end to end on CPU in a few seconds. It does not include the harness that calls hosted LLMs for LoCoMo and LongMemEval. `configs/locomo_gpt4omini.yaml` records the settings used there.

## Install

```bash
pip install -e ".[dev,paper]"
```

## Quickstart

```python
from tide import run_toy
for policy in ["eager", "tide", "recency", "random"]:
    print(run_toy(budget=0.4, policy=policy, seed=0))
```

```bash
python scripts/run_toy.py --config configs/toy.yaml   # sweep budgets x policies x seeds
pytest -q                                              # unit + end-to-end tests
bash scripts/reproduce_paper.sh                        # regenerate paper tables and figures
```

## Results (from the paper; GPT-4o-mini reader, mean over 5 seeds)

| Method | LoCoMo J | LongMemEval_S | Consolidation tokens / session | Relative cost |
|---|---|---|---|---|
| Mem0 (rerun) | 66.3 | 59.1 | 29.3k | 0.71 |
| Eager graph | 68.6 | 62.4 | 41.3k | 1.00 |
| Access-triggered reconsolidation | 69.7 | 64.5 | 56.6k | 1.37 |
| **TIDE, B = 0.40** | **69.7** | **65.1** | **16.5k** | **0.40** |
| **TIDE, B = 1.00** | **71.4** | **66.2** | 41.3k | 1.00 |

## Layout

```
src/tide/         hazard.py  voc.py  scheduler.py  store.py  sim.py  types.py
configs/          toy.yaml  locomo_gpt4omini.yaml
scripts/          run_toy.py  reproduce_paper.sh  paper/{make_data.py, figures.py}
tests/            test_hazard.py  test_scheduler.py  test_end_to_end.py
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

Apache-2.0.
