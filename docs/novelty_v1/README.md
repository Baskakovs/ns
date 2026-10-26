# novelty_v1 — retired

The first implementation of the novelty experiment, retired when
`src/honey/novelty_score_exp/` was rebuilt around the decomposition in `Novelty Paper.md`.

**Do not read this to decide how the new code should work.** It is kept so results can be
traced back to the code that produced them, and for no other purpose. The paper is the
specification; this tree predates it and disagrees with it in several places.

It sits outside `src/` deliberately. There is no `__init__.py` and it is on no import
path, so nothing in the live package can depend on it by accident. `archive/ruff.toml`
exempts it from linting, `pytest.ini`'s `testpaths = tests` keeps its tests uncollected,
and `pyproject.toml` omits it from coverage.

## What is here

| path | what it was |
|---|---|
| `novelty_knn_model.py` | the 651-line monolith: store, scorer, dump parsing, both aggregation routes, portfolio, eval helpers |
| `dispersion_scan.py` | the N4 dispersion/centroid experiment |
| `invariance.py` | measured how far article exclusion moves the production signal |
| `sweep_status.py`, `status_dashboard.py` | progress viewers |
| `test.py` | a scratch marimo notebook, not a test |
| `sql/` | DuckDB DDL that nothing ever loaded |
| `notebooks/` | the five novelty notebooks, the event-exclusion notebook, and the superseded `NOVELTY.md` and `PIPELINE.md` |
| `event_exclusion_exp/` | a separate experiment: ranked 149 event types by whether excluding their articles improved IC. Null result |
| `tests/` | the tests for the above |
| `old-implementation-notes.md` | the former subpackage `CLAUDE.md`, renamed so it is not auto-loaded as project instructions |

## Findings worth carrying forward

Two results here are cheap to lose and expensive to rediscover.

`invariance.py` measured that filtering at `novelty > 0.30` discards 75% of articles yet
leaves the production signal correlated **0.876** with baseline — against **0.939** for a
*random* drop of the same size. Most of what novelty filtering does to that book is inject
sampling noise, not information. Both figures are on the **raw** novelty scale and are
superseded by the calibrated score.

The event-exclusion experiment returned a null result across all 149 event types, for the
same structural reason: the signal is a ratio, and the per-name median/MAD step and the
cross-sectional z-score each independently absorb what exclusion changes.
