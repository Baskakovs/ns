# Working in this repository

## CI is the gate, and it is enforced

`main` is protected: a merge needs **`ci / CI gate`** and **`book / build`** green, and the branch
must be up to date with `main` first. Force-pushes and branch deletion are blocked.

Run the gates before pushing, not after:

```bash
make fmt test typecheck deptry docs-coverage     # the five that gate a PR
make all                                         # the above plus security, license, rhiza-test
```

Each one maps to a CI job, so a local failure is a CI failure you have not waited for yet.

### One gate disagrees with CI, on purpose

`make rhiza-test` runs the **vendored** `.rhiza/tests`, which assert that `classifiers` must
include a `License ::` entry. The **`pytest-rhiza`** that the `Rhiza repository checks` job
installs asserts the opposite — that a deprecated `License ::` classifier must be absent, since
`license = "MIT"` and `license-files` already carry it. They contradict, a template sync changes
nothing, and CI is what gates a merge, so the classifier stays out and
`test_license_classifier_present` stays red locally. Do not "fix" it by adding the classifier
back: that turns a red local assertion into a red merge gate.

## The notebooks cannot run in CI, and must not fail there

`docs/notebooks/*.py` read a private MinIO store and the STOXX 600 universe, which need
`S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET` and `YUKKA_TOKEN`. **This repository defines no
secrets**, so the `MARIMO` and `BOOK` jobs have none of them. Each notebook therefore calls
`mo.stop` in its first data cell, naming what is unset, and exits 0 rather than failing.

Two consequences when editing a notebook:

- **Keep the guard in the cell everything else depends on.** `mo.stop` only skips the cells
  downstream of the one that calls it, so a guard in a cell nobody reads from protects nothing.
- **Keep the PEP 723 header honest.** `BOOK` exports with `marimo export html --sandbox`, which
  builds an isolated environment from that header alone — a dependency that is only in the
  `notebook` group is invisible to it. A new import means a new line in the header.

Verify both paths before pushing a notebook change:

```bash
cd docs/notebooks
uv run --with marimo marimo export html --sandbox ic_baseline.py -o /tmp/nb.html     # with .env
env S3_ENDPOINT= S3_ACCESS_KEY= S3_SECRET= YUKKA_TOKEN= uv run ic_baseline.py        # as CI
```

## Things that are measured, not assumed

- **The counts panel is stored, not derived per caller.** `data(universe).counts(...)` reads
  `s3://quant-experiments/panels/counts.parquet` in ~2s. Deriving it from the 181M snippet rows
  takes 4–28 minutes depending on how busy the store is, and that is what `write_counts()` is
  for — run it when the store gains a month, and nothing else.
- **A stored panel can go stale, which is a correctness problem.** `counts()` compares the
  panel's stamp against the live manifest and raises `StalePanelError` rather than answering with
  a panel that stops short. The fix is always `write_counts()`.
- **`moto`'s test server reports binding `0.0.0.0`.** Windows refuses to *connect* to that
  (WinError 10049), so every fixture builds its endpoint from `127.0.0.1` and ignores the
  reported host. Three fixtures do this; keep it that way.
- **DuckDB's `httpfs` autoload is a flake source.** It is fetched from DuckDB's CDN the first
  time a connection sets `s3_endpoint`, and that download fails occasionally on CI runners. Do
  not "fix" it by loading the extension and applying the S3 settings via `SET`: credentials
  passed through SQL land in DuckDB's query log, which is why they go in the connection config.
