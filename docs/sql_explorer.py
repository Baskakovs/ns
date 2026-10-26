# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.19.6",
#     "polars",
#     "duckdb",
#     "pyarrow",
#     "honey",
# ]
# [tool.uv.sources]
# honey = { path = "../../../.." , editable = true }
# ///
"""A tour of the novelty query layer: what each view holds, and how they join.

Run with:

    uvx marimo edit --sandbox book/marimo/notebooks/novelty/sql_explorer.py

`uvx` rather than `uv run`, because marimo is not a project dependency — these notebooks carry
their own, in the PEP 723 header above, and `--sandbox` is what makes marimo build the
environment from it.

Needs the parquets under `data/`, the derived tables (`python -m
honey.novelty_score_exp.engine.derive`) and a scores file per window
(`python -m honey.novelty_score_exp.engine.run --window-hours 6`) — see
`src/honey/novelty_score_exp/CLAUDE.md`. A view whose parquet is absent is skipped; one whose
parquet was written by other code raises.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from honey.novelty_score_exp.core.functions import DEFAULT_F, FAMILY
    from honey.novelty_score_exp.engine import aggregation, checks, db, io

    return DEFAULT_F, FAMILY, aggregation, checks, db, io, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # The novelty query layer

    Storage is **parquet**; this is a DuckDB *view layer* over it. Nothing is copied into a
    `.duckdb` file, and **the views hold no rules**: each one projects columns of a parquet the
    engine wrote, and `mentions_scored` joins two of them on their key. The tone, the dedups,
    the clip, the calibration and the panels are computed in `engine/`, each pinned to `core/`.

    Every file a view reads is stamped with the code that wrote it. If `core/` or the engine has
    changed since, connecting raises rather than showing numbers the current definition would
    not produce.
    """)
    return


@app.cell
def _(DEFAULT_F, FAMILY, mo):
    f = mo.ui.dropdown(options=sorted(FAMILY), value=DEFAULT_F, label="candidate f")
    window = mo.ui.dropdown(options=["6", "24", "72", "168", "336"], value="6", label="lookback window (hours)")
    k = mo.ui.dropdown(options=["1", "3", "5", "7", "10", "20", "50", "70", "100"], value="5", label="k")
    theta = mo.ui.slider(0.0, 1.0, 0.05, value=0.30, label="theta", show_value=True)
    mo.md(f"""
    ## Pick a grid cell

    The scores view is one `(f, window, k)` cell at a time; `theta` parameterises the filter
    route. A window with no scores file yet leaves `scores` and `mentions_scored` out, and the
    cells that need them skip themselves.

    {f} &nbsp; {window} &nbsp; {k} &nbsp; {theta}
    """)
    return f, k, theta, window


@app.cell
def _(db, f, k, window):
    con, applied = db.connect(f=f.value, window=window.value, k=k.value)
    scored = "mentions_scored" in applied.views
    applied
    return con, scored


@app.cell
def _(io, mo):
    built = io.manifest()
    mo.md(
        "### Scores files on disk\n\n"
        + (
            "\n".join(f"- `{b.f}` @ `{b.label}` — {'current' if b.current else '**stale: re-score**'}" for b in built)
            or "*None yet.*"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What is in each view

    `mentions` and `articles` are built tables now, so counting them reads finished rows rather
    than re-aggregating the feed. The numbers below are from the author's store.

    | view | grain | note |
    |---|---|---|
    | `titles` | one row per distinct headline | 38.5M vectors, 256 dims |
    | `article_titles` | `article_id` -> headline | 54.2M, the two dumps unioned and deduped |
    | `sources` | one row per outlet | 359,755 |
    | `constituents` | one row per (company, membership spell) | 962 — use point-in-time |
    | `articles` | one row per article | 133.7M, built by `engine.derive` |
    | `mentions` | one row per (article, company) | **199.7M** — the grain the signal is built on |
    | `scores` | one row per (article, company) | one `(f, window, k)` cell, both scales |
    | `mentions_scored` | `mentions` + both scales | every mention survives; NULL where unscored |

    There is **no column called `novelty`**: `raw_novelty` is the score, `calibrated_novelty` its
    rank within its own reference set — what §3.1 and §3.2 threshold and weight.
    """)
    return


@app.cell
def _(con, mo):
    counted = ["titles", "article_titles", "sources", "constituents", "articles", "mentions", "scores"]
    have = {r[0] for r in con.sql("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()}
    counts = {v: con.table(v).aggregate("count(*)").fetchone()[0] for v in counted if v in have}
    mo.md("### Row counts\n\n" + "\n".join(f"- `{v}` — {n:,}" for v, n in counts.items()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Schemas

    `DESCRIBE` is free — it reads the parquet footer, not the data. The fastest way to see what
    you can join on.
    """)
    return


@app.cell
def _(mo):
    view = mo.ui.dropdown(
        options=[
            "titles",
            "article_titles",
            "sources",
            "constituents",
            "articles",
            "mentions",
            "scores",
            "mentions_scored",
        ],
        value="mentions",
        label="describe",
    )
    view
    return (view,)


@app.cell
def _(con, view):
    con.sql(f"DESCRIBE {view.value}").pl()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The joins that matter

    ### `mentions` -> `scores`, on `(article_id, company)`

    This is the one join the signal needs, and the one that is easy to get catastrophically
    wrong. Novelty is **(article, company)-scoped** — the candidate pool is that company's own
    earlier headlines, so one article scores differently for each company it mentions.

    **Never join on `(company, publish_date)`.** `scores` holds ~257 rows per company-day, so
    that predicate is a cartesian product per company-day rather than a lookup.

    `mentions_scored` already does this join. It is a LEFT JOIN, so every mention survives and
    both scales are NULL wherever no score exists — before 2024-07 that is all of them, since
    novelty reaches only as far as `es_metadata`.
    """)
    return


@app.cell
def _(con, mo, scored, window):
    mo.stop(not scored, mo.md(f"*Skipped: no scores file for window `{window.value}` yet.*"))
    con.sql("""
        SELECT article_id, company, publish_date, sentiment, raw_novelty, calibrated_novelty
        FROM mentions_scored
        WHERE calibrated_novelty IS NOT NULL
        LIMIT 10
    """).pl()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### `mentions` -> `articles`, to read the headline

    `articles` carries the headline already, so a NULL title means "article known, headline not"
    rather than a broken join. Coverage is **8.1%** over the whole feed — the headline dumps
    reach only part of it.
    """)
    return


@app.cell
def _(con):
    con.sql("""
        SELECT m.company, a.publish_time, a.lang, a.title, m.sentiment
        FROM mentions m
        JOIN articles a USING (article_id)
        WHERE a.title IS NOT NULL
        LIMIT 10
    """).pl()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### `constituents`, point in time

    The index membership bridge: `company` (the Yukka id, joining to `mentions.company`) <->
    `ric` / `isin`, with membership spells. Joining without the
    date predicate silently reintroduces survivorship bias — a name that left the index in 2018
    should not be in the 2024 universe.
    """)
    return


@app.cell
def _(con):
    con.sql("""
        SELECT c.company, c.ric, c.isin, c.name, c.first_in, c.last_in, c.current_member
        FROM constituents c
        WHERE c.current_member
        LIMIT 10
    """).pl()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## A worked question

    *Which outlets carry the most novel coverage of a company?* Joins `mentions_scored` to
    `articles` for the outlet, and averages calibrated novelty per source.

    Note the `HAVING` — a source with three articles will top any ranking by mean, which is the
    usual way this kind of query misleads.
    """)
    return


@app.cell
def _(mo):
    company = mo.ui.text(value="company:allianz", label="company")
    min_articles = mo.ui.number(1, 10000, value=50, label="min articles")
    mo.hstack([company, min_articles])
    return company, min_articles


@app.cell
def _(company, con, min_articles, mo, scored, window):
    mo.stop(not scored, mo.md(f"*Skipped: no scores file for window `{window.value}` yet.*"))
    # bound parameters rather than interpolation: the two values come from the widgets above
    con.execute(
        """
        SELECT
            a.source_id,
            count(*)                       AS articles,
            avg(s.calibrated_novelty)      AS mean_calibrated,
            median(s.calibrated_novelty)   AS median_calibrated,
            avg(s.raw_novelty)             AS mean_raw
        FROM mentions_scored s
        JOIN articles a USING (article_id)
        WHERE s.company = ?
          AND s.calibrated_novelty IS NOT NULL
        GROUP BY 1
        HAVING count(*) >= ?
        ORDER BY mean_calibrated DESC
        LIMIT 20
        """,
        [company.value, min_articles.value],
    ).pl()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The panels

    §3's routes, computed by `engine.aggregation.panel` — the implementation pinned to
    `core.aggregation` — and emitting production's `(date, entity, positive, neutral, negative)`
    shape so `build_portfolio` consumes any of them unchanged. Not SQL: a panel is a rule, and
    rules live in the engine.

    **Every route counts the same rows**: the mentions with a calibrated novelty. Production
    included — otherwise the baseline would trade a corpus the novelty arms cannot see. The
    `WHERE` below only prunes what is pulled into memory; `aggregation.scoreable` is the rule,
    and it applies it again.
    """)
    return


@app.cell
def _(con, mo, scored, window):
    mo.stop(not scored, mo.md(f"*Skipped: no scores file for window `{window.value}` yet.*"))
    scored_mentions = con.sql("""
        SELECT publish_date, company, sentiment, calibrated_novelty
        FROM mentions_scored
        WHERE calibrated_novelty IS NOT NULL
    """).pl()
    scored_mentions.height
    return (scored_mentions,)


@app.cell
def _(aggregation, mo, pl, scored_mentions, theta):
    panels = {
        "production": aggregation.panel(scored_mentions, "production"),
        "filtered": aggregation.panel(scored_mentions, "filtered", theta=theta.value),
        "weighted": aggregation.panel(scored_mentions, "weighted"),
    }
    summary = pl.DataFrame(
        {
            "route": list(panels),
            "rows": [p.height for p in panels.values()],
            "positive": [p["positive"].sum() for p in panels.values()],
            "negative": [p["negative"].sum() for p in panels.values()],
        }
    )
    mo.vstack([summary, panels["filtered"].sort("date", descending=True).head(10)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Checks

    Views carry no constraints, so the invariants live in `engine.checks` as functions returning
    the number of violating rows. `feed_repeats` and `company_coverage` are informational; the
    rest should be zero. Several scan the 217M-row feed, so they take a few minutes.
    """)
    return


@app.cell
def _(mo):
    run_checks = mo.ui.run_button(label="run checks (slow — scans the feed)")
    run_checks
    return (run_checks,)


@app.cell
def _(checks, f, io, mo, run_checks, window):
    mo.stop(not run_checks.value, mo.md("*Not run.*"))
    match = next((b for b in io.manifest() if b.f == f.value and b.hours == float(window.value)), None)
    results = checks.run(match.path if match else None)
    mo.md("\n".join(f"- `{name}` — {n:,}" for name, n in results.items()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Your own query

    Everything above is `con.sql(...).pl()` — a polars DataFrame out the other side.
    """)
    return


@app.cell
def _(mo):
    query = mo.ui.text_area(
        value="SELECT company, count(*) AS mentions\nFROM mentions\nGROUP BY 1\nORDER BY 2 DESC\nLIMIT 10",
        label="SQL",
        rows=8,
        full_width=True,
    )
    run = mo.ui.run_button(label="run")
    mo.vstack([query, run])
    return query, run


@app.cell
def _(con, mo, query, run):
    mo.stop(not run.value, mo.md("*Press run.*"))
    try:
        out = con.sql(query.value).pl()
    except Exception as e:  # noqa: BLE001
        out = mo.md(f"```\n{e}\n```")
    out
    return


if __name__ == "__main__":
    app.run()
