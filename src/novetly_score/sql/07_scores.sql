-- scores: per-(article, company) novelty for one (f, window, k) cell of the grid, on both scales.
--
-- There is no column called `novelty`. raw_novelty is the score, nu; calibrated_novelty is its
-- rank within its own reference set, nu_tilde -- what §3.1 and §3.2 threshold and weight. They
-- share a dtype and a range, so the name is the only thing that says which one a number is.
--
-- Both are stored exactly as core/ defines them (clipped into [0, 1], NULL where the title was
-- never embedded) by engine/scorer.py, and engine/db.py refuses to expose a file that scorer as
-- it stands now did not write. Nothing is computed here.
CREATE OR REPLACE VIEW scores AS
SELECT
    article_id,
    entity             AS company,
    date               AS publish_date,
    sentiment,
    $raw_col           AS raw_novelty,
    $calibrated_col    AS calibrated_novelty
FROM read_parquet('$scores');

-- mentions_scored: every mention, with its novelty where one exists.
--
-- Joined on (article_id, company), the key both sides are built at. It must NOT be joined on
-- (company, publish_date): scores holds ~257 rows per company-day, so that predicate is a
-- cartesian product per company-day rather than a lookup.
--
-- A LEFT JOIN, so every mention survives and both novelties are NULL where no score exists --
-- before 2024-07 that is all of them. Which rows a panel counts is engine/aggregation.py's
-- decision, not this view's.
CREATE OR REPLACE VIEW mentions_scored AS
SELECT
    m.*,
    s.raw_novelty,
    s.calibrated_novelty
FROM mentions m
LEFT JOIN scores s
       ON s.article_id = m.article_id
      AND s.company = m.company;
