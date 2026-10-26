-- mentions: one row per (article, company). The grain the signal is built on, and the grain a
-- novelty score attaches to -- novelty is (article, company)-scoped, because the candidate pool
-- is that company's own earlier headlines.
--
-- Built by engine/derive.py, which aggregates the feed's repeated rows for a pair (1.10 per
-- pair) into one, with the pair's tone computed as core.counts.tone defines it and pinned to it
-- by tests/test_novelty_derive.py. This view only exposes the result.
--
-- sentiment is -1 / 0 / +1, or NULL where no polarity was found anywhere for the pair. **NULL is
-- not neutral** -- 8.6% of the feed overall and 54% for the busiest names.
--
-- No membership filter: restricting the universe is the caller's decision -- join constituents.
CREATE OR REPLACE VIEW mentions AS
SELECT
    article_id,
    company,
    publish_date,
    sentiment,
    implied,
    feed_rows                   -- >1 means the feed repeated this pair
FROM read_parquet('$derived/mentions/*.parquet');
