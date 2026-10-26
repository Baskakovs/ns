-- article_titles: article_id -> headline. The link between the sentiment feed, which knows
-- which company an article is about, and the headline dumps, which know what it said.
--
-- Built by a Python pass over the two NDJSON dumps (they are 55 GB of JSON, so they are not
-- read through SQL). Deduplicated on article_id: the titles dump holds 48.5M rows for 45.1M
-- distinct ids, and leaving the 7.0% repeats in would make an article its own nearest
-- neighbour, so a fresh story would read as a reprint.
--
-- `dump` records which export a headline came from. The two are near-disjoint by article:
-- only 711 shared ids out of 54.2M.
CREATE OR REPLACE VIEW article_titles AS
SELECT
    article_id,
    title,
    CAST(published AS TIMESTAMP) AS publish_time,
    dump
FROM read_parquet('$data/article_titles.parquet');
