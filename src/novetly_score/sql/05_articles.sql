-- articles: one row per article. Grain is the article, NOT the (article, company) mention.
--
-- Built by engine/derive.py, which collapses the feed's per-company fan-out (1.579 rows per
-- article) and left-joins the headline. That is not a dedup of stories and drops no coverage.
--
-- title is NULL for most articles: the feed holds 133.7M and the headline dumps cover 10.86M
-- (8.1%). NULL states the truth -- the article exists, we do not have its headline -- and
-- `WHERE title IS NULL` is the exact coverage gap.
CREATE OR REPLACE VIEW articles AS
SELECT
    article_id,
    publish_time,
    source_id,
    lang,
    news_type,
    title,
    title_source
FROM read_parquet('$derived/articles/*.parquet');
