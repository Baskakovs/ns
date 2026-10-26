-- sources: one row per outlet, the distinct source_uri of the sentiment feed.
--
-- Built by engine/derive.py; this view only exposes it. `rank` in the previous schema held
-- Yukka's per-source t_stat and was always NULL: it is a server-side filter, not a column the
-- API returns, so it is not modelled. An empirically derived tier belongs in its own table,
-- derived from mention volume, so it is never mistaken for the vendor's figure.
CREATE OR REPLACE VIEW sources AS
SELECT source_id                   -- Yukka slug, e.g. 'source:die-presse', not a URL
FROM read_parquet('$derived/sources.parquet');
