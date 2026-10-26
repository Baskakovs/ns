-- titles: one row per distinct headline, with its embedding.
--
-- The embedding lives here rather than on the article, so a reprinted headline is embedded
-- once (2.1x repeat rate over the dump). That is a cost saving with no effect on any score:
-- identical strings have identical embeddings, and every *article* is still scored separately
-- against the full reference set.
--
-- Keyed on the title string, not a hash. titles.title_hash in the previous schema was polars'
-- hash(seed=0), which DuckDB cannot reproduce -- hash() returns a different value for the same
-- string. The hash remains internal to engine/store.py, where the numpy memmap needs a uint64
-- key. Joining on the string is also what makes this layer readable, and parquet
-- dictionary-encodes the repeats.
--
-- This view is where old and new embeddings combine: every run's shards land in the same
-- directory, so the glob unions them with no migration step.
CREATE OR REPLACE VIEW titles AS
SELECT
    title,
    embedding AS embedding_vector   -- FLOAT[256]; see params.DIMS
FROM read_parquet('$data/es_title_embeddings/titles_emb_*.parquet');
