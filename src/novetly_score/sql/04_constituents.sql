-- constituents: STOXX 600 membership, one row per company.
--
-- `spells` counts how many separate membership periods a name has had; first_in/last_in bound
-- the outer envelope rather than describing each spell, so a name that left and re-entered
-- cannot be masked exactly from this source alone. Point-in-time masking for a backtest needs
-- the per-spell data, not this view.
CREATE OR REPLACE VIEW constituents AS
SELECT
    yukka_id AS company,
    ric,
    isin,
    name,
    country,
    currency,
    first_in,
    last_in,
    spells,
    current_member
FROM read_parquet('$data/stoxx600_ever_members.parquet');
