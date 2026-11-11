"""Access to the experiment's data on the S3-compatible object store.

Two ways in, for two kinds of work. :class:`OpenClient` is the object store itself — listing,
fetching and writing whole keys. :func:`data` reads the datasets a universe of assets *has*: the
snippet rows the store holds for them, the daily article counts those rows reduce to
(:meth:`~novetly_score.data.Universe.counts`, aggregated in the store's engine rather than in
Python), and the EUR levels they are measured against
(:meth:`~novetly_score.data.Universe.prices`, from YUKKA's LSEG snapshot). Both panels can be
keyed by the identifier each company held on each day, so they join without a bridge per caller.

:class:`OpenDuck` is the engine underneath, for a query that outgrows :func:`data`.
"""

from novetly_score.data.duck import BUCKET, OpenDuck
from novetly_score.data.panels import (
    COUNTS_KEY,
    STAMP_KEY,
    StalePanelError,
    read_panel,
    stamp,
    write_counts,
)
from novetly_score.data.s3 import ConnectionFailedError, OpenClient
from novetly_score.data.snippets import (
    Counts,
    NoSnippetsError,
    Snippets,
    Universe,
    as_of,
    data,
    entities,
    entity,
    identifier_history,
)

__all__ = [
    "BUCKET",
    "COUNTS_KEY",
    "STAMP_KEY",
    "ConnectionFailedError",
    "Counts",
    "NoSnippetsError",
    "OpenClient",
    "OpenDuck",
    "Snippets",
    "StalePanelError",
    "Universe",
    "as_of",
    "data",
    "entities",
    "entity",
    "identifier_history",
    "read_panel",
    "stamp",
    "write_counts",
]
