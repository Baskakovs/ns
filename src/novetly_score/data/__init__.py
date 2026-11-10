"""Access to the experiment's data on the S3-compatible object store.

Two ways in, for two kinds of work. :class:`OpenClient` is the object store itself — listing,
fetching and writing whole keys. :func:`data` reads the datasets *through* the objects: it takes
the assets of a YUKKA universe and hands back the rows the store holds for them, reading only
the files and columns a query asks for — or, through :class:`Counts`, the daily article counts
those rows reduce to, aggregated in the store's engine rather than in Python.
:class:`OpenDuck` is the engine underneath, for a query that outgrows :func:`data`.
"""

from novetly_score.data.duck import BUCKET, OpenDuck
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
    "ConnectionFailedError",
    "Counts",
    "NoSnippetsError",
    "OpenClient",
    "OpenDuck",
    "Snippets",
    "Universe",
    "as_of",
    "data",
    "entities",
    "entity",
    "identifier_history",
]
