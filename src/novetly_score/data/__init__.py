"""Access to the experiment's data on the S3-compatible object store.

:class:`OpenClient` is the object store itself — listing, fetching and writing whole keys.
:func:`snippets` reads the datasets *through* the objects: it takes the assets of a YUKKA
universe and hands back the rows the store holds for them, fetching only the files and columns
a query asks for. :class:`OpenDuck` is the engine underneath, for a query that outgrows it.
"""

from novetly_score.data.duck import BUCKET, OpenDuck
from novetly_score.data.s3 import ConnectionFailedError, OpenClient
from novetly_score.data.snippets import COLUMNS, NoSnippetsError, entities, entity, paths, snippets

__all__ = [
    "BUCKET",
    "COLUMNS",
    "ConnectionFailedError",
    "NoSnippetsError",
    "OpenClient",
    "OpenDuck",
    "entities",
    "entity",
    "paths",
    "snippets",
]
