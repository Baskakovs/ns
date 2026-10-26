"""Access to the experiment's data on the S3-compatible object store."""

from novetly_score.data.s3 import ConnectionFailedError, OpenClient

__all__ = ["ConnectionFailedError", "OpenClient"]
