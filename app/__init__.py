"""FinDoc Guard web service (Phase 8): REST API, job queue and review UI hosting.

Run with ``findoc api serve`` (API + an in-process worker) or, in Docker Compose, one API
container and any number of ``findoc api worker`` containers sharing PostgreSQL.
"""

from findoc import __version__

__all__ = ["__version__"]
