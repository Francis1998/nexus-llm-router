"""Durable routing audit log."""

from pathlib import Path

from router.schemas import AuditRecord


class AuditLog:
    """Append-only JSONL audit log for routing decisions."""

    def __init__(self, path: str) -> None:
        """Initialize the audit log.

        Args:
            path: JSONL file path.
        """
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: AuditRecord) -> None:
        """Append one audit record.

        Args:
            record: Audit record to persist.
        """
        with self._path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(record.model_dump_json())
            audit_file.write("\n")
