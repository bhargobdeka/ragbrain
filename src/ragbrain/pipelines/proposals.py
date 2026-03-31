"""ProposalStore — persistent store for architecture upgrade proposals.

Proposals flow through these states:
    pending  → approved  → implemented
                        → failed
             → skipped

The store persists to ~/.ragbrain/proposals.json so proposals survive
scheduler restarts and are visible across bot/scheduler sessions.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

_PROPOSALS_PATH = Path(os.path.expanduser("~/.ragbrain/proposals.json"))

ProposalStatus = str  # "pending" | "approved" | "skipped" | "implemented" | "failed"


@dataclass
class Proposal:
    title: str
    description: str
    implementation_plan: str
    component: str = ""
    priority: str = "MEDIUM"       # HIGH / MEDIUM / LOW
    news_signal: str = ""          # what Tuk news triggered this
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    status: ProposalStatus = "pending"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    implemented_at: str | None = None
    commit_sha: str | None = None
    result_summary: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Proposal":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def short_summary(self) -> str:
        """One-line summary for Telegram messages."""
        return f"[{self.priority}] {self.title} ({self.component})"

    def telegram_detail(self) -> str:
        """Mobile-friendly detail block for Telegram (HTML parse_mode safe)."""
        def _esc(text: str) -> str:
            """Escape <, >, & in user-supplied text to avoid Telegram parse errors."""
            return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = [
            f"<b>{_esc(self.title)}</b>",
            f"<i>Component:</i> {_esc(self.component)}  |  <i>Priority:</i> {self.priority}",
            "",
            f"<i>Why:</i> {_esc(self.description)}",
            "",
        ]
        if self.news_signal:
            lines += [f"<i>Triggered by:</i> {_esc(self.news_signal[:200])}", ""]
        lines += [
            "<i>Implementation plan:</i>",
            _esc(self.implementation_plan[:600]),
        ]
        return "\n".join(lines)


class ProposalStore:
    """Thread-safe JSON-backed store for upgrade proposals.

    Usage::

        store = ProposalStore()
        store.add(Proposal(title="Add streaming", ...))
        pending = store.list_pending()
        store.approve("a1b2c3")
    """

    def __init__(self, path: Path | str = _PROPOSALS_PATH) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Persistence -----------------------------------------------------

    def _load(self) -> list[Proposal]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [Proposal.from_dict(d) for d in data]
        except Exception:
            logger.exception("Failed to load proposals from %s", self._path)
            return []

    def _save(self, proposals: list[Proposal]) -> None:
        try:
            self._path.write_text(
                json.dumps([p.to_dict() for p in proposals], indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("Failed to save proposals to %s", self._path)

    # ---- CRUD ------------------------------------------------------------

    def add(self, proposal: Proposal) -> Proposal:
        """Persist a new proposal (status=pending)."""
        proposals = self._load()
        proposals.append(proposal)
        self._save(proposals)
        logger.info("Added proposal %s: %s", proposal.id, proposal.title)
        return proposal

    def get(self, proposal_id: str) -> Proposal | None:
        for p in self._load():
            if p.id == proposal_id:
                return p
        return None

    def _update(self, proposal_id: str, **kwargs) -> Proposal | None:
        proposals = self._load()
        for p in proposals:
            if p.id == proposal_id:
                for k, v in kwargs.items():
                    setattr(p, k, v)
                self._save(proposals)
                return p
        logger.warning("Proposal %s not found", proposal_id)
        return None

    def approve(self, proposal_id: str) -> Proposal | None:
        return self._update(proposal_id, status="approved")

    def skip(self, proposal_id: str) -> Proposal | None:
        return self._update(proposal_id, status="skipped")

    def mark_implemented(
        self, proposal_id: str, commit_sha: str = "", summary: str = ""
    ) -> Proposal | None:
        return self._update(
            proposal_id,
            status="implemented",
            implemented_at=datetime.now(timezone.utc).isoformat(),
            commit_sha=commit_sha,
            result_summary=summary,
        )

    def mark_failed(self, proposal_id: str, reason: str = "") -> Proposal | None:
        return self._update(proposal_id, status="failed", result_summary=reason)

    # ---- Queries ---------------------------------------------------------

    def list_pending(self) -> list[Proposal]:
        return [p for p in self._load() if p.status == "pending"]

    def list_approved(self) -> list[Proposal]:
        return [p for p in self._load() if p.status == "approved"]

    def list_all(self) -> list[Proposal]:
        return self._load()

    def status_summary(self) -> str:
        """Telegram-friendly status table."""
        all_p = self._load()
        if not all_p:
            return "No proposals yet."

        lines = ["<b>Proposal Status</b>", ""]
        buckets = {
            "pending": "Pending",
            "approved": "Approved",
            "implemented": "Implemented",
            "failed": "Failed",
            "skipped": "Skipped",
        }
        for status, label in buckets.items():
            group = [p for p in all_p if p.status == status]
            if not group:
                continue
            lines.append(f"<b>{label} ({len(group)})</b>")
            for p in group[-3:]:   # show last 3 per bucket
                lines.append(f"  #{p.id}  {p.title[:50]}")
            lines.append("")

        return "\n".join(lines).strip()


# Module-level singleton
_store: ProposalStore | None = None


def get_store() -> ProposalStore:
    global _store
    if _store is None:
        _store = ProposalStore()
    return _store
