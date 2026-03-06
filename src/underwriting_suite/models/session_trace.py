"""Session trace model – stores ReAct planning logs per session."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, JSON, Text, Float, func
from sqlalchemy.orm import Mapped, mapped_column

from underwriting_suite.db.database import Base


class SessionTrace(Base):
    __tablename__ = "agent_session_traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    step_index: Mapped[int] = mapped_column(default=0)
    step_type: Mapped[str] = mapped_column(
        String(30)
    )  # plan, tool_exec, reflect, synthesize
    tool_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    tool_input_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tool_output_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thought_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    success: Mapped[Optional[bool]] = mapped_column(nullable=True)
    duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
