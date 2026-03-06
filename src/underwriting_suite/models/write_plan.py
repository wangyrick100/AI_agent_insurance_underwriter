"""Write plan domain model – X5 write workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, JSON, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from underwriting_suite.db.database import Base


class WritePlan(Base):
    __tablename__ = "write_plans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    applicant_id: Mapped[str] = mapped_column(String(36), index=True)
    status: Mapped[str] = mapped_column(
        String(30), default="pending"
    )  # pending, confirmed, committed, rejected, failed
    sql_statements: Mapped[list] = mapped_column(JSON)  # list of SQL statements to execute
    diff_preview: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    impacted_tables: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    impacted_row_count: Mapped[int] = mapped_column(default=0)
    confirmation_token: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    confirmed_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    committed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
