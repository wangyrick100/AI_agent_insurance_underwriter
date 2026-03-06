"""Extraction domain model – storage for X1 outputs."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Text, Float, JSON, func
from sqlalchemy.orm import Mapped, mapped_column

from underwriting_suite.db.database import Base


class Extraction(Base):
    __tablename__ = "extractions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    applicant_id: Mapped[str] = mapped_column(String(36), index=True)
    document_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    entity_type: Mapped[str] = mapped_column(String(50))  # medication, diagnosis, lab_result, vital, procedure
    entity_name: Mapped[str] = mapped_column(String(300))
    entity_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_chunk_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    evidence_snippet: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
