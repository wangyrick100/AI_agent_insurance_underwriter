"""Risk score domain model – storage for X2 outputs."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Float, JSON, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from underwriting_suite.db.database import Base


class RiskScore(Base):
    __tablename__ = "risk_scores"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    applicant_id: Mapped[str] = mapped_column(String(36), index=True)
    score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    risk_class: Mapped[str] = mapped_column(String(50))  # preferred, standard, substandard, decline
    feature_rationale: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    similar_cases: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    model_version: Mapped[str] = mapped_column(String(50), default="v1.0")
    disclaimers: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
