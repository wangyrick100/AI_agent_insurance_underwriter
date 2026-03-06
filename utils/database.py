"""Database utilities: schema definition, seeding, and query helpers.

Uses SQLAlchemy Core with a SQLite backend by default.  The schema models
a simplified insurance underwriting data store.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    Date,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine

import config

metadata = MetaData()

# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

applicants = Table(
    "applicants",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(120), nullable=False),
    Column("age", Integer),
    Column("occupation", String(80)),
    Column("annual_income", Float),
    Column("credit_score", Integer),
    Column("years_insured", Integer),
    Column("risk_tier", String(10)),  # LOW / MEDIUM / HIGH
)

policies = Table(
    "policies",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("applicant_id", Integer, nullable=False),
    Column("policy_type", String(60)),
    Column("coverage_amount", Float),
    Column("premium", Float),
    Column("deductible", Float),
    Column("effective_date", Date),
    Column("expiry_date", Date),
    Column("status", String(20)),  # ACTIVE / EXPIRED / CANCELLED
)

claims = Table(
    "claims",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("policy_id", Integer, nullable=False),
    Column("applicant_id", Integer, nullable=False),
    Column("claim_date", Date),
    Column("amount", Float),
    Column("description", Text),
    Column("status", String(20)),  # OPEN / CLOSED / DENIED
)

risk_scores = Table(
    "risk_scores",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("applicant_id", Integer, nullable=False),
    Column("score", Float),
    Column("risk_tier", String(10)),
    Column("scored_at", String(30)),
    Column("model_version", String(20)),
)

# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

_engine: Optional[Engine] = None


def get_engine(url: Optional[str] = None) -> Engine:
    """Return (and cache) the SQLAlchemy engine."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        _engine = create_engine(url or config.DATABASE_URL, echo=False)
    return _engine


def init_db(url: Optional[str] = None) -> Engine:
    """Create all tables and return the engine."""
    engine = get_engine(url)
    metadata.create_all(engine)
    return engine


def reset_engine() -> None:
    """Clear the cached engine (used in tests)."""
    global _engine  # noqa: PLW0603
    _engine = None


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

SEED_APPLICANTS = [
    {
        "id": 1, "name": "Alice Johnson", "age": 34, "occupation": "Software Engineer",
        "annual_income": 95000, "credit_score": 780, "years_insured": 8, "risk_tier": "LOW",
    },
    {
        "id": 2, "name": "Bob Martinez", "age": 52, "occupation": "Contractor",
        "annual_income": 62000, "credit_score": 620, "years_insured": 3, "risk_tier": "HIGH",
    },
    {
        "id": 3, "name": "Carol Smith", "age": 45, "occupation": "Teacher",
        "annual_income": 58000, "credit_score": 710, "years_insured": 12, "risk_tier": "LOW",
    },
    {
        "id": 4, "name": "David Lee", "age": 29, "occupation": "Freelancer",
        "annual_income": 48000, "credit_score": 660, "years_insured": 1, "risk_tier": "MEDIUM",
    },
    {
        "id": 5, "name": "Eve Brown", "age": 38, "occupation": "Nurse",
        "annual_income": 72000, "credit_score": 740, "years_insured": 6, "risk_tier": "LOW",
    },
    {
        "id": 6, "name": "Frank Wilson", "age": 61, "occupation": "Retired",
        "annual_income": 38000, "credit_score": 590, "years_insured": 20, "risk_tier": "HIGH",
    },
    {
        "id": 7, "name": "Grace Chen", "age": 31, "occupation": "Data Scientist",
        "annual_income": 105000, "credit_score": 800, "years_insured": 5, "risk_tier": "LOW",
    },
    {
        "id": 8, "name": "Henry Park", "age": 47, "occupation": "Real Estate Agent",
        "annual_income": 88000, "credit_score": 695, "years_insured": 9, "risk_tier": "MEDIUM",
    },
]

SEED_POLICIES = [
    {
        "id": 1, "applicant_id": 1, "policy_type": "Homeowners",
        "coverage_amount": 350000, "premium": 1200, "deductible": 1000,
        "effective_date": datetime.date(2024, 1, 1), "expiry_date": datetime.date(2025, 1, 1),
        "status": "ACTIVE",
    },
    {
        "id": 2, "applicant_id": 2, "policy_type": "Commercial",
        "coverage_amount": 500000, "premium": 4500, "deductible": 5000,
        "effective_date": datetime.date(2024, 3, 15), "expiry_date": datetime.date(2025, 3, 15),
        "status": "ACTIVE",
    },
    {
        "id": 3, "applicant_id": 3, "policy_type": "Auto",
        "coverage_amount": 50000, "premium": 900, "deductible": 500,
        "effective_date": datetime.date(2023, 6, 1), "expiry_date": datetime.date(2024, 6, 1),
        "status": "EXPIRED",
    },
    {
        "id": 4, "applicant_id": 4, "policy_type": "Renters",
        "coverage_amount": 30000, "premium": 420, "deductible": 250,
        "effective_date": datetime.date(2024, 8, 1), "expiry_date": datetime.date(2025, 8, 1),
        "status": "ACTIVE",
    },
    {
        "id": 5, "applicant_id": 5, "policy_type": "Homeowners",
        "coverage_amount": 280000, "premium": 980, "deductible": 1000,
        "effective_date": datetime.date(2024, 2, 1), "expiry_date": datetime.date(2025, 2, 1),
        "status": "ACTIVE",
    },
    {
        "id": 6, "applicant_id": 6, "policy_type": "Auto",
        "coverage_amount": 25000, "premium": 1800, "deductible": 1000,
        "effective_date": datetime.date(2024, 5, 1), "expiry_date": datetime.date(2025, 5, 1),
        "status": "ACTIVE",
    },
    {
        "id": 7, "applicant_id": 7, "policy_type": "Homeowners",
        "coverage_amount": 600000, "premium": 1800, "deductible": 2500,
        "effective_date": datetime.date(2024, 1, 15), "expiry_date": datetime.date(2025, 1, 15),
        "status": "ACTIVE",
    },
    {
        "id": 8, "applicant_id": 8, "policy_type": "Commercial",
        "coverage_amount": 750000, "premium": 6200, "deductible": 10000,
        "effective_date": datetime.date(2024, 7, 1), "expiry_date": datetime.date(2025, 7, 1),
        "status": "ACTIVE",
    },
]

SEED_CLAIMS = [
    {
        "id": 1, "policy_id": 2, "applicant_id": 2,
        "claim_date": datetime.date(2024, 5, 10), "amount": 12000,
        "description": "Fire damage to warehouse roof", "status": "CLOSED",
    },
    {
        "id": 2, "policy_id": 2, "applicant_id": 2,
        "claim_date": datetime.date(2024, 9, 3), "amount": 8500,
        "description": "Theft of equipment", "status": "OPEN",
    },
    {
        "id": 3, "policy_id": 3, "applicant_id": 3,
        "claim_date": datetime.date(2023, 11, 20), "amount": 3200,
        "description": "Rear-end collision", "status": "CLOSED",
    },
    {
        "id": 4, "policy_id": 6, "applicant_id": 6,
        "claim_date": datetime.date(2024, 6, 15), "amount": 5800,
        "description": "Hail damage to vehicle", "status": "CLOSED",
    },
    {
        "id": 5, "policy_id": 8, "applicant_id": 8,
        "claim_date": datetime.date(2024, 10, 1), "amount": 22000,
        "description": "Water damage from burst pipe", "status": "OPEN",
    },
]


def seed_database(engine: Optional[Engine] = None) -> None:
    """Insert sample data if the tables are empty."""
    if engine is None:
        engine = get_engine()

    with engine.connect() as conn:
        if conn.execute(text("SELECT COUNT(*) FROM applicants")).scalar() == 0:
            conn.execute(applicants.insert(), SEED_APPLICANTS)
        if conn.execute(text("SELECT COUNT(*) FROM policies")).scalar() == 0:
            conn.execute(policies.insert(), SEED_POLICIES)
        if conn.execute(text("SELECT COUNT(*) FROM claims")).scalar() == 0:
            conn.execute(claims.insert(), SEED_CLAIMS)
        conn.commit()


def execute_query(sql: str, engine: Optional[Engine] = None) -> List[Dict[str, Any]]:
    """Execute *sql* and return results as a list of dicts.

    Only SELECT statements are permitted to prevent mutation.
    """
    stripped = sql.strip().lstrip(";").strip()
    if not stripped.upper().startswith("SELECT"):
        raise ValueError("Only SELECT statements are permitted via execute_query().")

    if engine is None:
        engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(text(sql))
        cols = list(result.keys())
        return [dict(zip(cols, row)) for row in result.fetchall()]


def get_schema_ddl() -> str:
    """Return a human-readable DDL string for all tables."""
    lines = []
    for tbl in metadata.sorted_tables:
        col_defs = []
        for col in tbl.columns:
            nullable = "" if col.nullable else " NOT NULL"
            pk = " PRIMARY KEY" if col.primary_key else ""
            col_defs.append(f"    {col.name} {col.type}{pk}{nullable}")
        lines.append(f"CREATE TABLE {tbl.name} (\n" + ",\n".join(col_defs) + "\n);")
    return "\n\n".join(lines)
