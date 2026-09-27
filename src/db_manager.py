"""SQLite access layer for the SDG Impact Dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sdg_data.db"
TABLE_NAME = "sdg_indicators"


def get_engine(db_path: Path = DB_PATH) -> Engine:
    """Return a SQLAlchemy engine for the local SQLite database."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run `python -m src.etl` first."
        )
    return create_engine(f"sqlite:///{db_path}")


def get_available_regions(db_path: Path = DB_PATH) -> list[str]:
    """Return sorted unique country names stored in the database."""
    query = text(
        """
        SELECT DISTINCT country_name
        FROM sdg_indicators
        WHERE country_name IS NOT NULL
        ORDER BY country_name
        """
    )
    engine = get_engine(db_path)
    with engine.connect() as connection:
        rows = connection.execute(query).fetchall()
    return [row[0] for row in rows]


def get_indicator_data(
    country_name: str,
    indicator_code: str,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Return the time series for one country and indicator as a DataFrame."""
    query = text(
        """
        SELECT
            country_code,
            country_name,
            year,
            indicator_code,
            indicator_name,
            value
        FROM sdg_indicators
        WHERE country_name = :country_name
          AND indicator_code = :indicator_code
        ORDER BY year
        """
    )
    engine = get_engine(db_path)
    with engine.connect() as connection:
        frame = pd.read_sql(
            query,
            connection,
            params={"country_name": country_name, "indicator_code": indicator_code},
        )
    if "year" in frame.columns:
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    if "value" in frame.columns:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def _read_indicator_frame(query, params: dict, db_path: Path) -> pd.DataFrame:
    engine = get_engine(db_path)
    with engine.connect() as connection:
        frame = pd.read_sql(query, connection, params=params)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def get_country_indicators(
    country_name: str,
    indicator_codes: list[str],
    start_year: int,
    end_year: int,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Return several indicators for one country within a year range (long format)."""
    query = text(
        """
        SELECT country_code, country_name, year, indicator_code, indicator_name, value
        FROM sdg_indicators
        WHERE country_name = :country_name
          AND indicator_code IN :indicator_codes
          AND year BETWEEN :start_year AND :end_year
        ORDER BY indicator_code, year
        """
    ).bindparams(bindparam("indicator_codes", expanding=True))
    params = {
        "country_name": country_name,
        "indicator_codes": list(indicator_codes),
        "start_year": int(start_year),
        "end_year": int(end_year),
    }
    return _read_indicator_frame(query, params, db_path)


def get_countries_indicator(
    country_names: list[str],
    indicator_code: str,
    start_year: int,
    end_year: int,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Return one indicator for several countries within a year range (long format)."""
    query = text(
        """
        SELECT country_code, country_name, year, indicator_code, indicator_name, value
        FROM sdg_indicators
        WHERE country_name IN :country_names
          AND indicator_code = :indicator_code
          AND year BETWEEN :start_year AND :end_year
        ORDER BY country_name, year
        """
    ).bindparams(bindparam("country_names", expanding=True))
    params = {
        "country_names": list(country_names),
        "indicator_code": indicator_code,
        "start_year": int(start_year),
        "end_year": int(end_year),
    }
    return _read_indicator_frame(query, params, db_path)


def get_year_bounds(db_path: Path = DB_PATH) -> tuple[int, int]:
    """Return the minimum and maximum years available in the database."""
    query = text("SELECT MIN(year) AS min_year, MAX(year) AS max_year FROM sdg_indicators")
    engine = get_engine(db_path)
    with engine.connect() as connection:
        row = connection.execute(query).mappings().one()
    min_year = int(row["min_year"])
    max_year = int(row["max_year"])
    return min_year, max_year


def get_country_code(country_name: str, db_path: Path = DB_PATH) -> str | None:
    """Return the ISO-3 country code for a stored country name."""
    query = text(
        """
        SELECT country_code
        FROM sdg_indicators
        WHERE country_name = :country_name
        LIMIT 1
        """
    )
    engine = get_engine(db_path)
    with engine.connect() as connection:
        row = connection.execute(query, {"country_name": country_name}).fetchone()
    return str(row[0]) if row and row[0] is not None else None


def get_global_indicator_snapshot(
    indicator_code: str,
    year: int | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Return all-country values for an indicator in one year (latest if omitted)."""
    engine = get_engine(db_path)
    with engine.connect() as connection:
        if year is None:
            # Prefer the most recent year with the broadest country coverage.
            year_row = connection.execute(
                text(
                    """
                    SELECT year AS latest_year
                    FROM sdg_indicators
                    WHERE indicator_code = :indicator_code
                      AND value IS NOT NULL
                    GROUP BY year
                    ORDER BY COUNT(*) DESC, year DESC
                    LIMIT 1
                    """
                ),
                {"indicator_code": indicator_code},
            ).mappings().one_or_none()
            year = year_row["latest_year"] if year_row else None
            if year is None:
                return pd.DataFrame(
                    columns=["country_code", "country_name", "year", "indicator_code", "indicator_name", "value"]
                )

        frame = pd.read_sql(
            text(
                """
                SELECT
                    country_code,
                    country_name,
                    year,
                    indicator_code,
                    indicator_name,
                    value
                FROM sdg_indicators
                WHERE indicator_code = :indicator_code
                  AND year = :year
                ORDER BY country_name
                """
            ),
            connection,
            params={"indicator_code": indicator_code, "year": int(year)},
        )

    if "year" in frame.columns:
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    if "value" in frame.columns:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame
