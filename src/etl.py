"""Extract, transform, and load SDG indicators from the World Bank API."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import wbgapi as wb
from sqlalchemy import create_engine, text

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "sdg_data.db"
TABLE_NAME = "sdg_indicators"

# Core SDG indicators for the dashboard (World Development Indicators).
INDICATORS: dict[str, str] = {
    "SI.POV.DDAY": "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)",
    "SH.STA.MMRT": "Maternal mortality ratio (modeled estimate, per 100,000 live births)",
    "SE.XPD.TOTL.GD.ZS": "Government expenditure on education, total (% of GDP)",
    "EG.ELC.ACCS.ZS": "Access to electricity (% of population)",
}

YEARS_BACK = 20


def _year_window(years_back: int = YEARS_BACK) -> range:
    """Return an inclusive calendar-year range covering the last N years."""
    current_year = datetime.now().year
    start_year = current_year - years_back + 1
    return range(start_year, current_year + 1)


def _lookup_indicator_names(indicator_codes: list[str]) -> dict[str, str]:
    """Resolve human-readable names from the World Bank series catalog."""
    names = dict(INDICATORS)
    for code in indicator_codes:
        try:
            meta = wb.series.get(code)
            label = meta.get("value") if isinstance(meta, dict) else None
            if label:
                names[code] = label
        except Exception as exc:  # noqa: BLE001 - API metadata is optional
            LOGGER.warning("Could not resolve series name for %s: %s", code, exc)
    return names


def _parse_year(time_value: object) -> int | None:
    """Normalize World Bank time values such as YR2020 or {'id': 'YR2020'}."""
    if isinstance(time_value, dict):
        time_value = time_value.get("id") or time_value.get("value")
    if time_value is None:
        return None
    raw = str(time_value).replace("YR", "").strip()
    try:
        return int(raw)
    except ValueError:
        LOGGER.debug("Unparseable year value: %s", time_value)
        return None


def _entity_id(entity: object) -> str | None:
    if isinstance(entity, dict):
        value = entity.get("id")
        return str(value) if value is not None else None
    return str(entity) if entity is not None else None


def _entity_label(entity: object) -> str | None:
    if isinstance(entity, dict):
        value = entity.get("value") or entity.get("id")
        return str(value) if value is not None else None
    return str(entity) if entity is not None else None


def extract_world_bank_data(
    indicator_codes: list[str] | None = None,
    years_back: int = YEARS_BACK,
) -> pd.DataFrame:
    """Fetch indicator time series for all countries (aggregates excluded)."""
    codes = indicator_codes or list(INDICATORS.keys())
    years = _year_window(years_back)
    LOGGER.info(
        "Fetching %s indicators for %s–%s from the World Bank API",
        len(codes),
        years.start,
        years.stop - 1,
    )

    records = list(
        wb.data.fetch(
            codes,
            economy="all",
            time=years,
            skipAggs=True,
            labels=True,
        )
    )
    if not records:
        raise RuntimeError("World Bank API returned no records for the requested indicators.")

    return pd.DataFrame(records)


def transform_indicator_frame(raw: pd.DataFrame, indicator_codes: list[str]) -> pd.DataFrame:
    """Clean API output into a tidy table ready for SQLite."""
    if raw.empty:
        raise ValueError("Cannot transform an empty DataFrame.")

    names = _lookup_indicator_names(indicator_codes)
    frame = raw.copy()

    frame["country_code"] = frame["economy"].map(_entity_id)
    frame["country_name"] = frame["economy"].map(_entity_label)
    frame["indicator_code"] = frame["series"].map(_entity_id)
    frame["year"] = frame["time"].map(_parse_year)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["indicator_name"] = frame["indicator_code"].map(names)

    cleaned = (
        frame.loc[:, ["country_code", "country_name", "year", "indicator_code", "indicator_name", "value"]]
        .dropna(subset=["country_code", "country_name", "year", "indicator_code", "value"])
        .drop_duplicates(subset=["country_code", "year", "indicator_code"])
        .sort_values(["indicator_code", "country_name", "year"])
        .reset_index(drop=True)
    )
    cleaned["year"] = cleaned["year"].astype(int)
    cleaned["country_code"] = cleaned["country_code"].astype(str)
    cleaned["country_name"] = cleaned["country_name"].astype(str)
    cleaned["indicator_code"] = cleaned["indicator_code"].astype(str)
    cleaned["indicator_name"] = cleaned["indicator_name"].astype(str)

    LOGGER.info(
        "Cleaned dataset: %s rows, %s countries, %s indicators",
        len(cleaned),
        cleaned["country_name"].nunique(),
        cleaned["indicator_code"].nunique(),
    )
    return cleaned


def load_to_sqlite(frame: pd.DataFrame, db_path: Path = DB_PATH, table_name: str = TABLE_NAME) -> Path:
    """Replace the SQLite table with the cleaned indicator dataset."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.begin() as connection:
        frame.to_sql(table_name, connection, if_exists="replace", index=False)
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_sdg_indicators_country ON sdg_indicators (country_name)"))
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_sdg_indicators_lookup "
                "ON sdg_indicators (country_name, indicator_code, year)"
            )
        )

    LOGGER.info("Wrote %s rows to %s [%s]", len(frame), db_path, table_name)
    return db_path


def run_etl() -> Path:
    """Execute the full extract → transform → load pipeline."""
    codes = list(INDICATORS.keys())
    raw = extract_world_bank_data(codes)
    cleaned = transform_indicator_frame(raw, codes)
    return load_to_sqlite(cleaned)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    db_path = run_etl()
    print(f"ETL complete. Database saved to: {db_path}")


if __name__ == "__main__":
    main()
