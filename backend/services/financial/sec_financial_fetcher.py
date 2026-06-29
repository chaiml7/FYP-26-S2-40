"""Fetch and normalize historical quarterly statements from SEC companyfacts."""

from datetime import datetime, timezone
import re
import time

import requests


SEC_BASE_URL = "https://data.sec.gov"
SEC_WWW_URL = "https://www.sec.gov"
USER_AGENT = "StockLens student project contact: no-reply@example.com"
QUARTERLY_FRAME_PATTERN = re.compile(r"^CY\d{4}Q[1-4]$")
INSTANT_FRAME_PATTERN = re.compile(r"^CY\d{4}Q[1-4]I$")

FIELD_TAGS = {
    "total_revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": (
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ),
    "ebitda": (),
    "research_development": ("ResearchAndDevelopmentExpense",),
    "interest_expense": (
        "InterestExpenseNonOperating",
        "InterestExpense",
    ),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "total_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "cash_and_equivalents": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashCashEquivalentsAndShortTermInvestments",
    ),
    "total_debt": (
        "ShortTermBorrowings",
        "ShortTermDebtCurrent",
        "LongTermDebtCurrent",
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtNoncurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
    ),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "operating_cashflow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capex": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
    "free_cashflow": (),
    "investing_cashflow": ("NetCashProvidedByUsedInInvestingActivities",),
    "financing_cashflow": ("NetCashProvidedByUsedInFinancingActivities",),
}

SUM_FIELDS = {"total_debt"}
CASHFLOW_FIELDS = {
    "operating_cashflow",
    "capex",
    "free_cashflow",
    "investing_cashflow",
    "financing_cashflow",
}
INSTANT_FIELDS = {
    "total_assets",
    "total_liabilities",
    "total_equity",
    "cash_and_equivalents",
    "total_debt",
    "current_assets",
    "current_liabilities",
}
NEGATE_FIELDS = {"capex"}
CORE_FIELDS = (
    "total_revenue",
    "total_assets",
    "total_equity",
    "current_liabilities",
)


def _sec_get_json(url: str) -> dict:
    response = requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
            "Host": url.split("/")[2],
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _ticker_cik_map() -> dict:
    data = _sec_get_json(f"{SEC_WWW_URL}/files/company_tickers.json")
    return {
        row["ticker"].upper(): str(row["cik_str"]).zfill(10)
        for row in data.values()
    }


def _facts_for_tag(company_facts: dict, tag: str) -> list:
    facts = (
        company_facts
        .get("facts", {})
        .get("us-gaap", {})
        .get(tag, {})
        .get("units", {})
    )
    rows = []
    for unit_rows in facts.values():
        rows.extend(unit_rows)
    return rows


def _is_usable_fact(fact: dict, instant: bool) -> bool:
    frame = fact.get("frame")
    pattern = INSTANT_FRAME_PATTERN if instant else QUARTERLY_FRAME_PATTERN
    return (
        fact.get("val") is not None
        and fact.get("end")
        and fact.get("form") in {"10-Q", "10-K"}
        and isinstance(frame, str)
        and pattern.fullmatch(frame)
    )


def _period_key(fact: dict, instant: bool) -> str:
    frame = fact["frame"]
    return frame[:-1] if instant else frame


def _calendar_period_key(end_date: str) -> str:
    parsed = datetime.strptime(end_date, "%Y-%m-%d")
    quarter = ((parsed.month - 1) // 3) + 1
    return f"CY{parsed.year}Q{quarter}"


def _select_latest(existing: dict | None, candidate: dict) -> dict:
    if existing is None:
        return candidate
    return candidate if candidate.get("filed", "") >= existing.get("filed", "") else existing


def _latest_ytd_facts(company_facts: dict, field: str) -> dict:
    selected = {}
    for tag in FIELD_TAGS[field]:
        for fact in _facts_for_tag(company_facts, tag):
            if (
                fact.get("val") is None
                or not fact.get("end")
                or fact.get("form") != "10-Q"
                or fact.get("fp") not in {"Q1", "Q2", "Q3"}
                or fact.get("fy") is None
            ):
                continue

            key = (int(fact["fy"]), fact["fp"])
            value = float(fact["val"])
            if field in NEGATE_FIELDS:
                value = -abs(value)
            selected[key] = _select_latest(
                selected.get(key),
                {
                    "value": value,
                    "filed": fact.get("filed", ""),
                    "end": fact["end"],
                },
            )

    return selected


def _quarterly_from_ytd(company_facts: dict, field: str) -> dict:
    selected = _latest_ytd_facts(company_facts, field)
    values = {}
    for fiscal_year in sorted({key[0] for key in selected}):
        previous_value = 0.0
        for quarter in ("Q1", "Q2", "Q3"):
            fact = selected.get((fiscal_year, quarter))
            if fact is None:
                continue
            quarter_value = fact["value"] - previous_value
            values[_calendar_period_key(fact["end"])] = quarter_value
            previous_value = fact["value"]
    return values


def _field_values(company_facts: dict, field: str) -> dict:
    values_by_period = {}
    instant = field in INSTANT_FIELDS

    for tag in FIELD_TAGS[field]:
        for fact in _facts_for_tag(company_facts, tag):
            if not _is_usable_fact(fact, instant):
                continue

            period_key = _period_key(fact, instant)
            value = float(fact["val"])
            if field in NEGATE_FIELDS:
                value = -abs(value)

            if field in SUM_FIELDS:
                values_by_period[period_key] = (
                    values_by_period.get(period_key, 0.0) + value
                )
                continue

            existing = values_by_period.get(period_key)
            selected = _select_latest(
                existing,
                {"value": value, "filed": fact.get("filed", "")},
            )
            values_by_period[period_key] = selected

    if field in SUM_FIELDS:
        values = values_by_period
    else:
        values = {
        period_key: selected["value"]
        for period_key, selected in values_by_period.items()
        }

    if field in CASHFLOW_FIELDS:
        for period_key, value in _quarterly_from_ytd(company_facts, field).items():
            values.setdefault(period_key, value)

    return values


def _period_ends(company_facts: dict) -> dict:
    selected_by_period = {}
    for field in FIELD_TAGS:
        instant = field in INSTANT_FIELDS
        for tag in FIELD_TAGS[field]:
            for fact in _facts_for_tag(company_facts, tag):
                if not _is_usable_fact(fact, instant):
                    continue
                period_key = _period_key(fact, instant)
                selected_by_period[period_key] = _select_latest(
                    selected_by_period.get(period_key),
                    {"value": fact["end"], "filed": fact.get("filed", "")},
                )

    return {
        period_key: selected["value"]
        for period_key, selected in selected_by_period.items()
    }


def _derive_free_cashflow(row: dict) -> None:
    operating_cashflow = row["operating_cashflow"]
    capex = row["capex"]
    if operating_cashflow is not None and capex is not None:
        row["free_cashflow"] = operating_cashflow + capex


def _derive_total_liabilities(row: dict) -> None:
    if row["total_liabilities"] is not None:
        return
    total_assets = row["total_assets"]
    total_equity = row["total_equity"]
    if total_assets is not None and total_equity is not None:
        row["total_liabilities"] = total_assets - total_equity


def _invalid_core_fields(row: dict) -> list:
    return [
        field
        for field in CORE_FIELDS
        if row[field] is None or row[field] == 0
    ]


def fetch_sec_quarterly_financial_statements(symbol: str, stock_id: int) -> dict:
    """Fetch older quarterly SEC companyfacts rows for one US-listed ticker."""
    symbol = symbol.upper()
    cik = _ticker_cik_map().get(symbol)
    if cik is None:
        return {
            "symbol": symbol,
            "cik": None,
            "rows": [],
            "skipped_periods": [],
            "message": "Ticker was not found in the SEC company ticker list.",
        }

    time.sleep(0.12)
    company_facts = _sec_get_json(
        f"{SEC_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
    )
    values = {
        field: _field_values(company_facts, field)
        for field in FIELD_TAGS
    }
    period_ends = _period_ends(company_facts)
    periods = sorted(
        {
            period
            for field_values in values.values()
            for period in field_values
        }
    )

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    skipped_periods = []
    for period in periods:
        row = {
            "stock_id": stock_id,
            "ticker": symbol,
            "period": period_ends.get(
                period,
                f"{period[:4]}-{(int(period[-1]) * 3):02d}-01",
            ),
            "period_type": "quarterly",
            "fetched_at": fetched_at,
        }
        for field in FIELD_TAGS:
            row[field] = values[field].get(period)

        _derive_total_liabilities(row)
        _derive_free_cashflow(row)
        missing_core_fields = _invalid_core_fields(row)
        if missing_core_fields:
            skipped_periods.append({
                "period": row["period"],
                "sec_period": period,
                "missing_core_fields": missing_core_fields,
            })
            continue

        rows.append(row)

    return {
        "symbol": symbol,
        "cik": cik,
        "rows": rows,
        "skipped_periods": skipped_periods,
    }
