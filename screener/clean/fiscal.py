"""Fiscal-year alignment and reporting-lag rules (documented in PLAN.md, Week 2).

fiscal_year: year of the fiscal-year-end (FYE) if FYE month >= fy_cutoff_month (default 6), else year - 1.
  Mar-2024 FYE (India/Japan) -> FY2023; Jun-2024 -> FY2024; Dec-2024 -> FY2024. (Compustat-style.)
as_of_date: FYE + filing_lag_days. The panel builder must use as_of_date, never FYE, to decide what is known at t.
"""
import pandas as pd


def fiscal_year(fye, cutoff_month=6):
    fye = pd.Timestamp(fye)
    return fye.year if fye.month >= cutoff_month else fye.year - 1


def as_of_date(fye, lag_days=120):
    return pd.Timestamp(fye) + pd.Timedelta(days=lag_days)


def known_at(fye, t, lag_days=120):
    """True if a figure for the FYE is public by date t."""
    return as_of_date(fye, lag_days) <= pd.Timestamp(t)
