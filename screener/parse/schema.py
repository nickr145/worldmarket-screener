"""Structured company profile parsed from a free-text description. The profile does exactly three things downstream:
selects the industry model, and removes existing_markets and home_country from the candidate list (display-only: size band).
It is never a model feature."""
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

INDUSTRIES = ("non_bank_financials", "software", "retail", "healthcare")
ISO2 = re.compile(r"^[A-Z]{2}$")


class CompanyProfile(BaseModel):
    status: Literal["supported", "unsupported", "needs_info"]
    industry: Optional[Literal["non_bank_financials", "software", "retail", "healthcare"]] = None
    unsupported_reason: Optional[str] = None
    home_country: Optional[str] = None
    existing_markets: list[str] = Field(default_factory=list)
    revenue_usd: Optional[float] = None
    operating_margin: Optional[float] = None  # fraction, 0.25 = 25%
    business_model: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "low"
    unparsed_notes: list[str] = Field(default_factory=list)

    @field_validator("home_country")
    @classmethod
    def _iso_home(cls, v):
        v = v.strip().upper() if isinstance(v, str) else v
        if v is not None and not ISO2.match(v):
            raise ValueError(f"home_country must be an ISO 3166-1 alpha-2 code, got {v!r}")
        return v

    @field_validator("existing_markets")
    @classmethod
    def _iso_list(cls, v):
        out = []
        for x in v:
            x = x.strip().upper()
            if not ISO2.match(x):
                raise ValueError(f"existing_markets entries must be ISO alpha-2 codes, got {x!r}")
            if x not in out:
                out.append(x)
        return out

    @field_validator("revenue_usd")
    @classmethod
    def _rev(cls, v):
        if v is not None and v < 0:
            raise ValueError("revenue_usd must be >= 0")
        return v

    @field_validator("operating_margin")
    @classmethod
    def _margin(cls, v):
        if v is not None and not -5 <= v <= 1:
            raise ValueError("operating_margin must be a fraction between -5 and 1 (25% = 0.25)")
        return v

    @model_validator(mode="after")
    def _consistent(self):
        self.existing_markets = [m for m in self.existing_markets if m != self.home_country]  # home market is never an "existing market"
        if self.status == "supported" and self.industry is None:
            raise ValueError("supported profile needs an industry")
        if self.status != "supported" and self.industry is not None:
            raise ValueError("industry must be null unless status is supported")
        if self.status == "unsupported" and not self.unsupported_reason:
            raise ValueError("unsupported profile needs a reason")
        return self

    @property
    def size_band(self) -> Optional[str]:  # display-only, derived
        r = self.revenue_usd
        return None if r is None else "small (<$50M)" if r < 5e7 else "mid ($50M-$500M)" if r < 5e8 else "large ($500M-$5B)" if r < 5e9 else "very large (>$5B)"

    def missing_fields(self) -> list[str]:
        want = ["industry", "home_country", "revenue_usd", "operating_margin"]
        return [f for f in want if getattr(self, f) is None]
