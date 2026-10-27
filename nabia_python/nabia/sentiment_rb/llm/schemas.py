"""pydantic schemas of the structured outputs of prompts A (triage), B (scoring) and C (digest)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class NodeMap(BaseModel):
    node_id: str
    relevance: float = Field(ge=0, le=1)
    channel: Literal["direct", "spillover"]


class Triage(BaseModel):
    relevant: bool
    reason: Literal["new_information", "price_recap", "opinion_only", "irrelevant"]
    event_type: Literal["earnings", "guidance", "m&a", "rating_action", "default_restructuring", "litigation_regulatory",
                        "management", "central_bank", "macro_data", "fiscal_policy", "geopolitics", "supply_disruption",
                        "inventory", "opec_policy", "weather", "flows_positioning", "other"]
    nodes: list[NodeMap]


class Score(BaseModel):  # only what the features use: a short output keeps a local model fast
    node_id: str
    sentiment: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)


class Scores(BaseModel):
    scores: list[Score]


class Digest(BaseModel):
    risk_appetite: float = Field(ge=-1, le=1)
    growth_tone: float = Field(ge=-1, le=1)
    inflation_tone: float = Field(ge=-1, le=1)
    geopolitical_stress: float = Field(ge=0, le=1)
    credit_tone: float = Field(ge=-1, le=1)
    cb_tone: dict[str, float | None]
    commodity_supply_tone: dict[str, float | None]
    item_count_used: int
    key_drivers: list[str]
