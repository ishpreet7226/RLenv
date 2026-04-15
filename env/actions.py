"""
Action model for the Customer Support Email Triage environment.

An agent submits one action per email — its triage decision.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class Action(BaseModel):
    """
    The agent's triage decision for the current email.

    Fields
    ------
    category : str
        Primary category assigned to the email.
        One of: "billing", "technical", "account", "general", "spam"

    priority : str
        Urgency level assigned to the email.
        One of: "urgent", "high", "medium", "low"

    response_draft : Optional[str]
        An optional draft reply to the customer. Graded by keyword overlap
        with the ground-truth response keywords. Rewarded when present and
        relevant; no penalty for omitting.

    escalate : bool
        Whether the agent flags this email for human escalation.
        Correct escalation of genuinely urgent / sensitive emails is rewarded;
        false escalations incur a small penalty.
    """

    category: Literal["billing", "technical", "account", "general", "spam"] = Field(
        ...,
        description="Primary category of the email.",
    )
    priority: Literal["urgent", "high", "medium", "low"] = Field(
        ...,
        description="Urgency level for the email.",
    )
    response_draft: Optional[str] = Field(
        default=None,
        description="Optional draft reply to the sender.",
    )
    escalate: bool = Field(
        default=False,
        description="True if the email should be escalated to a human agent.",
    )
