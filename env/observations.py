"""
Observation model for the Customer Support Email Triage environment.

The agent receives one Observation per email. It contains the raw email
content plus contextual metadata to help determine triage priority.
"""
from pydantic import BaseModel, Field
from typing import List


class Observation(BaseModel):
    """
    What the agent sees when processing each email.

    Fields
    ------
    email_id : str
        Unique identifier for the email (e.g. "basic_001").

    subject : str
        The email subject line.

    body : str
        The full email body text.

    sender_tier : str
        Subscription tier of the sender: "premium", "standard", or "trial".
        Higher-tier customers may warrant faster responses.

    previous_tickets : int
        Number of previous support tickets from this sender.
        High counts may indicate a serial complainer or a complex ongoing issue.

    sentiment_score : float
        Pre-computed sentiment of the email body in the range [-1.0, 1.0].
        Negative values indicate frustrated/angry tone.

    queue_position : int
        Position of this email in the current task's queue (1-indexed).

    emails_remaining : int
        Number of emails left to triage in this episode (including current).

    task_context : str
        A plain-language description of the current task's objective.

    valid_categories : List[str]
        The set of valid category labels the agent may use.

    valid_priorities : List[str]
        The set of valid priority labels the agent may use.
    """

    email_id: str = Field(..., description="Unique email identifier.")
    subject: str = Field(..., description="Email subject line.")
    body: str = Field(..., description="Full email body text.")
    sender_tier: str = Field(..., description="Sender subscription tier: premium | standard | trial.")
    previous_tickets: int = Field(..., description="Number of past support tickets from this sender.")
    sentiment_score: float = Field(..., description="Sentiment score in [-1.0, 1.0].")
    queue_position: int = Field(..., description="1-indexed position of this email in the queue.")
    emails_remaining: int = Field(..., description="Emails left to process (including current).")
    task_context: str = Field(..., description="Natural language description of the current task objective.")
    valid_categories: List[str] = Field(
        default=["billing", "technical", "account", "general", "spam"],
        description="Valid category labels.",
    )
    valid_priorities: List[str] = Field(
        default=["urgent", "high", "medium", "low"],
        description="Valid priority labels.",
    )
