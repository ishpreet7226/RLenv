"""
Reward model for the Customer Support Email Triage environment.

Each step produces a structured reward breakdown so the agent can
receive fine-grained credit for different aspects of its triage decision.
"""
from pydantic import BaseModel, Field


class Reward(BaseModel):
    """
    Structured reward breakdown for one email triage action.

    All component fields are in [0.0, 1.0] or small penalty ranges.
    step_reward is the weighted sum (also clamped to [0.0, 1.0]).

    Fields
    ------
    category_accuracy : float
        1.0 if the chosen category exactly matches ground truth; 0.0 otherwise.

    priority_accuracy : float
        1.0 = exact match; 0.5 = one level off; 0.0 = two+ levels off.
        Priority order (for partial credit): urgent > high > medium > low.

    response_quality : float
        Fraction of ground-truth response keywords present in the draft reply.
        0.0 if no draft was provided.

    escalation_bonus : float
        +0.2 bonus if the agent correctly escalated a case that should_escalate.
        0.0 otherwise (no penalty for NOT escalating).

    false_escalation_penalty : float
        -0.1 penalty if the agent escalated a case that should NOT be escalated.
        0.0 otherwise.

    spam_correct : float
        +0.1 bonus for correctly identifying spam (avoids always-spam gaming).
        -0.15 penalty for marking a real email as spam.

    step_reward : float
        Weighted combination of all components, clamped to [0.0, 1.0].
        Weights: category=0.40, priority=0.30, response=0.15, escalation=0.10, spam=0.05.

    cumulative_reward : float
        Running sum of step_reward across all steps in the episode.

    email_id : str
        Identifier of the email this reward corresponds to.
    """

    category_accuracy: float = Field(0.0, description="Category match score [0, 1].")
    priority_accuracy: float = Field(0.0, description="Priority match score [0, 1].")
    response_quality: float = Field(0.0, description="Response keyword coverage [0, 1].")
    escalation_bonus: float = Field(0.0, description="Correct escalation bonus.")
    false_escalation_penalty: float = Field(0.0, description="False escalation penalty.")
    spam_correct: float = Field(0.0, description="Spam classification reward/penalty.")
    step_reward: float = Field(0.0, description="Total step reward [0, 1].")
    cumulative_reward: float = Field(0.0, description="Running cumulative reward.")
    email_id: str = Field("", description="Email this reward belongs to.")
