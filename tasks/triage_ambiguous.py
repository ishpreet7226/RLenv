"""
Task 2: Triage Ambiguous (Medium)

10 context-dependent customer support emails. Emails may mention multiple
issues, buried signals, high-ticket-count senders, or compliance concerns.
The agent must identify the PRIMARY issue and assign correct triage.

Success threshold: 0.70
"""
from env.graders import grade_triage_ambiguous

TASK_CONFIG = {
    "name": "triage_ambiguous",
    "difficulty": "medium",
    "max_steps": 10,
    "success_threshold": 0.70,
    "description": (
        "Triage 10 ambiguous customer support emails. Identify the primary issue "
        "when multiple topics are mentioned. Consider sender tier, ticket history, "
        "and implicit urgency signals. Escalate appropriately."
    ),
    "grader": grade_triage_ambiguous,
}


class TriageAmbiguousTask:
    name = "triage_ambiguous"
    difficulty = "medium"
    max_steps = 10
    success_threshold = 0.70

    task_objective = (
        "Identify the PRIMARY issue in each ambiguous email and assign its "
        "correct category and priority. Response quality begins to matter. "
        "Escalate when the situation genuinely requires human intervention."
    )

    scoring_weights = {
        "category_accuracy": 0.40,
        "priority_accuracy": 0.30,
        "escalation_accuracy": 0.20,
        "response_quality": 0.10,
    }

    @staticmethod
    def grade(trajectory) -> float:
        return grade_triage_ambiguous(trajectory)
