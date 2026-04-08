"""
Task 1: Triage Basic (Easy)

10 clear-cut customer support emails. The agent must assign the correct
category and priority. Emails have unambiguous signals.

Success threshold: 0.75
"""
from env.graders import grade_triage_basic

TASK_CONFIG = {
    "name": "triage_basic",
    "difficulty": "easy",
    "max_steps": 10,
    "success_threshold": 0.75,
    "description": (
        "Triage 10 clear-cut customer support emails by assigning the correct "
        "category (billing/technical/account/general/spam) and priority "
        "(urgent/high/medium/low). Emails have unambiguous intent signals."
    ),
    "grader": grade_triage_basic,
}


class TriageBasicTask:
    name = "triage_basic"
    difficulty = "easy"
    max_steps = 10
    success_threshold = 0.75

    task_objective = (
        "Correctly classify each email by category and priority. "
        "Watch for obvious spam; escalate only when strictly needed."
    )

    scoring_weights = {
        "category_accuracy": 0.50,
        "priority_accuracy": 0.40,
        "escalation_accuracy": 0.10,
    }

    @staticmethod
    def grade(trajectory) -> float:
        return grade_triage_basic(trajectory)
