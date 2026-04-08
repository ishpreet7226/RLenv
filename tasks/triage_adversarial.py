"""
Task 3: Triage Adversarial (Hard)

10 adversarial customer support emails designed to mislead the agent:
  - Emotional language inflating urgency (trial users with big drama)
  - Vendor spam disguised as legitimate invoices
  - Hidden technical issues buried in positive messages
  - Security/legal notices requiring immediate escalation
  - GDPR right-to-erasure requests with legal implications

An expert human agent should achieve ~0.85. Frontier LLMs typically reach
~0.70–0.80. A naive classifier that always picks "billing/high" scores ~0.35.

Success threshold: 0.65
"""
from env.graders import grade_triage_adversarial

TASK_CONFIG = {
    "name": "triage_adversarial",
    "difficulty": "hard",
    "max_steps": 10,
    "success_threshold": 0.65,
    "description": (
        "Triage 10 adversarial support emails crafted to mislead. Resist "
        "emotional urgency from low-tier users, catch spam disguised as invoices, "
        "identify hidden technical issues, and handle legal/security escalations. "
        "Draft concise professional responses for complex cases."
    ),
    "grader": grade_triage_adversarial,
}


class TriageAdversarialTask:
    name = "triage_adversarial"
    difficulty = "hard"
    max_steps = 10
    success_threshold = 0.65

    task_objective = (
        "Navigate adversarial emails: don't over-escalate dramatic trial users, "
        "catch vendor spam, find hidden bugs in positive messages, handle "
        "security vulnerability reports, and process GDPR deletion requests. "
        "Response drafts are heavily weighted at this difficulty."
    )

    scoring_weights = {
        "category_accuracy": 0.35,
        "priority_accuracy": 0.25,
        "escalation_accuracy": 0.25,
        "response_quality": 0.15,
    }

    @staticmethod
    def grade(trajectory) -> float:
        return grade_triage_adversarial(trajectory)
