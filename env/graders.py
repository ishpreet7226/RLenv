"""
Graders for the Customer Support Email Triage environment.

Three deterministic grader functions, one per task:
  grade_triage_basic(trajectory)       → float in [0.0, 1.0]
  grade_triage_ambiguous(trajectory)   → float in [0.0, 1.0]
  grade_triage_adversarial(trajectory) → float in [0.0, 1.0]

All graders operate on the trajectory list returned by env.get_trajectory().

Each trajectory item has:
  {
    "email_id": str,
    "ground_truth_category": str,
    "ground_truth_priority": str,
    "pred_category": str,
    "pred_priority": str,
    "should_escalate": bool,
    "escalated": bool,
    "response_quality": float,    # [0, 1] keyword coverage
    "step_reward": float,         # step-level reward
  }
"""
from typing import List, Dict, Any

PRIORITY_ORDER = {"urgent": 0, "high": 1, "medium": 2, "low": 3}


def _category_score(step: Dict[str, Any]) -> float:
    return 1.0 if step["pred_category"] == step["ground_truth_category"] else 0.0


def _priority_score(step: Dict[str, Any]) -> float:
    pred_idx = PRIORITY_ORDER.get(step["pred_priority"], 2)
    gt_idx = PRIORITY_ORDER.get(step["ground_truth_priority"], 2)
    diff = abs(pred_idx - gt_idx)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.5
    elif diff == 2:
        return 0.2
    return 0.0


def _escalation_score(step: Dict[str, Any]) -> float:
    if step["should_escalate"] and step["escalated"]:
        return 1.0
    elif not step["should_escalate"] and step["escalated"]:
        return -0.5  # false escalation penalty
    elif step["should_escalate"] and not step["escalated"]:
        return 0.0   # missed escalation — no bonus
    return 0.0


# ──────────────────────────────────────────────────────────────────────────
# Task 1: grade_triage_basic
# Weights: category 50%, priority 40%, escalation 10%
# No response quality required for easy task
# ──────────────────────────────────────────────────────────────────────────

def grade_triage_basic(trajectory: List[Dict[str, Any]]) -> float:
    """
    Grade the basic email triage task.

    Scoring:
      - Category accuracy: 50% of score
      - Priority accuracy: 40% of score (with partial credit)
      - Escalation accuracy: 10% of score

    Returns a float in [0.0, 1.0].
    """
    if not trajectory:
        return 0.0

    cat_scores = []
    pri_scores = []
    esc_scores = []

    for step in trajectory:
        cat_scores.append(_category_score(step))
        pri_scores.append(_priority_score(step))
        esc_scores.append(max(0.0, _escalation_score(step)))

    avg_cat = sum(cat_scores) / len(cat_scores)
    avg_pri = sum(pri_scores) / len(pri_scores)
    avg_esc = sum(esc_scores) / max(1, sum(1 for s in trajectory if s["should_escalate"]))

    score = 0.50 * avg_cat + 0.40 * avg_pri + 0.10 * avg_esc
    return max(0.0, min(1.0, score))


# ──────────────────────────────────────────────────────────────────────────
# Task 2: grade_triage_ambiguous
# Weights: category 40%, priority 30%, escalation 20%, response 10%
# Response quality starts to matter at medium difficulty
# ──────────────────────────────────────────────────────────────────────────

def grade_triage_ambiguous(trajectory: List[Dict[str, Any]]) -> float:
    """
    Grade the ambiguous email triage task.

    Scoring:
      - Category accuracy: 40% of score
      - Priority accuracy: 30% of score (with partial credit)
      - Escalation accuracy: 20% of score (false escalations penalized)
      - Response quality:   10% of score

    Returns a float in [0.0, 1.0].
    """
    if not trajectory:
        return 0.0

    cat_scores = []
    pri_scores = []
    esc_scores = []
    resp_scores = []

    for step in trajectory:
        cat_scores.append(_category_score(step))
        pri_scores.append(_priority_score(step))
        raw_esc = _escalation_score(step)
        esc_scores.append(raw_esc)
        resp_scores.append(step.get("response_quality", 0.0))

    avg_cat = sum(cat_scores) / len(cat_scores)
    avg_pri = sum(pri_scores) / len(pri_scores)
    avg_esc = min(1.0, max(0.0, sum(esc_scores) / len(esc_scores) + 0.5))  # normalize around 0
    avg_resp = sum(resp_scores) / len(resp_scores)

    score = 0.40 * avg_cat + 0.30 * avg_pri + 0.20 * avg_esc + 0.10 * avg_resp
    return max(0.0, min(1.0, score))


# ──────────────────────────────────────────────────────────────────────────
# Task 3: grade_triage_adversarial
# Weights: category 35%, priority 25%, escalation 25%, response 15%
# Hardest task — response quality and escalation carry more weight
# Bonus: correctly identifying all adversarial categories
# ──────────────────────────────────────────────────────────────────────────

def grade_triage_adversarial(trajectory: List[Dict[str, Any]]) -> float:
    """
    Grade the adversarial email triage task.

    Scoring:
      - Category accuracy: 35% of score
      - Priority accuracy: 25% of score (with partial credit)
      - Escalation accuracy: 25% of score (false escalations penalized heavily)
      - Response quality:   15% of score

    Adversarial bonus: +0.05 if ≥80% of categories correct (rewards consistency).

    Returns a float in [0.0, 1.0].
    """
    if not trajectory:
        return 0.0

    cat_scores = []
    pri_scores = []
    esc_scores = []
    resp_scores = []

    for step in trajectory:
        cat_scores.append(_category_score(step))
        pri_scores.append(_priority_score(step))
        raw_esc = _escalation_score(step)
        esc_scores.append(raw_esc)
        resp_scores.append(step.get("response_quality", 0.0))

    avg_cat = sum(cat_scores) / len(cat_scores)
    avg_pri = sum(pri_scores) / len(pri_scores)
    # Heavily normalize escalation — false escalations are very bad here
    avg_esc = min(1.0, max(0.0, sum(esc_scores) / len(esc_scores) + 0.5))
    avg_resp = sum(resp_scores) / len(resp_scores)

    base_score = 0.35 * avg_cat + 0.25 * avg_pri + 0.25 * avg_esc + 0.15 * avg_resp

    # Adversarial consistency bonus
    bonus = 0.05 if avg_cat >= 0.80 else 0.0

    score = base_score + bonus
    return max(0.0, min(1.0, score))
