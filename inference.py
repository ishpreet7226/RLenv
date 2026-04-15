"""
inference.py — Baseline inference script for the Customer Support Email Triage
OpenEnv environment.

Runs an LLM agent (via OpenAI-compatible API) against all 3 tasks and
emits structured stdout logs in the required [START] / [STEP] / [END] format.

Required environment variables:
  API_BASE_URL   — LLM API base URL (default: https://api.openai.com/v1)
  MODEL_NAME     — Model identifier  (default: gpt-4o-mini)
  HF_TOKEN       — API key / HuggingFace token

Usage:
  python inference.py
"""

import os
import sys
import json
import traceback
from typing import Optional

from openai import OpenAI

from env.environment import EmailTriageEnv
from env.actions import Action
from env.graders import grade_triage_basic, grade_triage_ambiguous, grade_triage_adversarial

# ── Configuration ──────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

# Task registry
TASKS = [
    {
        "task_id":           "triage_basic",
        "grader":            grade_triage_basic,
        "success_threshold": 0.75,
    },
    {
        "task_id":           "triage_ambiguous",
        "grader":            grade_triage_ambiguous,
        "success_threshold": 0.70,
    },
    {
        "task_id":           "triage_adversarial",
        "grader":            grade_triage_adversarial,
        "success_threshold": 0.65,
    },
]

ENV_NAME = "email_triage_v1"

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert customer support triage agent.

For each email you receive, respond with a valid JSON object containing:
  - "category": one of ["billing", "technical", "account", "general", "spam"]
  - "priority":  one of ["urgent", "high", "medium", "low"]
  - "response_draft": a brief, professional reply to the customer (2-4 sentences)
  - "escalate": true if this needs human escalation, false otherwise

Triage guidelines:
- billing:   payment issues, invoices, refunds, subscription changes, pricing
- technical: bugs, crashes, API errors, integrations, performance issues
- account:   login, password, account access, security, data deletion
- general:   how-to questions, feature requests, compliance info
- spam:      unsolicited commercial messages, phishing, vendor invoices you didn't request

Priority:
- urgent: production down, data loss, security breach, legal deadlines, double-charges on premium
- high:   significant disruption, premium user issues, repeated unresolved problems
- medium: moderate impact, standard issues with a clear path
- low:    questions, minor bugs, informational requests

Escalate when: security vulnerabilities, legal notices (GDPR, HIPAA), data breaches,
               persistent unresolved issues (5+ previous tickets), post-cancellation charges.

Do NOT escalate: trial users with inflated urgency, routine billing changes, general how-to.

Respond ONLY with the JSON object. No markdown, no explanation."""

# ── OpenAI client ──────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def build_user_prompt(obs) -> str:
    return f"""Email #{obs.queue_position} of {obs.queue_position + obs.emails_remaining - 1}

Task context: {obs.task_context}

--- EMAIL ---
Subject: {obs.subject}
From tier: {obs.sender_tier}
Previous tickets: {obs.previous_tickets}
Sentiment score: {obs.sentiment_score:.2f}  (-1=very negative, +1=very positive)

{obs.body}
--- END ---

Valid categories: {obs.valid_categories}
Valid priorities: {obs.valid_priorities}

Respond with JSON only."""


def call_llm(client: OpenAI, user_prompt: str) -> Optional[Action]:
    """Call LLM and parse response into an Action. Returns None on failure."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        data = json.loads(raw)
        return Action(
            category=data.get("category", "general"),
            priority=data.get("priority", "medium"),
            response_draft=data.get("response_draft", None),
            escalate=bool(data.get("escalate", False)),
        )
    except Exception:
        return None


# ── Task runner ────────────────────────────────────────────────────────────

def run_task(task_id: str, grader, success_threshold: float) -> None:
    """Run a single task to completion and print structured logs."""
    client = get_client()
    env = EmailTriageEnv()

    obs = env.reset(task_id)

    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")
    sys.stdout.flush()

    step_num    = 0
    rewards     = []
    done        = False

    while not done:
        step_num += 1
        user_prompt = build_user_prompt(obs)
        action = call_llm(client, user_prompt)

        error_str = "null"
        if action is None:
            # Fallback action on parse failure
            error_str = "parse_error"
            action = Action(category="general", priority="medium", escalate=False)

        try:
            obs, reward, done, info = env.step(action)
            step_reward = round(reward.step_reward, 2)
            rewards.append(step_reward)

            action_summary = json.dumps({
                "category": action.category,
                "priority": action.priority,
                "escalate": action.escalate,
            })

            print(
                f"[STEP] step={step_num} "
                f"action={action_summary} "
                f"reward={step_reward:.2f} "
                f"done={str(done).lower()} "
                f"error={error_str}"
            )
            sys.stdout.flush()

        except Exception as e:
            error_str = type(e).__name__
            print(
                f"[STEP] step={step_num} "
                f"action=null "
                f"reward=0.0000 "
                f"done=false "
                f"error={error_str}"
            )
            sys.stdout.flush()
            break

    trajectory = env.get_trajectory()
    score      = grader(trajectory)
    success    = score >= success_threshold
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"

    print(
        f"[END] success={str(success).lower()} "
        f"steps={step_num} "
        f"score={score:.2f} "
        f"rewards={rewards_str}"
    )
    sys.stdout.flush()
    print()  # blank line between tasks


# ── Entry point ────────────────────────────────────────────────────────────

def run_all_tasks() -> None:
    for task in TASKS:
        run_task(
            task_id=task["task_id"],
            grader=task["grader"],
            success_threshold=task["success_threshold"],
        )


if __name__ == "__main__":
    run_all_tasks()
