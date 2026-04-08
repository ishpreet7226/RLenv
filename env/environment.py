"""
Customer Support Email Triage Environment — Core Implementation

Implements the OpenEnv interface:
  reset(task_id)  → Observation
  step(action)    → (Observation, Reward, done, info)
  state()         → Dict

The agent processes a queue of 10 support emails per episode and must
correctly triage each one (category + priority) while optionally drafting
a response and deciding whether to escalate.
"""
import re
from typing import Optional, Dict, Any, Tuple, Union

from env.data.emails import get_emails_for_task, VALID_TASK_IDS
from env.observations import Observation
from env.actions import Action
from env.rewards import Reward

# ── Priority ordering for partial credit ──────────────────────────────────
PRIORITY_ORDER = {"urgent": 0, "high": 1, "medium": 2, "low": 3}

TASK_CONTEXTS = {
    "triage_basic": (
        "You are triaging customer support emails. Each email has a clear "
        "category (billing/technical/account/general/spam) and priority "
        "(urgent/high/medium/low). Choose accurately and escalate ONLY when "
        "the customer explicitly needs urgent human intervention."
    ),
    "triage_ambiguous": (
        "You are triaging support emails that may touch multiple topics. "
        "Identify the PRIMARY issue driving the email and assign the most "
        "appropriate category and priority. Consider sender tier and past "
        "ticket history. Some cases warrant escalation."
    ),
    "triage_adversarial": (
        "You are triaging difficult support emails. Watch for: emotional "
        "language that inflates urgency, vendor spam disguised as invoices, "
        "hidden technical issues buried in positive messages, legal/security "
        "notices requiring escalation, and GDPR/compliance requests. "
        "Draft a concise, professional reply for complex cases."
    ),
}


class EmailTriageEnv:
    """
    Customer Support Email Triage — OpenEnv-compliant environment.

    Episode flow:
    1. reset(task_id) → loads 10 emails for the task, returns first Observation
    2. step(action)   → grades action, advances queue, returns next Observation
    3. Episode ends when all 10 emails are processed (done=True)
    """

    def __init__(self):
        self.task_id: str = "triage_basic"
        self._emails: list = []
        self._queue_index: int = 0
        self._cumulative_reward: float = 0.0
        self._step_count: int = 0
        self._done: bool = False
        self._trajectory: list = []  # stores per-step info for graders

    # ── OpenEnv required methods ───────────────────────────────────────────

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """Initialize a new episode for the given task."""
        self.task_id = task_id or "triage_basic"
        if self.task_id not in VALID_TASK_IDS:
            raise ValueError(f"Unknown task '{self.task_id}'. Valid: {VALID_TASK_IDS}")

        self._emails = get_emails_for_task(self.task_id)
        self._queue_index = 0
        self._cumulative_reward = 0.0
        self._step_count = 0
        self._done = False
        self._trajectory = []

        return self._build_observation()

    def step(
        self,
        action: Union[Dict[str, Any], Action],
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process the agent's triage decision for the current email.

        Returns
        -------
        observation : Observation  — next email (or terminal obs)
        reward      : Reward       — detailed reward breakdown
        done        : bool         — True when all emails are processed
        info        : dict         — episode metadata
        """
        if self._done:
            # Return terminal observation if already done
            obs = self._build_observation(terminal=True)
            rew = Reward(email_id="terminal", cumulative_reward=self._cumulative_reward)
            return obs, rew, True, self.state()

        if isinstance(action, dict):
            action = Action(**action)

        current_email = self._emails[self._queue_index]
        ground_truth = current_email["ground_truth"]

        # ── Compute reward components ──────────────────────────────────
        cat_acc = self._score_category(action.category, ground_truth["category"])
        pri_acc = self._score_priority(action.priority, ground_truth["priority"])
        resp_q = self._score_response(action.response_draft, ground_truth["response_keywords"])
        esc_bonus, false_esc_penalty = self._score_escalation(
            action.escalate, ground_truth["should_escalate"]
        )
        spam_score = self._score_spam(
            action.category, ground_truth["category"], action.escalate
        )

        # Weighted step reward (clamped to [0, 1])
        raw_reward = (
            0.40 * cat_acc
            + 0.30 * pri_acc
            + 0.15 * resp_q
            + 0.10 * (esc_bonus + false_esc_penalty)
            + 0.05 * spam_score
        )
        step_reward = max(0.0, min(1.0, raw_reward))
        self._cumulative_reward += step_reward

        reward = Reward(
            category_accuracy=cat_acc,
            priority_accuracy=pri_acc,
            response_quality=resp_q,
            escalation_bonus=esc_bonus,
            false_escalation_penalty=false_esc_penalty,
            spam_correct=spam_score,
            step_reward=step_reward,
            cumulative_reward=self._cumulative_reward,
            email_id=current_email["email_id"],
        )

        # Record for graders
        self._trajectory.append(
            {
                "email_id": current_email["email_id"],
                "ground_truth_category": ground_truth["category"],
                "ground_truth_priority": ground_truth["priority"],
                "pred_category": action.category,
                "pred_priority": action.priority,
                "should_escalate": ground_truth["should_escalate"],
                "escalated": action.escalate,
                "response_quality": resp_q,
                "step_reward": step_reward,
            }
        )

        self._queue_index += 1
        self._step_count += 1

        if self._queue_index >= len(self._emails):
            self._done = True

        obs = self._build_observation(terminal=self._done)
        info = self.state()

        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return current episode state metadata."""
        return {
            "task_id": self.task_id,
            "step": self._step_count,
            "queue_index": self._queue_index,
            "total_emails": len(self._emails),
            "emails_remaining": max(0, len(self._emails) - self._queue_index),
            "cumulative_reward": round(self._cumulative_reward, 4),
            "done": self._done,
            "trajectory_length": len(self._trajectory),
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_observation(self, terminal: bool = False) -> Observation:
        """Build an Observation from the current email in queue."""
        if terminal or self._queue_index >= len(self._emails):
            # Return a terminal/empty observation
            return Observation(
                email_id="__terminal__",
                subject="[Episode Complete]",
                body="All emails in this task have been processed.",
                sender_tier="none",
                previous_tickets=0,
                sentiment_score=0.0,
                queue_position=len(self._emails) + 1,
                emails_remaining=0,
                task_context=TASK_CONTEXTS.get(self.task_id, ""),
            )

        email = self._emails[self._queue_index]
        return Observation(
            email_id=email["email_id"],
            subject=email["subject"],
            body=email["body"],
            sender_tier=email["sender_tier"],
            previous_tickets=email["previous_tickets"],
            sentiment_score=email["sentiment_score"],
            queue_position=self._queue_index + 1,
            emails_remaining=len(self._emails) - self._queue_index,
            task_context=TASK_CONTEXTS.get(self.task_id, ""),
        )

    @staticmethod
    def _score_category(predicted: str, ground_truth: str) -> float:
        """1.0 for exact match, 0.0 otherwise."""
        return 1.0 if predicted == ground_truth else 0.0

    @staticmethod
    def _score_priority(predicted: str, ground_truth: str) -> float:
        """
        Partial credit for priority:
          exact match      → 1.0
          one level off    → 0.5
          two levels off   → 0.2
          three+ levels    → 0.0
        """
        pred_idx = PRIORITY_ORDER.get(predicted, 2)
        gt_idx = PRIORITY_ORDER.get(ground_truth, 2)
        diff = abs(pred_idx - gt_idx)
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.5
        elif diff == 2:
            return 0.2
        return 0.0

    @staticmethod
    def _score_response(draft: Optional[str], keywords: list) -> float:
        """
        Score response quality by keyword coverage.
        Returns 0.0 if no draft provided or no keywords required (spam).
        Returns fraction of keywords found in the draft (case-insensitive).
        """
        if not draft or not keywords:
            return 0.0
        draft_lower = draft.lower()
        hits = sum(
            1 for kw in keywords
            if re.search(re.escape(kw.lower()), draft_lower)
        )
        return hits / len(keywords)

    @staticmethod
    def _score_escalation(escalated: bool, should_escalate: bool) -> Tuple[float, float]:
        """
        Returns (escalation_bonus, false_escalation_penalty).
        Correct escalation  → bonus  +1.0 (weighted to 0.10 in step reward)
        False escalation    → penalty -1.0 (weighted to -0.10 in step reward)
        No escalation req   → 0.0, 0.0
        """
        if should_escalate and escalated:
            return 1.0, 0.0
        elif not should_escalate and escalated:
            return 0.0, -1.0
        return 0.0, 0.0

    @staticmethod
    def _score_spam(predicted_category: str, ground_truth_category: str, escalated: bool) -> float:
        """
        Extra spam classification reward/penalty.
          Correctly tagged spam   → +1.0
          Non-spam tagged as spam → -3.0 (strong deterrent)
          Anything else           → 0.0
        """
        if ground_truth_category == "spam" and predicted_category == "spam":
            return 1.0
        elif ground_truth_category != "spam" and predicted_category == "spam":
            return -3.0
        return 0.0

    def get_trajectory(self) -> list:
        """Return the full trajectory for grader access."""
        return self._trajectory
