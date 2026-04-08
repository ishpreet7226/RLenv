---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - nlp
  - email-triage
license: mit
---

# 📧 Customer Support Email Triage — `email_triage_v1`

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-6366f1?style=flat-square)](https://openenv.dev)
[![HuggingFace Space](https://img.shields.io/badge/HF%20Space-Live-orange?style=flat-square)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

## Overview

**`email_triage_v1`** is a real-world OpenEnv environment where an AI agent learns to triage customer support emails — a task performed millions of times daily in enterprise support centers worldwide.

The agent processes a queue of 10 customer emails per episode and must:
- **Classify** each email (billing / technical / account / general / spam)
- **Prioritize** urgency (urgent / high / medium / low)
- **Decide** whether to escalate to a human agent
- **Draft** a professional response (rewarded at medium/hard difficulty)

This environment fills a direct gap in the RL/agent evaluation ecosystem: no existing OpenEnv environment benchmarks the core enterprise workflow of email classification and triage, a task where incorrect agent decisions have measurable downstream business impact.

---

## Environment Design

### State Management
Each episode is fully isolated. `reset(task_id)` loads a deterministic queue of 10 emails for the specified task. The internal state tracks queue position, cumulative reward, and per-step trajectory.

### Action Space

| Field | Type | Values |
|-------|------|--------|
| `category` | `string` | `billing`, `technical`, `account`, `general`, `spam` |
| `priority` | `string` | `urgent`, `high`, `medium`, `low` |
| `response_draft` | `string \| null` | Any text reply to the sender |
| `escalate` | `boolean` | `true` = flag for human agent |

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `email_id` | `string` | Unique email identifier |
| `subject` | `string` | Email subject line |
| `body` | `string` | Full email body |
| `sender_tier` | `string` | `premium`, `standard`, or `trial` |
| `previous_tickets` | `int` | Past support tickets from sender |
| `sentiment_score` | `float` | Tone score in `[-1.0, 1.0]` |
| `queue_position` | `int` | Current position in episode queue |
| `emails_remaining` | `int` | Emails left to process |
| `task_context` | `string` | Natural-language task description |
| `valid_categories` | `list[str]` | The 5 valid category labels |
| `valid_priorities` | `list[str]` | The 4 valid priority labels |

### Reward Function

Each step produces a **structured reward** (not sparse):

| Component | Weight | Description |
|-----------|--------|-------------|
| Category accuracy | **40%** | Exact match with ground truth |
| Priority accuracy | **30%** | Exact match (partial: 0.5 if 1 level off) |
| Response quality | **15%** | Keyword coverage in drafted reply |
| Escalation accuracy | **10%** | Correct escalation (+) / false escalation (−) |
| Spam classification | **5%** | Bonus for correct spam ID; heavy penalty for false spam |

All step rewards are clamped to `[0.0, 1.0]`.

### Termination
Episodes end deterministically after all 10 emails in the queue are processed (`done=True`). No early termination.

---

## Tasks

### Task 1 — Triage Basic `[easy]`
- **10 emails** with clear, unambiguous intent
- Emails include: obvious billing disputes, login failures, app crashes, feature questions, clear spam
- **Scored by**: category (50%) + priority (40%) + escalation (10%)
- **Success threshold**: 0.75

### Task 2 — Triage Ambiguous `[medium]`
- **10 emails** where multiple issues coexist or signals are indirect
- Challenges include: multi-issue emails (identify PRIMARY), compliance questions, long-running unresolved disputes, API integration breaks
- **Scored by**: category (40%) + priority (30%) + escalation (20%) + response (10%)
- **Success threshold**: 0.70

### Task 3 — Triage Adversarial `[hard]`
- **10 emails** crafted to mislead the agent:
  - Trial users with manufactured urgency ("DISASTER!!!")
  - Vendor spam disguised as legitimate invoices
  - Hidden technical bugs buried in glowing praise
  - Security vulnerability disclosures requiring immediate escalation
  - GDPR Article 17 right-to-erasure requests
- **Scored by**: category (35%) + priority (25%) + escalation (25%) + response (15%)
- **Success threshold**: 0.65

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Environment info + health |
| `GET` | `/health` | Lightweight liveness probe |
| `GET` | `/tasks` | List all 3 tasks with metadata |
| `POST` | `/reset` | Start new episode `{"task_id": "triage_basic"}` |
| `POST` | `/step` | Submit action `{"category":..., "priority":..., "escalate":...}` |
| `GET` | `/state` | Current episode metadata |

### Example Usage

```bash
# Reset for Task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "triage_basic"}'

# Submit a triage action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "category": "billing",
    "priority": "medium",
    "response_draft": "Thank you for reaching out. We will review the overcharge and issue a credit within 3-5 business days.",
    "escalate": false
  }'

# Get current state
curl http://localhost:7860/state
```

---

## Setup & Installation

### Local (Python)

```bash
git clone <your-repo-url>
cd RLenv

pip install -r requirements.txt

# Start the API server
uvicorn server:app --host 0.0.0.0 --port 7860

# OR run the inference baseline directly
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
python inference.py
```

### Docker

```bash
# Build
docker build -t email-triage-env .

# Run server
docker run -p 7860:7860 email-triage-env

# Run inference baseline
docker run \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="sk-..." \
  email-triage-env \
  python inference.py
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API base URL | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier | `gpt-4o-mini` |
| `HF_TOKEN` | HuggingFace / OpenAI API key | *(required for inference)* |

---

## Inference Script

The baseline `inference.py` runs an LLM agent against all 3 tasks and emits structured logs:

```
[START] task=triage_basic env=email_triage_v1 model=gpt-4o-mini
[STEP] step=1 action={"category": "account", "priority": "high", "escalate": false} reward=0.9000 done=false error=null
[STEP] step=2 action={"category": "billing", "priority": "medium", "escalate": false} reward=0.8500 done=false error=null
...
[END] success=true steps=10 score=0.8234 rewards=0.9000,0.8500,...
```

---

## Baseline Scores

Scores from running `gpt-4o-mini` at `temperature=0` (deterministic):

| Task | Score | Success |
|------|-------|---------|
| `triage_basic` (easy) | ~0.82 | ✅ |
| `triage_ambiguous` (medium) | ~0.71 | ✅ |
| `triage_adversarial` (hard) | ~0.58 | ❌ (near threshold) |

A naive classifier (always predicts `billing/high`) scores approximately **0.20–0.35**, demonstrating the environment requires genuine language understanding.

---

## Project Structure

```
RLenv/
├── env/
│   ├── __init__.py
│   ├── environment.py      # EmailTriageEnv (reset / step / state)
│   ├── actions.py          # Action Pydantic model
│   ├── observations.py     # Observation Pydantic model
│   ├── rewards.py          # Reward Pydantic model
│   ├── graders.py          # Deterministic graders (x3)
│   └── data/
│       ├── __init__.py
│       └── emails.py       # 30-email hardcoded dataset (10 per task)
├── tasks/
│   ├── triage_basic.py     # Task 1 config (easy)
│   ├── triage_ambiguous.py # Task 2 config (medium)
│   └── triage_adversarial.py # Task 3 config (hard)
├── server.py               # FastAPI server
├── inference.py            # Baseline LLM inference script
├── openenv.yaml            # OpenEnv spec manifest
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Novelty & Motivation

Unlike classical RL benchmarks (CartPole, LunarLander, Atari), this environment:

1. **Models a genuine enterprise workflow** — email triage is a multi-billion-dollar operational problem with direct ROI implications
2. **Tests language understanding under adversarial conditions** — not just retrieval or generation, but nuanced classification with misleading signals
3. **Provides dense, structured rewards** — 5 reward components give fine-grained learning signal beyond sparse binary success
4. **Includes difficulty-tiered adversarial scenarios** — most NLP benchmarks lack adversarial email patterns (manufactured urgency, vendor spam as invoices, buried bugs)
5. **Maps directly to LLM agent evaluation** — the agent must combine classification, prioritization, and generation skills in a single decision

This environment is immediately useful for evaluating and training customer support automation agents, LLM routing systems, and enterprise email AI products.
