"""
FastAPI server for the Customer Support Email Triage OpenEnv environment.

Endpoints (OpenEnv spec-compliant):
  GET  /            — environment info
  GET  /health      — liveness probe → {"status": "healthy"}
  GET  /metadata    — name + description (required by openenv validate)
  GET  /schema      — action / observation / state schemas
  GET  /tasks       — list all 3 tasks
  POST /reset       — initialize episode (body: {"task_id": "triage_basic"})
  POST /step        — submit action (body: Action JSON)
  GET  /state       — current episode state
"""
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import traceback

from env.environment import EmailTriageEnv
from env.actions import Action
from env.observations import Observation
from env.rewards import Reward

app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "Customer Support Email Triage — an OpenEnv-compliant environment "
        "where an AI agent learns to classify, prioritize, and respond to "
        "customer support emails."
    ),
    version="1.0.0",
)

# Shared environment instance (stateful per server process)
env = EmailTriageEnv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = "triage_basic"


# ── Core OpenEnv Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe — returns 'healthy' as required by openenv validate."""
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    """Return environment name and description (required by openenv validate)."""
    return {
        "name": "email_triage_v1",
        "description": (
            "Customer Support Email Triage — an OpenEnv environment where an AI "
            "agent learns to classify, prioritize, escalate, and reply to customer "
            "support emails. Simulates a real task performed millions of times daily "
            "in enterprise support centers."
        ),
        "version": "1.0.0",
        "tasks": ["triage_basic", "triage_ambiguous", "triage_adversarial"],
    }


@app.get("/schema")
def schema():
    """
    Return JSON Schema definitions for action, observation, and state.
    Required by openenv validate runtime check.
    """
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {
            "type": "object",
            "properties": {
                "task_id":             {"type": "string"},
                "step":                {"type": "integer"},
                "queue_index":         {"type": "integer"},
                "total_emails":        {"type": "integer"},
                "emails_remaining":    {"type": "integer"},
                "cumulative_reward":   {"type": "number"},
                "done":                {"type": "boolean"},
                "trajectory_length":   {"type": "integer"},
            },
        },
    }


# ── Episode Endpoints ──────────────────────────────────────────────────────

@app.get("/")
def root():
    """Environment info."""
    return {
        "name": "email_triage_v1",
        "version": "1.0.0",
        "description": "Customer Support Email Triage OpenEnv Environment",
        "tasks": ["triage_basic", "triage_ambiguous", "triage_adversarial"],
        "endpoints": {
            "reset":    "POST /reset",
            "step":     "POST /step",
            "state":    "GET  /state",
            "metadata": "GET  /metadata",
            "schema":   "GET  /schema",
        },
        "status": "ok",
    }


@app.get("/tasks")
def list_tasks():
    """Return metadata for all 3 tasks."""
    return {
        "tasks": [
            {
                "task_id":           "triage_basic",
                "difficulty":        "easy",
                "max_steps":         10,
                "success_threshold": 0.75,
                "description": (
                    "Triage 10 clear-cut emails. Category and priority signals are "
                    "unambiguous."
                ),
            },
            {
                "task_id":           "triage_ambiguous",
                "difficulty":        "medium",
                "max_steps":         10,
                "success_threshold": 0.70,
                "description": (
                    "Triage 10 ambiguous emails. Identify the primary issue when "
                    "multiple topics are present. Escalation and response quality matter."
                ),
            },
            {
                "task_id":           "triage_adversarial",
                "difficulty":        "hard",
                "max_steps":         10,
                "success_threshold": 0.65,
                "description": (
                    "Triage 10 adversarial emails: inflated urgency, vendor spam, "
                    "hidden bugs, legal notices, and GDPR requests."
                ),
            },
        ]
    }


@app.post("/reset")
def reset_env(request: Optional[ResetRequest] = None):
    """Initialize a new episode for the given task."""
    try:
        task_id = request.task_id if request is not None else "triage_basic"
        obs = env.reset(task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/step")
def step_env(action: Action):
    """Submit the agent's triage action for the current email."""
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward":      reward.model_dump(),
            "done":        done,
            "info":        info,
        }
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/state")
def get_state():
    """Return current episode state metadata."""
    return env.state()


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    """Server entry point — used by `openenv serve` and `[project.scripts]`."""
    uvicorn.run(app, host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
