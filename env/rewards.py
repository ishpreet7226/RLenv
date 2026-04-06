from pydantic import BaseModel

class Reward(BaseModel):
    altitude_accuracy_reward: float
    stability_reward: float
    energy_efficiency_reward: float
    disturbance_recovery_reward: float
    oscillation_penalty: float
    shutdown_penalty: float
    task_completion_bonus: float
    total_reward: float
    step_reward: float
    cumulative_reward: float
