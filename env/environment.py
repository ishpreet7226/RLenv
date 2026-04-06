import numpy as np
from typing import Optional, Dict, Any, Union
from env.observations import Observation
from env.actions import Action, ActionType
from env.rewards import Reward

class AntiGravityControlEnv:
    def __init__(self):
        self.task_id = "hover"
        self.current_altitude = 10.0
        self.vertical_velocity = 0.0
        self.field_stability_score = 1.0
        self.energy_remaining = 100.0
        self.oscillation_level = 0.0
        self.external_disturbance = 0.0
        self.target_altitude = 10.0
        self.steps_remaining = 100
        self.history = []
        self.reward_so_far = 0.0
        self.done = False

    def reset(self, task_id: Optional[str] = None) -> Observation:
        self.task_id = task_id or "hover"
        self.current_altitude = 10.0
        self.vertical_velocity = 0.0
        self.field_stability_score = 1.0
        self.energy_remaining = 100.0
        self.oscillation_level = 0.0
        self.external_disturbance = 0.0
        self.target_altitude = 10.0
        self.steps_remaining = 100
        self.history = []
        self.reward_so_far = 0.0
        self.done = False
        
        return self._get_obs()

    def _get_obs(self) -> Observation:
        return Observation(
            current_altitude=self.current_altitude,
            vertical_velocity=self.vertical_velocity,
            field_stability_score=self.field_stability_score,
            energy_remaining=self.energy_remaining,
            oscillation_level=self.oscillation_level,
            external_disturbance=self.external_disturbance,
            target_altitude=self.target_altitude,
            steps_remaining=self.steps_remaining,
            history=self.history
        )

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "altitude": self.current_altitude,
            "velocity": self.vertical_velocity,
            "stability": self.field_stability_score,
            "energy_remaining": self.energy_remaining,
            "steps_remaining": self.steps_remaining,
            "reward_so_far": self.reward_so_far,
            "done": self.done
        }

    def step(self, action: Union[Dict[str, Any], Action]) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        if isinstance(action, dict):
            action = Action(**action)

        current_delta = 0.0
        # Basic physics core for Phase 4
        if action.action_type == ActionType.INCREASE_LIFT:
            current_delta = action.delta if action.delta is not None else 1.0
            self.vertical_velocity += current_delta
        elif action.action_type == ActionType.DECREASE_LIFT:
            current_delta = action.delta if action.delta is not None else 1.0
            self.vertical_velocity -= current_delta
        elif action.action_type == ActionType.LOCK_ALTITUDE:
            if action.target_altitude is not None:
                self.target_altitude = action.target_altitude

        # 1. energy consumption proportional to lift adjustments
        energy_consumption = abs(current_delta) * 1.0
        self.energy_remaining -= energy_consumption

        # 4. disturbance influence on velocity (stochastic)
        self.external_disturbance = np.random.normal(0, 0.02)
        self.vertical_velocity += self.external_disturbance

        # 2. oscillation accumulation based on velocity
        self.oscillation_level += abs(self.vertical_velocity) * 0.05

        # 3. stability degradation based on oscillation
        self.field_stability_score -= self.oscillation_level * 0.01

        # Update core state
        self.current_altitude += self.vertical_velocity
        self.steps_remaining -= 1
        
        # 5. termination when energy <= 0
        # 6. termination when stability < 0.2
        if self.steps_remaining <= 0 or self.energy_remaining <= 0 or self.field_stability_score < 0.2:
            self.done = True

        # Base step reward and soft instability penalty
        step_reward = 0.1
        stability_penalty = (1.0 - self.field_stability_score) * 0.1
        step_reward -= stability_penalty
        self.reward_so_far += step_reward

        rew = Reward(
            altitude_accuracy_reward=0.0,
            stability_reward=0.0,
            energy_efficiency_reward=0.0,
            disturbance_recovery_reward=0.0,
            oscillation_penalty=0.0,
            shutdown_penalty=0.0,
            task_completion_bonus=0.0,
            total_reward=step_reward,
            step_reward=step_reward,
            cumulative_reward=self.reward_so_far
        )

        obs = self._get_obs()
        info = self.state()

        return obs, rew, self.done, info
