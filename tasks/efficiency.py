PROFILE_DURATION_PER_STEP = 20

TASK_CONFIG = {
    "name": "efficiency",
    "difficulty": "hard",
    "max_steps": 100,
    "success_threshold": 0.8
}

class EfficiencyOptimizationTask:
    initial_conditions = {
        "altitude": 10.0, 
        "velocity": 0.0, 
        "stability": 1.0, 
        "energy": 100.0
    }
    task_objective = "Follow changing altitude profile, minimize energy usage, avoid instability collapse, handle disturbance noise."
    difficulty_parameters = {
        "target_profile": [10.0, 15.0, 12.0, 8.0, 10.0], 
        "profile_duration_per_step": PROFILE_DURATION_PER_STEP
    }

    @staticmethod
    def evaluate(env, current_step: int) -> float:
        profile = EfficiencyOptimizationTask.difficulty_parameters["target_profile"]
        duration = EfficiencyOptimizationTask.difficulty_parameters["profile_duration_per_step"]
        idx = min(current_step // duration, len(profile) - 1)
        target = profile[idx]

        trajectory_accuracy = max(0.0, 1.0 - abs(env.current_altitude - target) / 10.0)
        energy_efficiency = max(0.0, env.energy_remaining / 100.0)
        stability_maintenance = max(0.0, env.field_stability_score)

        score = trajectory_accuracy * 0.4 + energy_efficiency * 0.4 + stability_maintenance * 0.2
        return max(0.0, min(1.0, score))
