DISTURBANCE_MAGNITUDE = 0.15
STABILITY_THRESHOLD = 0.8

TASK_CONFIG = {
    "name": "disturbance",
    "difficulty": "medium",
    "max_steps": 100,
    "success_threshold": 0.8
}

class DisturbanceRecoveryTask:
    initial_conditions = {
        "altitude": 10.0, 
        "velocity": 0.0, 
        "stability": 1.0, 
        "energy": 100.0, 
        "disturbance_spike": DISTURBANCE_MAGNITUDE
    }
    task_objective = f"Restore field_stability_score > {STABILITY_THRESHOLD} within 10 steps after a disturbance spike."
    difficulty_parameters = {
        "max_steps": 10, 
        "target_stability": STABILITY_THRESHOLD, 
        "target_altitude": 10.0
    }

    @staticmethod
    def evaluate(env, steps_taken: int) -> float:
        stability_restored = max(0.0, min(1.0, env.field_stability_score))
        recovery_speed = max(0.0, 1.0 - (steps_taken / DisturbanceRecoveryTask.difficulty_parameters["max_steps"]))
        altitude_accuracy = max(0.0, 1.0 - abs(env.current_altitude - DisturbanceRecoveryTask.difficulty_parameters["target_altitude"]) / 10.0)

        score = recovery_speed * 0.4 + stability_restored * 0.4 + altitude_accuracy * 0.2
        return max(0.0, min(1.0, score))
