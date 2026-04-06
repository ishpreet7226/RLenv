ALTITUDE_TOLERANCE = 0.5

TASK_CONFIG = {
    "name": "hover",
    "difficulty": "easy",
    "max_steps": 100,
    "success_threshold": 0.8
}

class HoverTask:
    initial_conditions = {
        "altitude": 10.0, 
        "velocity": 0.0, 
        "stability": 1.0, 
        "energy": 100.0
    }
    task_objective = f"Maintain altitude within ±{ALTITUDE_TOLERANCE} of target_altitude, minimize oscillation, and minimize energy consumption"
    difficulty_parameters = {
        "target_altitude": 10.0, 
        "tolerance": ALTITUDE_TOLERANCE
    }

    @staticmethod
    def evaluate(env) -> float:
        alt_diff = abs(env.current_altitude - HoverTask.difficulty_parameters["target_altitude"])
        altitude_accuracy = 1.0 if alt_diff <= HoverTask.difficulty_parameters["tolerance"] else max(0.0, 1.0 - (alt_diff - ALTITUDE_TOLERANCE) / 5.0)
        stability_score = max(0.0, env.field_stability_score)
        energy_efficiency = max(0.0, env.energy_remaining / 100.0)
        
        score = altitude_accuracy * 0.5 + stability_score * 0.3 + energy_efficiency * 0.2
        return max(0.0, min(1.0, score))
