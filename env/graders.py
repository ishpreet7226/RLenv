import numpy as np

def clamp(value):
    return max(0.0, min(1.0, float(value)))

def grade_hover(trajectory):
    if not trajectory:
        return 0.0

    target_altitude = 10.0
    tolerance = 0.5

    alt_accuracies = []
    stability_scores = []
    
    for step in trajectory:
        alt = step.get('altitude', target_altitude) if isinstance(step, dict) else getattr(step, 'current_altitude', target_altitude)
        stab = step.get('stability', 0.0) if isinstance(step, dict) else getattr(step, 'field_stability_score', 0.0)
        
        alt_diff = abs(alt - target_altitude)
        if alt_diff <= tolerance:
            acc = 1.0
        else:
            acc = max(0.0, 1.0 - (alt_diff - tolerance) / 5.0)
        
        alt_accuracies.append(acc)
        stability_scores.append(max(0.0, stab))

    final_step = trajectory[-1]
    energy = final_step.get('energy_remaining', 0.0) if isinstance(final_step, dict) else getattr(final_step, 'energy_remaining', 0.0)

    avg_alt_acc = sum(alt_accuracies) / len(alt_accuracies) if alt_accuracies else 0.0
    avg_stab = sum(stability_scores) / len(stability_scores) if stability_scores else 0.0
    energy_eff = max(0.0, energy / 100.0)

    score = 0.5 * avg_alt_acc + 0.3 * avg_stab + 0.2 * energy_eff
    score = max(0.0, min(1.0, score))
    return score

def grade_disturbance(trajectory):
    if not trajectory:
        return 0.0

    target_altitude = 10.0
    target_stability = 0.8
    
    recovery_step = len(trajectory)
    
    for i, step in enumerate(trajectory):
        stab = step.get('stability', 0.0) if isinstance(step, dict) else getattr(step, 'field_stability_score', 0.0)
        if stab >= target_stability:
            recovery_step = i
            break
            
    recovery_speed = max(0.0, 1.0 - (recovery_step / max(1, len(trajectory))))
    
    final_step = trajectory[-1]
    final_stab = final_step.get('stability', 0.0) if isinstance(final_step, dict) else getattr(final_step, 'field_stability_score', 0.0)
    final_alt = final_step.get('altitude', target_altitude) if isinstance(final_step, dict) else getattr(final_step, 'current_altitude', target_altitude)
    
    final_stability_score = max(0.0, final_stab)
    altitude_accuracy = max(0.0, 1.0 - abs(final_alt - target_altitude) / 10.0)
    
    score = 0.5 * recovery_speed + 0.3 * final_stability_score + 0.2 * altitude_accuracy
    score = max(0.0, min(1.0, score))
    return score

def grade_efficiency(trajectory):
    if not trajectory:
        return 0.0

    target_profile = [10.0, 15.0, 12.0, 8.0, 10.0]
    duration = 20
    
    traj_accuracies = []
    stability_scores = []
    
    for i, step in enumerate(trajectory):
        idx = min(i // duration, len(target_profile) - 1)
        target = target_profile[idx]
        
        alt = step.get('altitude', 10.0) if isinstance(step, dict) else getattr(step, 'current_altitude', 10.0)
        stab = step.get('stability', 0.0) if isinstance(step, dict) else getattr(step, 'field_stability_score', 0.0)
        
        traj_accuracies.append(max(0.0, 1.0 - abs(alt - target) / 10.0))
        stability_scores.append(max(0.0, stab))
        
    final_step = trajectory[-1]
    energy = final_step.get('energy_remaining', 0.0) if isinstance(final_step, dict) else getattr(final_step, 'energy_remaining', 0.0)

    avg_traj_acc = sum(traj_accuracies) / len(traj_accuracies) if traj_accuracies else 0.0
    avg_stab = sum(stability_scores) / len(stability_scores) if stability_scores else 0.0
    energy_eff = max(0.0, energy / 100.0)
    
    # 0.4 tracking, 0.4 energy, 0.2 stability
    score = 0.4 * avg_traj_acc + 0.4 * energy_eff + 0.2 * avg_stab
    score = max(0.0, min(1.0, score))
    return score
