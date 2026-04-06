import os
import sys

try:
    from openai import OpenAI
except ImportError:
    pass

from env.environment import AntiGravityControlEnv
from env.actions import Action, ActionType
from env.graders import grade_hover, grade_disturbance, grade_efficiency

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def get_heuristic_action(env) -> Action:
    # heuristic policy: increase_lift if altitude < target, decrease if >, stabilize otherwise
    if env.current_altitude < env.target_altitude:
        return Action(action_type=ActionType.INCREASE_LIFT, delta=1.0)
    elif env.current_altitude > env.target_altitude:
        return Action(action_type=ActionType.DECREASE_LIFT, delta=1.0)
    else:
        return Action(action_type=ActionType.STABILIZE_FIELD)

def map_action_to_name(action: Action) -> str:
    if action.action_type == ActionType.INCREASE_LIFT:
        return "increase_lift"
    elif action.action_type == ActionType.DECREASE_LIFT:
        return "decrease_lift"
    elif action.action_type == ActionType.STABILIZE_FIELD:
        return "stabilize_field"
    elif action.action_type == ActionType.LOCK_ALTITUDE:
        return "lock_altitude"
    elif action.action_type == ActionType.REDISTRIBUTE_ENERGY:
        return "redistribute_energy"
    elif action.action_type == ActionType.EMERGENCY_SHUTDOWN:
        return "emergency_shutdown"
    return "unknown"

def run_task(task_name, grader_func, success_threshold):
    env = AntiGravityControlEnv()
    obs = env.reset(task_name)
    
    print(f"[START] task={task_name} env=anti_gravity_control_env_v1 model={MODEL_NAME}")
    
    # Optional task-specific initializations
    if task_name == "disturbance":
        env.external_disturbance = 0.15
        env.vertical_velocity += 0.15
        
    trajectory = []
    rewards_history = []
    
    max_steps = 8
    steps_taken = 0
    done = False
    
    # Store initial state for grader
    trajectory.append({
        "altitude": env.current_altitude,
        "stability": env.field_stability_score,
        "energy_remaining": env.energy_remaining
    })
    
    for i in range(1, max_steps + 1):
        if done:
            break
            
        action = get_heuristic_action(env)
        action_name = map_action_to_name(action)
        
        obs, rew, done, info = env.step(action)
        
        trajectory.append({
            "altitude": env.current_altitude,
            "stability": env.field_stability_score,
            "energy_remaining": env.energy_remaining
        })
        
        rewards_history.append(rew.step_reward)
        steps_taken += 1
        
        print(f"[STEP] step={i} action={action_name} reward={rew.step_reward:.2f} done={str(done).lower()} error=null")
        
    score = grader_func(trajectory)
    success = score >= success_threshold
    
    rewards_str = ",".join([f"{r:.2f}" for r in rewards_history])
    if not rewards_str:
        rewards_str = "0.00"
        
    print(f"[END] success={str(success).lower()} steps={steps_taken} score={score:.4f} rewards={rewards_str}\n")

def run_all_tasks():
    run_task("hover", grade_hover, 0.8)
    run_task("disturbance", grade_disturbance, 0.75)
    run_task("efficiency", grade_efficiency, 0.7)

if __name__ == "__main__":
    run_all_tasks()
