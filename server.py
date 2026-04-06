from fastapi import FastAPI
from env.environment import AntiGravityControlEnv
from env.actions import Action
import traceback

app = FastAPI()
env = AntiGravityControlEnv()

@app.post("/reset")
def reset_env():
    obs = env.reset()
    return obs

@app.post("/step")
def step_env(action: Action):
    obs, rew, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": rew,
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state():
    return env.state()
