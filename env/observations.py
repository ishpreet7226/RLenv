from pydantic import BaseModel
from typing import List, Dict, Any

class Observation(BaseModel):
    current_altitude: float
    vertical_velocity: float
    field_stability_score: float
    energy_remaining: float
    oscillation_level: float
    external_disturbance: float
    target_altitude: float
    steps_remaining: int
    history: List[Dict[str, Any]]
