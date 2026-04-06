from pydantic import BaseModel
from typing import Optional
from enum import Enum

class ActionType(str, Enum):
    INCREASE_LIFT = "increase_lift"
    DECREASE_LIFT = "decrease_lift"
    STABILIZE_FIELD = "stabilize_field"
    REDISTRIBUTE_ENERGY = "redistribute_energy"
    LOCK_ALTITUDE = "lock_altitude"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"

class Action(BaseModel):
    action_type: ActionType
    delta: Optional[float] = 0.0
    energy_ratio: Optional[float] = 0.0
    target_altitude: Optional[float] = None
