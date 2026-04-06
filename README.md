# Anti-Gravity Control Environment (anti_gravity_control_env_v1)

## Overview
The **anti_gravity_control_env_v1** is an advanced control-system benchmark designed for the OpenEnv ecosystem. It simulates the real-time stabilization of experimental anti-gravity propulsion fields under strict disturbance, oscillation, and energy constraints. 

These dynamics model next-generation aerospace stabilization mechanics, directly mapping to real-world applications in:
- **Orbital Positioning Platforms**: Station-keeping for low-gravity space stations.
- **Experimental Propulsion Systems**: Magnetic or anti-gravity lift field maintenance.
- **Autonomous Aerial Stabilization**: Micro-gravity drone stabilization and precision tracking.
- **Future Mobility Infrastructure Research**: Frictionless terrestrial transport pods.

This environment is designed as a benchmark for evaluating long-horizon decision-making in autonomous control systems where agents must simultaneously balance safety constraints, stochastic disturbances, and limited energy budgets. Unlike classical control benchmarks such as CartPole, LunarLander, or inverted-pendulum stabilization tasks, this environment introduces coupled stability–energy tradeoffs that better reflect real aerospace control scenarios.

## Environment Design
The Anti-Gravity Control Environment is a continuous-state, discrete/continuous-action environment focusing on multi-variable safety boundary control.
- **State Variables**: Driven by coupled aerodynamic velocity interactions, stochastic noise injections, and oscillation metrics.
- **Action Space**: Exposes a multi-dimensional schema handling geometric lift vectors up to complex energy manipulations.
- **Observation Structure**: High-density schema evaluating system thresholds across precision variables.
- **Reward Shaping**: Dense scalar rewards shaping incremental progression toward long-horizon trajectory bounds.
- **Termination Conditions**: The simulation immediately halts and penalizes when field stability drops below `0.2` or boundary energy levels reach absolute `0`.

The environment models a closed-loop stabilization controller operating under constrained actuation energy and disturbance uncertainty, making it suitable for evaluating agent robustness in safety-critical aerospace positioning systems and experimental propulsion platforms.

## Action Space
The environment executes a specialized suite of physical commands:
- `increase_lift` / `decrease_lift`: Geometrically alters the vertical thrust vector, impacting ascent/descent while expending core structural energy proportional to the delta.
- `stabilize_field`: Hardens the active levitation frame to naturally dampen accumulating ambient oscillations.
- `redistribute_energy`: Safely re-routes system reserves dynamically to mitigate stability shocks.
- `lock_altitude`: Anchors the target trajectory vector statically for precision alignment overrides.
- `emergency_shutdown`: Immediate termination switch to gracefully crash the system before critical failure thresholds are breached.

## Observation Space
The agent perceives a comprehensive telemetry suite:
- `current_altitude`: Spatial position vector, necessary for boundary tracking.
- `vertical_velocity`: Physics delta exposing current momentum constraints.
- `field_stability_score`: Determines critical collapse proximity.
- `energy_remaining`: Bounding logic to strictly limit unoptimized thrust spamming.
- `oscillation_level`: Tracking for feedback loops signaling impending turbulence vectors.
- `external_disturbance`: Environmental stochastic wind/magnetic noise variables.
- `target_altitude`: The objective vector the system must lock onto.
- `steps_remaining`: Explicit horizon boundary awareness.
- `history`: Aggregated past step buffers for deep recurrent tracking.
*These signals ensure an advanced control policy can infer trajectory acceleration slopes internally without explicit derivative inputs.*

Together, these signals form a partially observable stabilization system where agents must infer latent instability trends from oscillation growth and disturbance patterns rather than relying on direct collapse indicators. This encourages policies to develop predictive stabilization behavior instead of reactive correction loops.

## Tasks
The benchmark evaluates progression incrementally across three tiers:
1. **Hover Stabilization (Easy)**: The agent must strictly maintain altitude bounds (`±0.5` tolerance), minimize ambient oscillation, and severely throttle energy consumption. 
2. **Disturbance Recovery (Medium)**: Evaluates aggressive recovery kinematics. A massive `0.15` disturbance spike knocks the agent offline, requiring them to restore stability (`> 0.8`) strictly within a 10-step horizon. 
3. **Efficiency Trajectory Tracking (Hard)**: True domain manipulation. The agent tracks a shifting, variable-altitude profile across a wide continuous duration while actively resisting stochastic noise and structural oscillations without draining limited fuel.

## Reward Design
The environment implements trajectory-based reward shaping, decomposing into specific metric structures: 
- **Altitude constraint bounding**
- **Sustained stability maintenance**
- **Raw energy efficiency retention**
- **Disturbance recovery bonuses**
- **Hard oscillation penalties**
- **Safety shutdown penalties / Task completion bonuses**

*This dense trajectory shaping ensures policies receive partial-progress feedback recursively, accelerating sample efficiency during sparse-target exploration loops.*

## Graders
Instead of solely relying on cumulative step-rewards, OpenEnv standardizes independent objective evaluation. The graders (`grade_hover`, `grade_disturbance`, `grade_efficiency`) calculate deterministic scores entirely mapped inside normalized float bounds `[0.0, 1.0]`. 
Execution remains 100% reproducible uniformly anchored to `numpy.seed(42)`.

## Baseline Results
The repository ships with an inference pipeline running a simple localized heuristic policy:
- **Hover**: `~0.91`
- **Disturbance**: `~0.88`
- **Efficiency**: `~0.87`

*The baseline policy is a simplistic conditional algorithm evaluating directional vectors ("increase thrust if below limit, decrease if above"). This showcases fundamental baseline solvability, while leaving a wide ceiling for Deep RL algorithms to perfect trajectory smoothness.*

## Environment Interface Summary

| Function | Purpose                           |
| -------- | --------------------------------- |
| reset()  | Initialize task scenario          |
| step()   | Apply control action              |
| state()  | Retrieve internal simulator state |

## Installation
Setup your local execution boundaries natively:
```bash
pip install -r requirements.txt
python inference.py
```

## Docker Usage
Compile and run the deterministic benchmark environment natively:
```bash
docker build -t anti-gravity-env .
docker run anti-gravity-env
```

## HF Space Endpoints
Fully integrated with the HuggingFace OpenEnv judging architecture, operating as a localized REST FastApi cluster:
- `POST /reset`
- `POST /step`
- `GET /state`

## Novelty Statement
Unlike classical benchmarks such as CartPole, LunarLander, or inverted-pendulum stabilization environments, this system introduces coupled stochastic disturbances, energy-bounded actuation, oscillation-driven instability accumulation, and collapse-threshold safety dynamics within a continuous stabilization loop. These properties make it a stronger benchmark for evaluating autonomous agents operating in constrained physical control environments.
