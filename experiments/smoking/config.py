"""Experimental values, not a validated model of nicotine biology."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 0
    train_episodes: int = 5
    test_episodes: int = 3
    decisions: int = 120
    decision_sec: float = 0.15
    timestep: float = 0.0001
    cigarette_position: tuple = (14.0, 7.0)
    control_position: tuple = (14.0, -7.0)
    spawn_position: tuple = (0.0, 0.0, 0.2)
    odor_sigma: float = 9.0
    smoking_radius: float = 2.5
    nicotine_intake: float = 0.15  # per second of contact
    nicotine_decay: float = 0.995  # per decision_sec
    tolerance_increase: float = 0.001  # per second of contact
    pam_gain: float = 100.0
    pam_max_rate: float = 80.0
    pam_gate_hz: float = 1.0
    learning_rate: float = 0.03  # per decision, on actually active KC synapses
    kc_threshold: int = 1
    orn_hz: float = 500.0
    kcmbon_gain: float = 20.0  # inherited engineered readout gain from nav_demo
    sniff_ms: float = 150.0
    steer_gain: float = 1.2
    exploration_std: float = 0.12


def nicotine_to_pam(nicotine, tolerance, config):
    return min(config.pam_max_rate, config.pam_gain * nicotine / (1 + tolerance))
