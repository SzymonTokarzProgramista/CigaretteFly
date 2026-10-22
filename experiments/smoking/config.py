"""Experimental values, not a validated model of nicotine biology."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 0
    arena_seed: int = 0
    scenery: bool = True
    solid_scenery: bool = False
    ignition_cycle: bool = True
    live_fire_positions: tuple = ()
    train_episodes: int = 5
    test_episodes: int = 1
    decisions: int = 40
    decision_sec: float = 0.15
    timestep: float = 0.0001
    cigarette_position: tuple = (14.0, 7.0)
    live_cigarette_positions: tuple = ((14.0, 7.0), (28.0, 10.0), (42.0, 4.0), (56.0, -3.0), (70.0, 7.0))
    control_position: tuple = (14.0, -7.0)
    spawn_position: tuple = (0.0, 0.0, 0.2)
    odor_sigma: float = 9.0
    smoking_radius: float = 2.5
    cigarette_duration_sec: float = 5.0  # supply in LIVE, seconds of intake
    nicotine_intake: float = 0.15  # per second of contact
    nicotine_decay: float = 0.995  # per decision_sec
    tolerance_increase: float = 0.001  # per second of contact
    pam_gain: float = 100.0
    pam_max_rate: float = 80.0
    pam_gate_hz: float = 1.0
    learning_rate: float = 0.3  # episodic LTD, as in upstream learning_driver_mb
    kc_threshold: int = 1
    orn_hz: float = 500.0
    kcmbon_gain: float = 20.0  # inherited engineered readout gain from nav_demo
    sniff_ms: float = 150.0
    steer_gain: float = 1.2
    exploration_std: float = 0.03
    training_decisions: int = 12
    training_protocol: str = 'balanced_exposure'
    low_signal_threshold: float = 80.0
    slow_drive: float = 0.55
    feeding_odor_threshold: float = 0.97
    feeding_relative_valence: float = 0.8
    approach_odor_threshold: float = 0.85
    approach_drive: float = 0.3
    feeding_release_odor: float = 0.95
    feeding_release_valence: float = 1.0
    video_play_speed: float = 0.5
    video_fps: int = 30
    camera: str = 'tracking'
    observe_after_arrival_sec: float = 5.0


def nicotine_to_pam(nicotine, tolerance, config):
    return min(config.pam_max_rate, config.pam_gain * nicotine / (1 + tolerance))
