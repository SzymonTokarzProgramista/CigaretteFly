"""Contact-conditioned spiking connectome in a FlyGym/MuJoCo 3D body."""
import argparse
import atexit
import csv
from datetime import datetime
from dataclasses import asdict, replace
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import random
import secrets
import time

from config import Config, nicotine_to_pam
from rendering import overview_position
from cigarette import CigaretteField
from navigation import learned_odor_turn, boundary_turn, odor_compass_turn
from scenery import random_cigarettes, scenery_layout, add_scenery
from recovery import StuckRecovery
from smoking_cycle import SmokingCycle
from movie import StreamingMovie
from physical_fly import ExperimentFly

ROOT = Path(__file__).resolve().parents[2]


def prepare_annotations(source, out):
    import pandas as pd
    table = pd.read_csv(source, sep='\t', low_memory=False)
    required = {'root_id', 'cell_class', 'super_class', 'cell_type', 'hemibrain_type'}
    if not required.issubset(table.columns):
        raise ValueError(f'Annotation columns missing: {required - set(table.columns)}')
    # In the published v2.1.0 table most ORN glomerulus labels live in
    # hemibrain_type. Reuse those labels instead of inventing neuron classes.
    orn = table.super_class.fillna('').str.contains('sensory') & table.cell_class.fillna('').str.contains('olfactory')
    missing = orn & table.cell_type.fillna('').eq('')
    table.loc[missing, 'cell_type'] = table.loc[missing, 'hemibrain_type']
    counts = table.loc[orn, 'cell_type'].value_counts()
    if (counts >= 40).sum() < 18:
        raise ValueError('Need at least 18 ORN classes with >=40 neurons for the upstream six-class protocol')
    path = out / 'annotations_resolved.tsv'
    table.to_csv(path, sep='\t', index=False)
    return path


def load_navigation(path, annotations):
    spec = importlib.util.spec_from_file_location('fly_api_navigation', path)
    nav = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nav)
    nav.ANN = str(annotations)
    return nav


def episode(brain, config, condition, phase, episode_id, out, video, phase_index, preview=None):
    import numpy as np
    from flygym import Fly, Camera
    from flygym.arena import FlatTerrain
    from flygym.examples.locomotion import HybridTurningController

    # A/B use identical body and exploration seeds, independent from brain RNG.
    seed = config.seed + phase_index
    rng = np.random.default_rng(seed)
    conditioning = phase == 'TRAIN' and config.training_protocol == 'balanced_exposure'
    exposure_a = phase_index % 2 == 0
    spawn = config.spawn_position
    if conditioning:
        source = config.cigarette_position if exposure_a else config.control_position
        # Controlled sensory exposure, identical in A/B. Only TRAIN is placed
        # at a source; PRE/TEST remain blind navigation from the common origin.
        spawn = (*source, config.spawn_position[2])
    decisions = config.training_decisions if conditioning else config.decisions
    target_decisions = decisions
    cigarette_positions = config.live_cigarette_positions if phase == 'LIVE' else (config.cigarette_position,)
    field = CigaretteField(cigarette_positions, config.cigarette_duration_sec)
    cycle = SmokingCycle(field, config.live_fire_positions, config.smoking_radius) if phase == 'LIVE' and config.ignition_cycle else None
    fire_positions = config.live_fire_positions if cycle else ()
    cigarette_geoms = []
    arena = FlatTerrain(size=(500, 500))
    points = [spawn[:2], *cigarette_positions, *fire_positions, config.control_position]
    arena_bounds = (min(p[0] for p in points)-20, max(p[0] for p in points)+20,
                    min(p[1] for p in points)-20, max(p[1] for p in points)+20)
    decor = scenery_layout(config.arena_seed, [*cigarette_positions, *fire_positions], spawn, config.control_position) if config.scenery and phase == 'LIVE' else []
    if decor:
        add_scenery(arena, decor, config.solid_scenery)
    (out / f'{condition}_{phase}_{episode_id}_arena.json').write_text(json.dumps({
        'arena_seed': config.arena_seed, 'cigarettes': cigarette_positions, 'fires': fire_positions,
        'behavior_controller': 'explicit_pickup_ignition' if cycle else 'contact',
        'contact_policy': 'ground_only_v2' if not config.solid_scenery else 'ground_and_solid_scenery_v2',
        'navigation_adapter': 'four_sample_odor_compass',
        'bounds': arena_bounds, 'solid_scenery': config.solid_scenery,
        'spawn': spawn, 'control': config.control_position, 'scenery': decor,
    }, indent=2))
    markers = [(f'cigarette_{i}', source, (0.9, 0.65, 0.2, 1))
               for i, source in enumerate(cigarette_positions)]
    markers.append(('control', config.control_position, (0.3, 0.5, 0.8, 1)))
    for name, source, color in markers:
        marker = arena.root_element.worldbody.add('body', name=name, pos=(*source, 0.25))
        geom = marker.add('geom', type='cylinder', size=(0.2, 0.8),
                          quat=(0.7071068, 0, 0.7071068, 0),
                          rgba=color, contype=0, conaffinity=0)
        if name.startswith('cigarette_'):
            cigarette_geoms.append(geom)
    for i, source in enumerate(fire_positions):
        body = arena.root_element.worldbody.add('body', name=f'fire_{i}', pos=(*source, 0))
        body.add('geom', type='cylinder', pos=(0, 0, .2), size=(.8, .2), rgba=(.2,.16,.12,1), contype=0, conaffinity=0)
        body.add('geom', type='ellipsoid', pos=(0,0,1), size=(.45,.45,1), rgba=(1,.3,.03,.85), contype=0, conaffinity=0)
        body.add('geom', type='ellipsoid', pos=(0,0,.7), size=(.25,.25,.7), rgba=(1,.85,.1,1), contype=0, conaffinity=0)
    carried_body = arena.root_element.worldbody.add('body', name='carried_cigarette', mocap=True, pos=(0,0,-100))
    carried_paper = carried_body.add('geom', type='cylinder', size=(.17,.7), pos=(.7,0,0), quat=(.7071068,0,.7071068,0), rgba=(.95,.94,.85,1), contype=0, conaffinity=0)
    carried_body.add('geom', type='cylinder', size=(.18,.2), pos=(-.2,0,0), quat=(.7071068,0,.7071068,0), rgba=(.7,.4,.16,1), contype=0, conaffinity=0)
    ember = carried_body.add('geom', type='sphere', size=(.18,), pos=(1.4,0,0), rgba=(.25,.2,.15,1), contype=0, conaffinity=0)
    puffs = [carried_body.add('geom', type='sphere', size=(.2+.15*i,), pos=(1.4,0,1+i*.8), rgba=(.8,.8,.8,0), contype=0, conaffinity=0) for i in range(3)]
    contacts = [f'{leg}{segment}' for leg in ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
                for segment in ['Tibia', 'Tarsus1', 'Tarsus2', 'Tarsus3', 'Tarsus4', 'Tarsus5']]
    fly = ExperimentFly(solid_scenery=config.solid_scenery, enable_adhesion=True, contact_sensor_placements=contacts,
              spawn_pos=spawn)
    movie_writer = StreamingMovie(out / f'{condition}_{phase}_{episode_id}.mp4', config.video_fps) if video else None
    cameras = []
    if video or preview is not None:
        class ExperimentCamera(Camera):
            behavior = 'WALK'
            distance = float('nan')
            position = spawn
            trail = None

            def _update_camera(self, physics, floor_height, observation):
                if config.camera == 'tracking':
                    position = np.asarray(observation['pos'])
                    physics.bind(self._cam).xpos = position + np.array([0, -10, 14])
                else:
                    camera_position = overview_position(
                        observation['pos'], [*cigarette_positions, *fire_positions, config.control_position])
                    physics.bind(self._cam).xpos = camera_position
                    # FlyGym's default far plane is just 3 * extent (60 mm).
                    # Adaptive overview cameras can be much farther away.
                    visual = physics.model.vis.map
                    visual.zfar = max(3.0, (camera_position[2] - floor_height + 40) /
                                      max(float(physics.model.stat.extent), 0.1))
                    visual.fogstart = visual.zfar + 1
                    visual.fogend = visual.zfar + 2

            def render(self, *args, **kwargs):
                import cv2
                frame = super().render(*args, **kwargs)
                if frame is not None:
                    # Camera retains this same array for save_video.
                    cv2.rectangle(frame, (10, 43), (790, 130), (25, 25, 25), -1)
                    cv2.putText(frame, f'{condition} | {phase} | {self.behavior}',
                                (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
                    cv2.putText(frame, f'cigarette distance: {self.distance:.2f} mm | orange=cigarette, blue=control',
                                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    if phase == 'LIVE':
                        cv2.putText(frame, 'Supplies: ' + ' | '.join(f'C{i+1}: {100 * item.fraction:.0f}%' for i, item in enumerate(field.supplies)),
                                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    if self.trail is None:
                        self.trail = []
                    self.trail.append(tuple(self.position[:2]))
                    if len(self.trail) > 2000:
                        del self.trail[0]
                    points = [*self.trail, *cigarette_positions, *fire_positions, config.control_position]
                    xlo = min(p[0] for p in points) - 4
                    xhi = max(p[0] for p in points) + 4
                    ylo = min(p[1] for p in points) - 4
                    yhi = max(p[1] for p in points) + 4
                    def pixel(point):
                        return (int(25 + 195 * (point[0] - xlo) / (xhi - xlo)),
                                int(580 - 140 * (point[1] - ylo) / (yhi - ylo)))
                    cv2.rectangle(frame, (10, 410), (240, 598), (25, 25, 25), -1)
                    cv2.putText(frame, 'MAP / PATH', (20, 429), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (255, 255, 255), 1)
                    cv2.polylines(frame, [np.array([pixel(p) for p in self.trail], dtype=np.int32)],
                                  False, (150, 150, 150), 1)
                    for point, label, color in [(p, f'C{i+1}', (245, 175, 55) if field.supplies[i].available else (130, 130, 130))
                                                for i, p in enumerate(cigarette_positions)] + [
                                                *[(p, f'F{i+1}', (255, 100, 40)) for i, p in enumerate(fire_positions)],
                                                (config.control_position, 'CTRL', (80, 150, 245)),
                                                (self.position, 'FLY', (95, 245, 130))]:
                        pt = pixel(point)
                        cv2.circle(frame, pt, 4, color, -1)
                        cv2.putText(frame, label, (min(pt[0] + 6, 200), pt[1] - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                    if preview is not None:
                        preview.update(frame)
                    if movie_writer is not None:
                        movie_writer.write(frame)
                    self._frames.clear()
                    self._timestamp_per_frame.clear()
                return frame

        camera_class = ExperimentCamera
        camera_parameters = {'mode': 'fixed', 'pos': (0, -10, 14),
                             'euler': (0.62, 0, 0), 'fovy': 45}
        if config.camera == 'arena':
            camera_parameters = {'mode': 'fixed', 'pos': (9, 0, 40),
                                 'euler': (0, 0, 0), 'fovy': 45}
        cameras = [camera_class(attachment_point=arena.root_element.worldbody,
                               camera_name='experiment_camera',
                               targeted_fly_names=[fly.name], camera_parameters=camera_parameters,
                               window_size=(800, 608), play_speed=config.video_play_speed,
                               fps=config.video_fps, timestamp_text=True)]
    sim = HybridTurningController(fly=fly, cameras=cameras, arena=arena,
                                  timestep=config.timestep, seed=seed)
    if not config.solid_scenery:
        ground_id = sim.physics.model.name2id('ground','geom')
        for first, second in zip(sim.physics.model.pair_geom1, sim.physics.model.pair_geom2):
            for geom_id in (first, second):
                name = sim.physics.model.id2name(int(geom_id),'geom')
                if geom_id != ground_id and not name.startswith(f'{fly.name}/'):
                    sim.close()
                    raise RuntimeError(f'Visual scenery unexpectedly has a compiled physical contact: {name}')
    nicotine = tolerance = near_time = 0.0
    reach_time = None
    rows = []
    obs, _ = sim.reset(seed)
    started = time.perf_counter()
    path_length = 0.0
    next_render_time = 0.0
    feeding_state = False
    recovery = StuckRecovery()
    previous_intake = 0.0
    learned_navigation_valence = getattr(brain, 'navigation_valence', 1.0)
    action = np.zeros(2)
    recovery_state = ''
    actual_action = np.zeros(2)
    initial_weights = np.asarray(brain.syn.w[brain.plastic_pos]).copy()

    def odor(position, source):
        if phase == 'LIVE' and source == config.cigarette_position:
            return 1.0 if cycle and cycle.carried is not None else field.odor(position, config.odor_sigma)
        return float(np.exp(-np.sum((position - source) ** 2) / (2 * config.odor_sigma ** 2)))

    def sniff(position, pam):
        c_a = odor(position, config.cigarette_position) if not conditioning or exposure_a else 0.0
        c_b = odor(position, config.control_position) if not conditioning or not exposure_a else 0.0
        rates = {i: c_a * config.orn_hz for i in brain.A}
        rates.update({i: c_b * config.orn_hz for i in brain.B})
        rates.update({int(i): pam for i in brain.g['pam']})
        spikes = brain._episode(rates, config.sniff_ms)
        return spikes, float(spikes[brain.g['mbon']].sum())

    try:
        decision = 0
        while decision < target_decisions:
            pos = np.array(obs['fly'][0][:2])
            distance = field.distance(pos) if phase == 'LIVE' else float(np.linalg.norm(pos - config.cigarette_position))
            contact = distance < config.smoking_radius
            if contact and reach_time is None:
                reach_time = decision * config.decision_sec
            pam = nicotine_to_pam(nicotine, tolerance, config) if phase in ('TRAIN', 'LIVE') and condition == 'A' else 0.0
            head = np.asarray(obs['fly_orientation'][:2])
            head = head / max(np.linalg.norm(head), 1e-9)
            lateral = np.array([-head[1], head[0]])
            left, v_left = sniff(pos + 1.5 * head + 1.2 * lateral, pam)
            right, v_right = sniff(pos + 1.5 * head - 1.2 * lateral, pam)
            counts = left + right
            pam_activity = float(counts[brain.g['pam']].mean() / (2 * config.sniff_ms / 1000))
            applied = False
            if phase in ('TRAIN', 'LIVE') and condition == 'A' and pam > 0 and pam_activity >= config.pam_gate_hz and (phase != 'LIVE' or previous_intake > 0):
                active = brain.g['kc'][counts[brain.g['kc']] >= config.kc_threshold]
                hot = np.isin(brain.plastic_pre, active)
                weights = np.array(brain.syn.w[brain.plastic_pos])
                weights[hot] *= 1 - config.learning_rate
                brain.syn.w[brain.plastic_pos] = weights * brain.br['volt']
                applied = bool(hot.any())
            # Fixed MBON-to-turn adapter inherited from nav_demo. Coordinates
            # are used only for sensory concentration and contact, never steering.
            delta = config.steer_gain * (v_left - v_right) / max(v_left + v_right, 200)
            delta = float(np.clip(delta + rng.normal(0, config.exploration_std), -0.32, 0.32))
            body_odor = max(odor(pos, config.cigarette_position), odor(pos, config.control_position))
            expected_valence = 0.0
            for antenna in [pos + 1.5 * head + 1.2 * lateral, pos + 1.5 * head - 1.2 * lateral]:
                c_a = odor(antenna, config.cigarette_position) if not conditioning or exposure_a else 0.0
                c_b = odor(antenna, config.control_position) if not conditioning or not exposure_a else 0.0
                expected_valence += c_a * brain.mbon_reference['A'] + c_b * brain.mbon_reference['B']
            relative_valence = (v_left + v_right) / max(expected_valence, 1.0)
            if phase == 'LIVE' and previous_intake > 0:
                learned_navigation_valence = min(learned_navigation_valence, relative_valence)
            if phase == 'LIVE':
                antenna_left = pos + 1.5 * head + 1.2 * lateral
                antenna_right = pos + 1.5 * head - 1.2 * lateral
                samples = [pos+2*head, pos-2*head, pos+2*lateral, pos-2*lateral]
                delta = odor_compass_turn(
                    [odor(p,config.cigarette_position) for p in samples],
                    [odor(p,config.control_position) for p in samples],
                    learned_navigation_valence, config.feeding_relative_valence)
                delta = float(np.clip(delta + rng.normal(0, config.exploration_std), -0.32, 0.32))
            # A generic feeding reflex on learned low avoidance in strong
            # odor; identical for either odor and both conditions. No target
            # coordinate is used to choose the steering direction.
            if feeding_state:
                feeding_state = body_odor > config.feeding_release_odor and relative_valence < config.feeding_release_valence
            else:
                feeding_state = relative_valence < config.feeding_relative_valence and body_odor > config.feeding_odor_threshold
            feeding = feeding_state
            learned_approach = relative_valence < config.feeding_relative_valence and body_odor > config.approach_odor_threshold
            if cycle:
                feeding = cycle.state in ('IGNITING', 'SMOKING')
                feeding_state = feeding
                if cycle.state in ('SEARCH_FIRE', 'LEAVE_FIRE'):
                    delta = cycle.fire_turn(pos, head)
            drive = (0.0 if feeding else config.approach_drive if learned_approach else
                     config.slow_drive if v_left + v_right < config.low_signal_threshold else 1.0)
            if phase == 'LIVE':
                drive = 0.0 if feeding else 0.8
            boundary_recovery = False
            if phase == 'LIVE' and not feeding:
                wall_turn = boundary_turn(pos, head, arena_bounds)
                if wall_turn is not None:
                    delta = wall_turn
                    boundary_recovery = True
            action = drive * np.array([1 + delta, 1 - delta])
            if boundary_recovery:
                # Turn using unequal forward amplitudes, without reversing CPG frequencies.
                action = np.array([1.0, 0.6] if delta > 0 else [0.6, 1.0])
            recovery_state = ''
            if phase == 'LIVE':
                recovery_state, recovery_action = recovery.update(
                    float(sim.curr_time), pos, drive > 0 and not feeding and not boundary_recovery, delta)
                if recovery_action is not None:
                    action = np.asarray(recovery_action)
                    drive = float(np.max(np.abs(action)))
            body_started = time.perf_counter()
            previous_position = pos.copy()
            contact_time = 0.0
            intake_time = 0.0
            for _ in range(round(config.decision_sec / config.timestep)):
                if phase == 'LIVE':
                    actual_action += np.clip(action-actual_action, -4*config.timestep, 4*config.timestep)
                else:
                    actual_action = action
                obs, _, terminated, truncated, _ = sim.step(actual_action)
                current_position = np.asarray(obs['fly'][0][:2])
                path_length += float(np.linalg.norm(current_position - previous_position))
                previous_position = current_position.copy()
                contact_distance = field.distance(current_position) if phase == 'LIVE' else float(np.linalg.norm(current_position - config.cigarette_position))
                contact_index = field.contact(current_position, config.smoking_radius)
                touching = contact_index is not None if phase == 'LIVE' else contact_distance < config.smoking_radius
                # Detect entry into a strong learned odor between brain
                # decisions; otherwise a 150 ms stride can skip the stop zone.
                if not cycle and not feeding and relative_valence < config.feeding_relative_valence:
                    sensed_a = odor(current_position, config.cigarette_position) if not conditioning or exposure_a else 0.0
                    sensed_b = odor(current_position, config.control_position) if not conditioning or not exposure_a else 0.0
                    if max(sensed_a, sensed_b) > config.feeding_odor_threshold:
                        feeding = True
                        feeding_state = True
                        recovery.cancel()
                        recovery_state = ''
                        action = np.zeros(2)
                        drive = 0.0
                if touching:
                    contact_time += config.timestep
                    near_time += config.timestep
                    if reach_time is None:
                        reach_time = float(sim.curr_time)
                intake = 0.0
                consumed_index = None
                if cycle:
                    previous_carried = cycle.carried
                    intake = cycle.update(current_position, config.timestep)
                    consumed_index = cycle.last_intake_index
                    if cycle.carried is not None and previous_carried is None:
                        sim.physics.bind(cigarette_geoms[cycle.carried]).rgba = (0,0,0,0)
                    feeding = cycle.state in ('IGNITING', 'SMOKING')
                    feeding_state = feeding
                    if feeding:
                        recovery.cancel()
                        recovery_state = ''
                        action = np.zeros(2)
                        drive = 0.0
                    elif previous_carried is not None and cycle.carried is None:
                        # Continue simulation after finishing; next brain decision resumes search.
                        action = 0.8 * np.ones(2)
                        drive = 0.8
                if phase in ('TRAIN', 'LIVE'):
                    nicotine *= config.nicotine_decay ** (config.timestep / config.decision_sec)
                    if not cycle and touching:
                        intake = field.supplies[contact_index].consume(config.timestep) if phase == 'LIVE' else config.timestep
                        consumed_index = contact_index
                    intake_time += intake
                    nicotine += config.nicotine_intake * intake
                    tolerance += config.tolerance_increase * intake
                    if phase == 'LIVE' and intake > 0 and not field.supplies[consumed_index].available:
                        sim.physics.bind(cigarette_geoms[consumed_index]).rgba = (0.25,0.25,0.25,1)
                        if cycle:
                            source = cigarette_positions[consumed_index]
                            sim.physics.bind(cigarette_geoms[consumed_index]).pos = (current_position[0]-source[0],current_position[1]-source[1],0)
                            sim.physics.bind(cigarette_geoms[consumed_index]).size = (.18,.2,0)
                # Sampling at video cadence avoids updating fly colors and
                # collecting camera observations on all 10,000 steps/second.
                if cameras and sim.curr_time >= next_render_time:
                    cameras[0].behavior = ('SMOKING - NICOTINE INTAKE' if touching and phase in ('TRAIN', 'LIVE') and (phase != 'LIVE' or (contact_index is not None and field.supplies[contact_index].available)) else
                                          'FEEDING - MOTOR OFF' if feeding else
                                          f'UNSTUCK - {recovery_state}' if recovery_state else
                                          'BOUNDARY TURN' if boundary_recovery else
                                          'LEARNED APPROACH' if learned_approach else 'WALK')
                    cameras[0].distance = contact_distance
                    cameras[0].position = current_position.copy()
                    if decision >= decisions:
                        cameras[0].behavior += ' | POST-ARRIVAL'
                    if cycle:
                        cameras[0].behavior = cycle.state + (f' | C{cycle.carried+1}' if cycle.carried is not None else '')
                        if recovery_state:
                            cameras[0].behavior += f' | UNSTUCK {recovery_state}'
                        if cycle.carried is not None:
                            orientation = np.asarray(obs['fly_orientation'][:2])
                            yaw = math.atan2(orientation[1], orientation[0])
                            height = float(obs['fly'][0][2])
                            carry_position = current_position + orientation
                            sim.physics.bind(carried_body).mocap_pos = (*carry_position, height+.6)
                            sim.physics.bind(carried_body).mocap_quat = (math.cos(yaw/2),0,0,math.sin(yaw/2))
                            burning = cycle.state == 'SMOKING'
                            sim.physics.bind(ember).rgba = (1,.2,.02,1) if burning else (.25,.2,.15,1)
                            for i, puff in enumerate(puffs):
                                sim.physics.bind(puff).rgba = (.8,.8,.8,.25 if burning else 0)
                                sim.physics.bind(puff).pos = (1.4+.25*math.sin(sim.curr_time*4+i),0,.6+(sim.curr_time*.8+i*.7)%2.7)
                        else:
                            sim.physics.bind(carried_body).mocap_pos = (0,0,-100)
                        sim.physics.forward()
                    sim.render()
                    next_render_time += config.video_play_speed / config.video_fps
                if terminated or truncated:
                    raise RuntimeError('Body simulation terminated before the scheduled episode end')
            previous_intake = intake_time
            new_distance = field.distance(obs['fly'][0][:2]) if phase == 'LIVE' else float(np.linalg.norm(np.array(obs['fly'][0][:2]) - config.cigarette_position))
            displacement = float(np.linalg.norm(np.asarray(obs['fly'][0][:2]) - pos))
            rows.append(dict(condition=condition, episode=episode_id, phase=phase,
                             time=decision * config.decision_sec, x=float(pos[0]), y=float(pos[1]),
                             distance_to_cigarette=distance, time_near_cigarette=near_time,
                             time_to_reach_cigarette=reach_time, nicotine=nicotine, tolerance=tolerance,
                             smoking=intake_time > 0, pam_rate=pam,
                             nicotine_intake_time_sec=intake_time,
                             cigarette_remaining_sec=sum(item.remaining_sec for item in field.supplies) if phase == 'LIVE' else None,
                             cigarette_depleted=phase == 'LIVE' and not any(item.available for item in field.supplies),
                             cigarette_supplies_json=json.dumps([item.remaining_sec for item in field.supplies]) if phase == 'LIVE' else '',
                             depleted_cigarettes=sum(not item.available for item in field.supplies) if phase == 'LIVE' else 0,
                             contact_time_sec=contact_time, feeding=feeding, motor_drive=drive,
                             training_exposure=('cigarette' if exposure_a else 'control') if conditioning else '',
                             relative_MBON_valence=relative_valence,
                             steering_delta=delta, boundary_recovery=boundary_recovery,
                             smoking_cycle_state=cycle.state if cycle else '',
                             carried_cigarette=cycle.carried if cycle else None,
                             finished_cigarettes=cycle.finished if cycle else 0,
                             ignition_progress_sec=cycle.ignition_progress if cycle else 0,
                             recovery_state=recovery_state, recovery_attempts=recovery.attempts,
                             descending_left=float(action[0]), descending_right=float(action[1]),
                             navigation_valence=learned_navigation_valence if phase == 'LIVE' else None,
                             selected_PAM_activity=pam_activity,
                             selected_MBON_activity=float(counts[brain.g['mbon']].mean() / (2 * config.sniff_ms / 1000)),
                             mean_KC_MBON_weight=float(np.asarray(brain.syn.w[brain.plastic_pos]).mean()),
                             approach_velocity=(distance - new_distance) / config.decision_sec,
                             x_end=float(obs['fly'][0][0]), y_end=float(obs['fly'][0][1]),
                             speed_mm_per_sec=displacement / config.decision_sec,
                             path_length_mm=path_length,
                             body_wall_sec=time.perf_counter() - body_started,
                             observation_tail=decision >= decisions,
                             plasticity_applied=applied))
            if phase in ('TEST', 'LIVE') and reach_time is not None:
                target_decisions = max(decisions, math.ceil(
                    (reach_time + config.observe_after_arrival_sec) / config.decision_sec))
            if decision == 0 or (decision + 1) % 10 == 0 or decision + 1 == target_decisions:
                print(f'[{condition}/{phase}] {decision + 1}/{target_decisions} '
                      f'body={(decision + 1) * config.decision_sec:.2f}s '
                      f'path={path_length:.1f}mm speed={displacement / config.decision_sec:.1f}mm/s '
                      f'distance={new_distance:.1f}mm contact={near_time:.2f}s PAM={pam:.1f}Hz '
                      f'state={cycle.state if cycle else ("FEEDING" if feeding else "WALK")} recovery={recovery_state or "none"} '
                      f'wall={time.perf_counter() - started:.1f}s', flush=True)
            decision += 1
        if phase not in ('TRAIN', 'LIVE'):
            assert np.array_equal(initial_weights, np.asarray(brain.syn.w[brain.plastic_pos]))
        if cameras and video:
            movie = out / f'{condition}_{phase}_{episode_id}.mp4'
            movie_writer.close()
            movie.with_suffix('.json').write_text(json.dumps({
                'condition': condition, 'phase': phase, 'episode': episode_id,
                'simulated_seconds': len(rows) * config.decision_sec,
                'planned_seconds': decisions * config.decision_sec,
                'post_arrival_observation_seconds': config.observe_after_arrival_sec,
                'time_near_cigarette_sec': near_time,
                'final_distance_mm': new_distance,
                'cigarettes': [{'position': p, 'remaining_sec': item.remaining_sec}
                               for p, item in zip(cigarette_positions, field.supplies)],
                'feeding_decisions': sum(r['feeding'] for r in rows),
                'description': ('Explicit pickup/fire/ignition controller; connectome learns odor preference.' if cycle else
                                'Continuous search, contact, nicotine intake and plasticity.' if phase == 'LIVE' else
                                'PRE is before training; B is the no-PAM control; learned stopping is evaluated in A/TEST.'),
            }, indent=2))
            print(f'Video: {movie} (~{len(rows) * config.decision_sec / config.video_play_speed:.1f}s)', flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        # Preserve completed decisions and frames; never silently restart invalid physics.
        prefix = out / f'{condition}_{phase}_{episode_id}_partial'
        if rows:
            with prefix.with_suffix('.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        np.save(out / f'{condition}_{phase}_{episode_id}_partial_weights.npy', np.asarray(brain.syn.w[brain.plastic_pos]))
        prefix.with_suffix('.json').write_text(json.dumps({'error': str(exc), 'error_type': type(exc).__name__,
            'simulated_seconds': float(sim.curr_time), 'completed_decisions': len(rows),
            'arena_seed': config.arena_seed, 'requested_action': action.tolist(),
            'applied_action': actual_action.tolist(), 'recovery_state': recovery_state,
            'smoking_cycle_state': cycle.state if cycle else None}, indent=2))
        if movie_writer is not None:
            movie_writer.close()
            if movie_writer.path.exists():
                movie_writer.path.rename(prefix.with_suffix('.mp4'))
        print(f'Interrupted simulation. Partial results: {prefix}', flush=True)
        raise
    finally:
        if movie_writer is not None:
            movie_writer.close()
        sim.close()
    return rows


def plots(rows, out, before, after, config):
    rows = [r for r in rows if not r['observation_tail']]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for condition in after:
        selected = [r for r in rows if r['condition'] == condition]
        episodes = sorted(set(r['episode'] for r in selected))
        groups = [[r for r in selected if r['episode'] == e] for e in episodes]
        axes[0, 0].plot(episodes, [np.mean([r['distance_to_cigarette'] for r in g]) for g in groups], label=condition)
        axes[0, 1].plot(episodes, [next((r['time_to_reach_cigarette'] for r in g if r['time_to_reach_cigarette'] is not None), float('nan')) for g in groups], label=condition)
        axes[0, 2].plot(episodes, [g[-1]['time_near_cigarette'] for g in groups], label=condition)
        axes[1, 0].plot(np.arange(len(selected)) * config.decision_sec, [r['nicotine'] for r in selected], label=condition)
        axes[1, 1].plot(np.arange(len(selected)) * config.decision_sec, [r['pam_rate'] for r in selected], label=condition)
        axes[1, 2].hist(after[condition], bins=40, histtype='step', label=f'{condition} after')
    axes[1, 2].hist(before, bins=40, histtype='step', label='before')
    for ax, title in zip(axes.flat, ['Mean distance (mm)', 'First contact (s; missing = not reached)', 'Time near (s)', 'Nicotine vs body time (s)', 'PAM drive (Hz) vs body time (s)', 'KC→MBON weights (V)']):
        ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'summary.png'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, default=ROOT / 'vendor/Drosophila_brain_model')
    ap.add_argument('--annotations', type=Path, default=ROOT / 'data/annotations.tsv')
    ap.add_argument('--out', type=Path, default=ROOT / 'results/smoking')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--arena-seed', type=int, help='reproduce the layout; otherwise LIVE gets a new random layout each run')
    ap.add_argument('--cigarette-count', type=int, default=5, help='number of random LIVE cigarettes (1-12)')
    ap.add_argument('--legacy-contact-smoking', action='store_true', help='use old LIVE contact intake without pickup and ignition')
    ap.add_argument('--fire-count', type=int, default=2)
    ap.add_argument('--fire-position', type=float, nargs=2, action='append', metavar=('X','Y'))
    ap.add_argument('--solid-scenery', action='store_true', help='experimental tree and rock collisions')
    ap.add_argument('--no-scenery', action='store_true', help='disable trees, rocks and grass')
    ap.add_argument('--quick', action='store_true', help='2 decisions per phase, validates execution only')
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--preview', action='store_true', help='show current simulation frames in a local browser')
    ap.add_argument('--preview-port', type=int, default=8765)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--test-only', action='store_true')
    mode.add_argument('--live', action='store_true', help='one continuous search and nicotine-intake simulation, with active learning')
    ap.add_argument('--cigarette-position', type=float, nargs=2, action='append', metavar=('X', 'Y'),
                    help='LIVE cigarette position in mm; repeat to override the default five sources')
    ap.add_argument('--cigarette-duration', type=float, default=5.0,
                    help='seconds of nicotine intake before a cigarette is exhausted in LIVE (default 5)')
    ap.add_argument('--duration', type=float, help='simulated seconds for --live (default 300)')
    ap.add_argument('--weights-dir', type=Path, help='directory containing A_weights.npy, B_weights.npy')
    ap.add_argument('--contact-start', action='store_true', help='start beside the object, for contact pathway diagnostics')
    ap.add_argument('--decisions', type=int)
    ap.add_argument('--train-episodes', type=int)
    ap.add_argument('--test-episodes', type=int)
    ap.add_argument('--observe-after-arrival', type=float, default=None,
                    help='minimum simulated seconds observed after first contact in TEST (default 5)')
    ap.add_argument('--training-decisions', type=int)
    ap.add_argument('--training-protocol', choices=['balanced_exposure', 'free'], default='balanced_exposure')
    ap.add_argument('--conditions', nargs='+', choices=['A', 'B'], default=['A', 'B'])
    ap.add_argument('--camera', choices=['tracking', 'arena'], default='tracking')
    ap.add_argument('--play-speed', type=float, default=0.5, help='body seconds / video seconds')
    args = ap.parse_args()
    if not 1 <= args.fire_count <= 8:
        ap.error('--fire-count must be between 1 and 8')
    if args.fire_position and (not args.live or not all(math.isfinite(v) for p in args.fire_position for v in p)):
        ap.error('--fire-position requires LIVE and finite coordinates')
    if not 1 <= args.cigarette_count <= 12:
        ap.error('--cigarette-count must be between 1 and 12')
    if not 0 <= args.preview_port <= 65535:
        ap.error('--preview-port must be between 0 and 65535 (0 selects a free port)')
    if args.test_only and not args.weights_dir:
        ap.error('--test-only requires --weights-dir from a completed training run')
    for path in [args.annotations, args.model_dir / 'model.py', args.model_dir / 'Connectivity_783.parquet']:
        if not path.is_file():
            ap.error(f'Missing input: {path}. See experiments/smoking/README.md')
    import numpy as np
    if args.play_speed <= 0:
        ap.error('--play-speed must be positive')
    if not math.isfinite(args.cigarette_duration) or args.cigarette_duration <= 0:
        ap.error('--cigarette-duration must be finite and positive')
    config = replace(Config(), cigarette_duration_sec=args.cigarette_duration, seed=args.seed, camera=args.camera, video_play_speed=args.play_speed,
                     training_protocol=args.training_protocol)
    if args.quick:
        config = replace(config, train_episodes=1, test_episodes=1, decisions=2, training_decisions=2,
                         observe_after_arrival_sec=0.0)
    if args.observe_after_arrival is not None:
        if not math.isfinite(args.observe_after_arrival) or args.observe_after_arrival < 0:
            ap.error('--observe-after-arrival must be finite and nonnegative')
        config = replace(config, observe_after_arrival_sec=args.observe_after_arrival)
    for name in ['decisions', 'train_episodes', 'test_episodes', 'training_decisions']:
        value = getattr(args, name)
        if value is not None:
            if value < 1:
                ap.error(f'{name} must be positive')
            config = replace(config, **{name: value})
    arena_seed = args.arena_seed if args.arena_seed is not None else secrets.randbits(32) if args.live else 0
    config = replace(config, arena_seed=arena_seed, scenery=not args.no_scenery, solid_scenery=args.solid_scenery)
    if args.live and not args.cigarette_position:
        config = replace(config, live_cigarette_positions=random_cigarettes(
            arena_seed, args.cigarette_count, config.spawn_position[:2], config.control_position))
    if args.cigarette_position:
        if not args.live:
            ap.error('--cigarette-position requires --live')
        if not all(math.isfinite(v) for p in args.cigarette_position for v in p):
            ap.error('Cigarette coordinates must be finite')
        positions = tuple(dict.fromkeys(tuple(p) for p in args.cigarette_position))
        config = replace(config, live_cigarette_positions=positions)
    if args.live:
        fires = tuple(tuple(p) for p in args.fire_position) if args.fire_position else random_cigarettes(
            arena_seed ^ 0xF1AE, args.fire_count, config.spawn_position[:2], config.control_position)
        config = replace(config, live_fire_positions=fires, ignition_cycle=not args.legacy_contact_smoking)
    if args.contact_start:
        start = config.live_cigarette_positions[0] if args.live else config.cigarette_position
        config = replace(config, spawn_position=(*start, config.spawn_position[2]))
    if args.duration is not None and not args.live:
        ap.error('--duration requires --live')
    if args.live:
        duration = args.duration if args.duration is not None else (0.3 if args.quick else 300.0)
        if not math.isfinite(duration) or duration <= 0:
            ap.error('--duration must be finite and positive')
        config = replace(config, decisions=math.ceil(duration / config.decision_sec), test_episodes=1)
    random.seed(config.seed); np.random.seed(config.seed)
    # Keep existing experiments intact, including their movie/CSV correspondence.
    if args.out.exists() and any(args.out.iterdir()):
        args.out = args.out / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'Output: {args.out}\nEpisode: {config.decisions * config.decision_sec:.2f}s simulated, '
          f'~{config.decisions * config.decision_sec / config.video_play_speed:.2f}s video', flush=True)
    if args.live:
        print(f'Arena seed: {config.arena_seed}; cigarettes: {config.live_cigarette_positions}', flush=True)
        if config.ignition_cycle:
            print(f'Fire sources: {config.live_fire_positions}', flush=True)
    preview = None
    if args.preview:
        from preview import BrowserPreview
        try:
            preview = BrowserPreview(args.preview_port)
        except OSError as exc:
            ap.error(f'Cannot open preview: {exc}. Try --preview-port 0')
        atexit.register(preview.close)
        print(f'Live preview: {preview.url} (open in your browser)', flush=True)
    nav_path = ROOT / 'vendor/fly-api/experiments/navigation/nav_demo.py'
    annotations = prepare_annotations(args.annotations, args.out)
    nav = load_navigation(nav_path, annotations.resolve())
    import brian2
    brian2.prefs.codegen.runtime.cython.cache_dir = str(ROOT / '.cache/brian2')
    brain = nav.Brain(str(args.model_dir.resolve()), seed=config.seed,
                      gain=config.kcmbon_gain, orn_hz=config.orn_hz)
    # This experiment only consumes spike counts, not individual spike times.
    # Avoid retaining minutes of events during long LIVE runs.
    brain.net.remove(brain.spk)
    brain.spk = brian2.SpikeMonitor(brain.neu, record=False)
    brain.net.add(brain.spk)
    before = brain.w_naive.copy()
    digest = hashlib.sha256()
    for path in [args.annotations, args.model_dir / 'Completeness_783.csv', args.model_dir / 'Connectivity_783.parquet']:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    digest.update(json.dumps({'seed': config.seed, 'gain': config.kcmbon_gain}, sort_keys=True).encode())
    fingerprint = digest.hexdigest()
    if args.weights_dir is not None:
        saved = json.loads((args.weights_dir / 'provenance.json').read_text())
        if saved['circuit_fingerprint'] != fingerprint:
            raise ValueError('Checkpoint circuit differs: use the same seed, gain, model and annotations')
    (args.out / 'provenance.json').write_text(json.dumps({
        'circuit_fingerprint': fingerprint,
        'fly_api_commit': 'a6ad07a810b1a43cd0356149c07b32105eb46d2a',
        'brain_model_commit': '91bdd1e7dcf193f3e7ca5a8933497fcef63b7960',
        'annotation_release': 'v2.1.0 (default)',
        'live_behavior': 'explicit_pickup_fire_ignition' if args.live and config.ignition_cycle else 'contact',
        'mode': 'live' if args.live else 'test_only' if args.test_only else 'pre_train_test',
    }, indent=2))
    rows, after = [], {}
    brain.net.store('initial')
    # Reference readout from the unmodified circuit, not a learned policy or
    # a source-specific preference. Restore neural state and RNG afterwards.
    brain.mbon_reference = {}
    for label, neurons in [('A', brain.A), ('B', brain.B)]:
        brain.net.restore('initial', restore_random_state=True)
        brain.prev = np.zeros_like(brain.prev)
        spikes = brain._episode({i: config.orn_hz for i in neurons}, config.sniff_ms)
        brain.mbon_reference[label] = float(spikes[brain.g['mbon']].sum())
    (args.out / 'readout_reference.json').write_text(json.dumps(brain.mbon_reference, indent=2))
    for condition in dict.fromkeys(args.conditions):
        # Restore time, neuronal state, monitors and random streams for matched controls.
        brain.net.restore('initial', restore_random_state=True)
        brain.prev = np.zeros_like(brain.prev)
        if args.weights_dir is not None:
            weights = np.load(args.weights_dir / f'{condition}_weights.npy')
            if weights.shape != before.shape:
                raise ValueError('Checkpoint shape differs from this circuit')
            if condition == 'A' and np.array_equal(weights, before):
                print('Checkpoint A contains naive weights: run new training to acquire an odor association.', flush=True)
            brain.syn.w[brain.plastic_pos] = weights * brain.br['volt']
        if args.live:
            # Probe current learned odor response without changing circuit state or RNG.
            brain.net.store('navigation_probe')
            saved_prev = brain.prev.copy()
            spikes = brain._episode({i: config.orn_hz for i in brain.A}, config.sniff_ms)
            brain.navigation_valence = float(spikes[brain.g['mbon']].sum()) / max(brain.mbon_reference['A'], 1.0)
            brain.net.restore('navigation_probe', restore_random_state=True)
            brain.prev = saved_prev
            print(f'Learned odor valence: {brain.navigation_valence:.3f}', flush=True)
        phases = ['LIVE'] if args.live else ['TEST'] if args.test_only else ['PRE', 'TRAIN', 'TEST']
        episode_id = 0
        for phase in phases:
            n = config.train_episodes if phase == 'TRAIN' else config.test_episodes
            for phase_index in range(n):
                print(f'[{condition}] {phase} episode={episode_id}', flush=True)
                rows.extend(episode(brain, config, condition, phase, episode_id, args.out, not args.no_video, phase_index, preview))
                with (args.out / 'smoking_experiment.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
                episode_id += 1
        after[condition] = np.asarray(brain.syn.w[brain.plastic_pos]).copy()
        np.save(args.out / f'{condition}_weights.npy', after[condition])
    np.save(args.out / 'initial_weights.npy', before)
    np.save(args.out / 'plastic_pre.npy', brain.plastic_pre)
    (args.out / 'config.json').write_text(json.dumps(asdict(config), indent=2))
    if not args.test_only and 'B' in after:
        assert np.array_equal(before, after['B']), 'Control weights unexpectedly changed'
    summary = {}
    for condition in after:
        summary[condition] = {}
        for phase in sorted({r['phase'] for r in rows}):
            selected = [r for r in rows if r['condition'] == condition and r['phase'] == phase
                        and not r['observation_tail']]
            last = {r['episode']: r for r in selected}
            reached = [r['time_to_reach_cigarette'] for r in last.values()
                       if r['time_to_reach_cigarette'] is not None]
            summary[condition][phase] = {
                'mean_distance_mm': float(np.mean([r['distance_to_cigarette'] for r in selected])),
                'mean_time_near_sec': float(np.mean([r['time_near_cigarette'] for r in last.values()])),
                'fraction_reached': len(reached) / len(last),
                'mean_reach_time_sec_among_reached': float(np.mean(reached)) if reached else None,
                'plasticity_decisions': sum(r['plasticity_applied'] for r in selected),
                'mean_path_length_mm': float(np.mean([r['path_length_mm'] for r in last.values()])),
            }
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    plots(rows, args.out, before, after, config)
    print(f'Results: {args.out}', flush=True)


if __name__ == '__main__':
    main()
