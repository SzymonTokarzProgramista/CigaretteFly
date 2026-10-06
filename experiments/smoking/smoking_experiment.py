"""Contact-conditioned spiking connectome in a FlyGym/MuJoCo 3D body."""
import argparse
import csv
from dataclasses import asdict, replace
import importlib.util
import hashlib
import json
from pathlib import Path
import random

from config import Config, nicotine_to_pam

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


def episode(brain, config, condition, phase, episode_id, out, video, phase_index):
    import numpy as np
    from flygym import Fly, Camera
    from flygym.arena import FlatTerrain
    from flygym.examples.locomotion import HybridTurningController

    # A/B use identical body and exploration seeds, independent from brain RNG.
    seed = config.seed + phase_index
    rng = np.random.default_rng(seed)
    arena = FlatTerrain()
    for name, source, color in [('cigarette', config.cigarette_position, (0.9, 0.65, 0.2, 1)),
                                ('control', config.control_position, (0.3, 0.5, 0.8, 1))]:
        marker = arena.root_element.worldbody.add('body', name=name, pos=(*source, 0.25))
        marker.add('geom', type='cylinder', size=(0.2, 0.8),
                   quat=(0.7071068, 0, 0.7071068, 0),
                   rgba=color, contype=0, conaffinity=0)
    contacts = [f'{leg}{segment}' for leg in ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
                for segment in ['Tibia', 'Tarsus1', 'Tarsus2', 'Tarsus3', 'Tarsus4', 'Tarsus5']]
    fly = Fly(enable_adhesion=True, contact_sensor_placements=contacts,
              spawn_pos=config.spawn_position)
    cameras = []
    if video:
        cameras = [Camera(attachment_point=arena.root_element.worldbody,
                          camera_name='birdeye_cam', camera_parameters={
                              'mode': 'fixed', 'pos': (9, -18, 28), 'euler': (0.55, 0, 0), 'fovy': 50},
                          window_size=(800, 608), play_speed=0.5)]
    sim = HybridTurningController(fly=fly, cameras=cameras, arena=arena,
                                  timestep=config.timestep, seed=seed)
    nicotine = tolerance = near_time = 0.0
    reach_time = None
    rows = []
    obs, _ = sim.reset(seed)
    initial_weights = np.asarray(brain.syn.w[brain.plastic_pos]).copy()

    def odor(position, source):
        return float(np.exp(-np.sum((position - source) ** 2) / (2 * config.odor_sigma ** 2)))

    def sniff(position, pam):
        rates = {i: odor(position, config.cigarette_position) * config.orn_hz for i in brain.A}
        rates.update({i: odor(position, config.control_position) * config.orn_hz for i in brain.B})
        rates.update({int(i): pam for i in brain.g['pam']})
        spikes = brain._episode(rates, config.sniff_ms)
        return spikes, float(spikes[brain.g['mbon']].sum())

    try:
        for decision in range(config.decisions):
            pos = np.array(obs['fly'][0][:2])
            distance = float(np.linalg.norm(pos - config.cigarette_position))
            contact = distance < config.smoking_radius
            if contact and reach_time is None:
                reach_time = decision * config.decision_sec
            if contact:
                near_time += config.decision_sec
            if phase == 'TRAIN':
                nicotine *= config.nicotine_decay
                if contact:
                    nicotine += config.nicotine_intake * config.decision_sec
                    tolerance += config.tolerance_increase * config.decision_sec
            pam = nicotine_to_pam(nicotine, tolerance, config) if phase == 'TRAIN' and condition == 'A' else 0.0
            head = np.asarray(obs['fly_orientation'][:2])
            head = head / max(np.linalg.norm(head), 1e-9)
            lateral = np.array([-head[1], head[0]])
            left, v_left = sniff(pos + 1.5 * head + 1.2 * lateral, pam)
            right, v_right = sniff(pos + 1.5 * head - 1.2 * lateral, pam)
            counts = left + right
            pam_activity = float(counts[brain.g['pam']].mean() / (2 * config.sniff_ms / 1000))
            applied = False
            if phase == 'TRAIN' and condition == 'A' and pam > 0 and pam_activity >= config.pam_gate_hz:
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
            action = np.array([1 + delta, 1 - delta])
            for _ in range(round(config.decision_sec / config.timestep)):
                obs, _, terminated, truncated, _ = sim.step(action)
                if cameras:
                    sim.render()
                if terminated or truncated:
                    raise RuntimeError('Body simulation terminated before the scheduled episode end')
            new_distance = float(np.linalg.norm(np.array(obs['fly'][0][:2]) - config.cigarette_position))
            rows.append(dict(condition=condition, episode=episode_id, phase=phase,
                             time=decision * config.decision_sec, x=float(pos[0]), y=float(pos[1]),
                             distance_to_cigarette=distance, time_near_cigarette=near_time,
                             time_to_reach_cigarette=reach_time, nicotine=nicotine, tolerance=tolerance,
                             smoking=contact and phase == 'TRAIN', pam_rate=pam,
                             selected_PAM_activity=pam_activity,
                             selected_MBON_activity=float(counts[brain.g['mbon']].mean() / (2 * config.sniff_ms / 1000)),
                             mean_KC_MBON_weight=float(np.asarray(brain.syn.w[brain.plastic_pos]).mean()),
                             approach_velocity=(distance - new_distance) / config.decision_sec,
                             plasticity_applied=applied))
        if phase != 'TRAIN':
            assert np.array_equal(initial_weights, np.asarray(brain.syn.w[brain.plastic_pos]))
        if cameras:
            cameras[0].save_video(out / f'{condition}_{phase}_{episode_id}.mp4')
    finally:
        sim.close()
    return rows


def plots(rows, out, before, after, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for condition in ['A', 'B']:
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
    ap.add_argument('--quick', action='store_true', help='2 decisions per phase, validates execution only')
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--test-only', action='store_true')
    ap.add_argument('--weights-dir', type=Path, help='directory containing A_weights.npy, B_weights.npy')
    ap.add_argument('--contact-start', action='store_true', help='start beside the object, for contact pathway diagnostics')
    ap.add_argument('--decisions', type=int)
    ap.add_argument('--train-episodes', type=int)
    ap.add_argument('--test-episodes', type=int)
    args = ap.parse_args()
    if args.test_only and not args.weights_dir:
        ap.error('--test-only requires --weights-dir from a completed training run')
    for path in [args.annotations, args.model_dir / 'model.py', args.model_dir / 'Connectivity_783.parquet']:
        if not path.is_file():
            ap.error(f'Missing input: {path}. See experiments/smoking/README.md')
    import numpy as np
    config = replace(Config(), seed=args.seed)
    if args.quick:
        config = replace(config, train_episodes=1, test_episodes=1, decisions=2)
    for name in ['decisions', 'train_episodes', 'test_episodes']:
        value = getattr(args, name)
        if value is not None:
            if value < 1:
                ap.error(f'{name} must be positive')
            config = replace(config, **{name: value})
    if args.contact_start:
        config = replace(config, spawn_position=(14.0, 7.0, 0.2))
    random.seed(config.seed); np.random.seed(config.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    nav_path = ROOT / 'vendor/fly-api/experiments/navigation/nav_demo.py'
    annotations = prepare_annotations(args.annotations, args.out)
    nav = load_navigation(nav_path, annotations.resolve())
    import brian2
    brian2.prefs.codegen.runtime.cython.cache_dir = str(ROOT / '.cache/brian2')
    brain = nav.Brain(str(args.model_dir.resolve()), seed=config.seed,
                      gain=config.kcmbon_gain, orn_hz=config.orn_hz)
    before = brain.w_naive.copy()
    digest = hashlib.sha256()
    for path in [args.annotations, args.model_dir / 'Completeness_783.csv', args.model_dir / 'Connectivity_783.parquet']:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    digest.update(json.dumps({'seed': config.seed, 'gain': config.kcmbon_gain}, sort_keys=True).encode())
    fingerprint = digest.hexdigest()
    if args.test_only:
        saved = json.loads((args.weights_dir / 'provenance.json').read_text())
        if saved['circuit_fingerprint'] != fingerprint:
            raise ValueError('Checkpoint circuit differs: use the same seed, gain, model and annotations')
    (args.out / 'provenance.json').write_text(json.dumps({
        'circuit_fingerprint': fingerprint,
        'fly_api_commit': 'a6ad07a810b1a43cd0356149c07b32105eb46d2a',
        'brain_model_commit': '91bdd1e7dcf193f3e7ca5a8933497fcef63b7960',
        'annotation_release': 'v2.1.0 (default)',
        'mode': 'test_only' if args.test_only else 'pre_train_test',
    }, indent=2))
    rows, after = [], {}
    for condition in ['A', 'B']:
        # Restore time, neuronal state, monitors and random streams for matched controls.
        if condition == 'A':
            brain.net.store('initial')
        else:
            brain.net.restore('initial', restore_random_state=True)
            brain.prev = np.zeros_like(brain.prev)
        if args.test_only:
            weights = np.load(args.weights_dir / f'{condition}_weights.npy')
            if weights.shape != before.shape:
                raise ValueError('Checkpoint shape differs from this circuit')
            brain.syn.w[brain.plastic_pos] = weights * brain.br['volt']
        phases = ['TEST'] if args.test_only else ['PRE', 'TRAIN', 'TEST']
        episode_id = 0
        for phase in phases:
            n = config.train_episodes if phase == 'TRAIN' else config.test_episodes
            for phase_index in range(n):
                print(f'[{condition}] {phase} episode={episode_id}', flush=True)
                rows.extend(episode(brain, config, condition, phase, episode_id, args.out, not args.no_video, phase_index))
                with (args.out / 'smoking_experiment.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
                episode_id += 1
        after[condition] = np.asarray(brain.syn.w[brain.plastic_pos]).copy()
        np.save(args.out / f'{condition}_weights.npy', after[condition])
    np.save(args.out / 'initial_weights.npy', before)
    np.save(args.out / 'plastic_pre.npy', brain.plastic_pre)
    (args.out / 'config.json').write_text(json.dumps(asdict(config), indent=2))
    if not args.test_only:
        assert np.array_equal(before, after['B']), 'Control weights unexpectedly changed'
    summary = {}
    for condition in ['A', 'B']:
        summary[condition] = {}
        for phase in sorted({r['phase'] for r in rows}):
            selected = [r for r in rows if r['condition'] == condition and r['phase'] == phase]
            last = {r['episode']: r for r in selected}
            reached = [r['time_to_reach_cigarette'] for r in last.values()
                       if r['time_to_reach_cigarette'] is not None]
            summary[condition][phase] = {
                'mean_distance_mm': float(np.mean([r['distance_to_cigarette'] for r in selected])),
                'mean_time_near_sec': float(np.mean([r['time_near_cigarette'] for r in last.values()])),
                'fraction_reached': len(reached) / len(last),
                'mean_reach_time_sec_among_reached': float(np.mean(reached)) if reached else None,
                'plasticity_decisions': sum(r['plasticity_applied'] for r in selected),
            }
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    plots(rows, args.out, before, after, config)
    print(f'Results: {args.out}', flush=True)


if __name__ == '__main__':
    main()
