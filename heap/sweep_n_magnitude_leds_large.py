"""Sweeps n_magnitude_leds across the exp8 (8-fish, "n8") experiment.

For each n_magnitude_leds value, sets the N_MAGNITUDE_LEDS env var (read by
simulation_noise.py) and runs
    python simulation_noise.py exp8
as a subprocess, then records which logfile prefix that run produced so
plot_sweep_n_magnitude_leds_large.py can find and compare them afterwards.

Usage:
    python sweep_n_magnitude_leds_large.py
"""
import json
import os
import re
import subprocess

FORMATIONS = ['n8']
# N_MAGNITUDE_LEDS_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5]

N_MAGNITUDE_LEDS_VALUES = [0.0, 0.1, 0.5]
EXPERIMENT = 'exp8_n22'
MANIFEST_PATH = './logfiles/sweep_n_magnitude_leds_large_manifest.json'

manifest = {}
for formation in FORMATIONS:
    manifest[formation] = {}
    for n_magnitude_leds in N_MAGNITUDE_LEDS_VALUES:
        print('\n=== Running {} ({}) with n_magnitude_leds={} ===\n'.format(EXPERIMENT, formation, n_magnitude_leds))
        env = os.environ.copy()
        env['FORMATION_CONFIG'] = formation
        env['N_MAGNITUDE_LEDS'] = str(n_magnitude_leds)
        env['MPLBACKEND'] = 'Agg'  # save plots without popping up a blocking window
        result = subprocess.run(['python', 'simulation_noise.py', EXPERIMENT], env=env, check=True,
                                 capture_output=True, text=True)
        print(result.stdout)
        match = re.search(r'\./logfiles/(\d{6}_\d{6})_data\.txt', result.stdout)
        if not match:
            raise RuntimeError('Could not find logfile prefix in simulation output for formation={}, n_magnitude_leds={}'.format(formation, n_magnitude_leds))
        manifest[formation][str(n_magnitude_leds)] = match.group(1)

with open(MANIFEST_PATH, 'w') as f:
    json.dump(manifest, f, indent=2)
print('\nWrote manifest to {}: {}'.format(MANIFEST_PATH, manifest))

os.system('python plot_sweep_n_magnitude_leds_large.py')
