import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

# Find most recent data file in ./logfiles
files = glob.glob('logfiles/*_data.txt')
if not files:
    raise SystemExit('No data files found in ./logfiles')
latest = max(files, key=os.path.getmtime)
meta_file = latest.replace('_data.txt','_meta.txt')
print('Loading', latest)

data = np.loadtxt(latest, delimiter=',')
with open(meta_file,'r') as f:
    meta = json.load(f)

clock_freq = meta.get('Clock frequency [Hz]', meta.get('clock_freq', 5))
# time vector
timesteps = data.shape[0]
t = np.arange(timesteps)/clock_freq

# leader actual
x = data[:,0]
y = data[:,1]

# commanded path parameters (should match exp9_traj8)
a = 1000.0
period = 20.0
center_x = 1000.0
center_y = -2800.0

s = 2 * np.pi * t / period
x_cmd = center_x + a * np.sin(s)
y_cmd = center_y + (a * np.sin(2*s)) / 2.0

plt.figure(figsize=(6,6))
plt.plot(x_cmd, y_cmd, '--', c='tab:orange', label='commanded (figure-8)')
plt.plot(x, y, '-k', linewidth=1, label='actual')
plt.scatter([x[0]],[y[0]], c='green', label='start')
plt.scatter([x[-1]],[y[-1]], c='red', label='end')
plt.gca().set_aspect('equal', 'box')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Leader actual vs commanded — {}'.format(os.path.basename(latest)))
plt.legend()

os.makedirs('logfigs', exist_ok=True)
outfile = os.path.join('logfigs','leader_vs_commanded_exp9.png')
plt.savefig(outfile, dpi=200)
print('Saved', outfile)
