import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Find most recent data file in ./logfiles
files = glob.glob('logfiles/*_data.txt')
if not files:
    raise SystemExit('No data files found in ./logfiles')

latest = max(files, key=os.path.getmtime)
print('Loading', latest)

data = np.loadtxt(latest, delimiter=',')
# leader is fish 0: x=data[:,0], y=data[:,1]
if data.shape[1] < 2:
    raise SystemExit('Unexpected data shape: {}'.format(data.shape))

x = data[:,0]
y = data[:,1]

plt.figure(figsize=(6,6))
plt.plot(x, y, '-k', linewidth=1)
plt.scatter([x[0]],[y[0]], c='green', label='start')
plt.scatter([x[-1]],[y[-1]], c='red', label='end')
plt.gca().set_aspect('equal', 'box')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Leader XY trajectory — {}'.format(os.path.basename(latest)))
plt.legend()

os.makedirs('logfigs', exist_ok=True)
outfile = os.path.join('logfigs','leader_traj_exp9.png')
plt.savefig(outfile, dpi=200)
print('Saved', outfile)
