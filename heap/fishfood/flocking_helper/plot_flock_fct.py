
from flock_fct import *
import numpy as np
import matplotlib.pyplot as plt



def plot_sigma_1(z):
    fig, ax = plt.subplots(figsize=(8, 6))
    # plt.figure(figsize=(8, 6))
    ax.plot(z, sigma_1(z), label='sigma_1(z)', color='blue')
    ax.set_title('Plot of sigma_1(z)')
    ax.set_xlabel('z')
    ax.set_ylabel('sigma_1(z)')
    ax.grid(True)


def plot_rho_h(z):
    rho_h_value =  np.array([rho_h(zi) for zi in z]) # This is just an example

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(z, rho_h_value, label='rho_h(z)', color='green')
    ax.set_title('Plot of rho_h(z)')
    ax.set_xlabel('z')
    ax.set_ylabel('rho_h(z)')
    ax.grid(True)
    # ax.set_xticks(np.arange(0, 10, 1))

def plot_phi(z):
    phi_value = np.array([phi(zi) for zi in z])  # This is just an example

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(z, phi_value, label='phi(z)', color='red')
    ax.set_title('Plot of phi(z)')
    ax.set_xlabel('z')
    ax.set_ylabel('phi(z)')
    ax.grid(True)



def plot_phi_a(z, d, r):

    r_a = sigma_norm(r)
    d_a = sigma_norm(d)
    print(" lattice distance is ", d)
    print(" interaction range is ", r)

    fig, ax = plt.subplots(figsize=(8, 6))

    phi_a_values = np.array([phi_a(0, zi, r_a, d_a) for zi in z])
    ax.plot(z, phi_a_values, label=f'phi_a(z), h={h}')
    ax.legend()
    ax.set_title('Plot of phi_a(q_i, q_j)')
    ax.set_xlabel('z')
    ax.set_ylabel('phi_a')
    ax.grid(True)

    ax.axvline(x=d, color='magenta', linestyle='--', label=f'd = {d}')
    ax.axvline(x=r, color='cyan', linestyle='--', label=f'r = {r}')
    ax.legend()
    # ax.set_xticks(np.arange(0, z[-1], 50))



# Create z array (example range from 0 to 10)
z = np.linspace(0, 1500, 100)

d = 2 * 130 # lattice scale (distance between a-agents)
r = 5 * d # interaction range of a-agents

plot_sigma_1(z)

plot_rho_h(z)
plot_phi(z)
plot_phi_a(z, d, r)

# Show the plots
plt.show()