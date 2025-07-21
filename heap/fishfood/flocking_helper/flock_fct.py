"""
Modify Flockig algroithm for BlueSwarm

Algo from: Reza Olfati-Saber,"Flocking for Multi-Agent Dynamic Systems: Algorithms and Theory", IEEE TRANSACTIONS ON AUTOMATIC CONTROL, 
Vol. 51 (3), 3 Mar 2006

Created on July 02 2025
"""

import numpy as np

#%% Setup flocking hyperparameters
# ================================
a = 1
b = 5
c = np.divide(np.abs(a-b),np.sqrt(4*a*b)) 
eps = 1
h = 0.9
pi = np.pi
c1_a = 1
c2_a = 2*np.sqrt(2)
c1_b = 1
c2_b = 2*np.sqrt(3)
c1_g = 1
c2_g = 2*np.sqrt(1)

# d = 2                       # lattice scale (distance between a-agents)
# r = 1.2*d                   # interaction range of a-agents

def sigma_norm(z):   
    """
    define sigma_norm - differentiable everywhere
    Equation (8) in the paper.
    """ 
    norm_sig = (1/eps)*(np.sqrt(1+eps*np.linalg.norm(z)**2)-1)
    return norm_sig
    
def rho_h(z):
    """     
    A smooth bump function rho_h(z).
    Equation (10) in the paper.
    rho_h(z) = 
    1,                          if z ∈ [0,h]
    0.5(1 + cos(π(z-h)/(1-h))), if z ∈ [h,1]
    0,                          if z > 1
    """    
    if 0 <= z < h:
        rho_h = 1        
    elif h <= z < 1:
        rho_h = 0.5*(1+np.cos(pi*np.divide(z-h,1-h)))    
    else:
        rho_h = 0  
    return rho_h
    

def sigma_1(z):    
    """
    Defined after equation (15) in the paper.
    sigma_1(z) = z/√(1 + z^2)"""
    sigma_1 = np.divide(z,np.sqrt(1+z**2))    
    return sigma_1

def phi(z):    
    phi = 0.5*((a+b)*sigma_1(z+c)+(a-b))    
    return phi 


def phi_a(q_i, q_j, r_a, d_a): 
    z = sigma_norm(q_j-q_i)        
    phi_a = rho_h(z/r_a) * phi(z-d_a)    
    return phi_a
    
def n_ij(q_i, q_j):
    n_ij = np.divide(q_j-q_i,np.sqrt(1+eps*np.linalg.norm(q_j-q_i)**2))    
    return n_ij

def a_ij(q_i, q_j, r_a):        
    a_ij = rho_h(sigma_norm(q_j-q_i)/r_a)
    return a_ij

def b_ik(q_i, q_ik, d_b):        
    b_ik = rho_h(sigma_norm(q_ik-q_i)/d_b)
    return b_ik

def phi_b(q_i, q_ik, d_b): 
    z = sigma_norm(q_ik-q_i)        
    phi_b = rho_h(z/d_b) * (sigma_1(z-d_b)-1)    
    return phi_b