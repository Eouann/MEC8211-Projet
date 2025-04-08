#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 10:09:20 2025

@author: cedcrx
"""

import numpy as np
import matplotlib.pyplot as plt
import config

# Constantes
alpha = config.alpha
e = config.e
T_0 = config.T_0
T_x_0 = config.T_x_0
T_x_e = config.T_x_e
t_max = config.t_max

def crank_nicolson(N_spatial):
    delta_x = e / (N_spatial - 1)
    delta_t = 10# pour rester raisonnable (même si CN est stable)
    N_temporel = int(t_max / delta_t)
    
    r = alpha * delta_t / (delta_x ** 2)

    # Vecteurs et matrices
    T = np.ones(N_spatial) * T_0
    T[-1] = T_x_e

    T_record = np.zeros((N_temporel, N_spatial))
    T_record[0, :] = T

    # Matrices A et B tridiagonales
    A = np.zeros((N_spatial, N_spatial))
    B = np.zeros((N_spatial, N_spatial))
    
    for i in range(1, N_spatial-1):
        A[i, i-1] = -r/2
        A[i, i]   = 1 + r
        A[i, i+1] = -r/2

        B[i, i-1] = r/2
        B[i, i]   = 1 - r
        B[i, i+1] = r/2

    A[0,0] = A[-1,-1] = 1
    B[0,0] = B[-1,-1] = 1

    for n in range(0, N_temporel-1):
        b = B @ T
        # imposer les conditions de Dirichlet
        b[0] = T_x_0
        b[-1] = T_x_e
        
        T = np.linalg.solve(A, b)
        T_record[n+1, :] = T

    x_i = np.linspace(0, e, N_spatial)
    t_i = np.linspace(0, t_max, N_temporel)
    return T_record, x_i, t_i

# Choix des instants à tracer
temps_affiches = [0, 900, 1800, 2700, 3600]

# Résolution Crank-Nicolson
N_spatial = 100
T_i_n, x_i, t_i = crank_nicolson(N_spatial)

# Affichage
plt.figure()
for t in temps_affiches:
    idx = np.argmin(np.abs(t_i - t))
    plt.plot(x_i, T_i_n[idx], label=f't={t}s')

plt.title("Diffusion thermique - Méthode Crank-Nicolson")
plt.xlabel('Position x (m)')
plt.ylabel('Température T (K)')
plt.legend()
plt.grid(True)
plt.show()
