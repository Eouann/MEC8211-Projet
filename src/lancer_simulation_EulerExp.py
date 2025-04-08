

import numpy as np
import matplotlib.pyplot as plt
import config

# Définition des constantes
alpha = config.alpha
e = config.e
T_0 = config.T_0
T_x_0 = config.T_x_0
T_x_e = config.T_x_e
t_max = config.t_max

# Calcul des températures pour N points

def Concentrations(N_spatial):
    """Méthode d'ordre 2 en espace et Euler explicite en temps"""
    
    delta_x = e / (N_spatial - 1)
    #delta_t = 1.71
    delta_t = 0.4 * delta_x**2 / alpha  # Condition CFL pour stabilité
    N_temporel = 3600/delta_t
    N_temporel = int(N_temporel)
    
    T_i = np.ones(N_spatial) * T_0  # Température initiale

    T_i_n = np.zeros((N_temporel, N_spatial))  # Stockage des températures
    
    x_i = np.linspace(0,e,N_spatial)
    t_i = np.linspace(0, t_max, N_temporel)
    
    T_i_n[0,:] = T_i
 
    # Simulation temporelle
    for n in range(0, N_temporel - 1):
        T_next = T_i.copy()
        
        # Schéma d'ordre 2 en espace
        for i in range(1, N_spatial - 1):
            T_next[i] = T_i[i] + alpha * delta_t / delta_x**2 * (T_i[i-1] - 2*T_i[i] + T_i[i+1])
        
        # Conditions aux bords (Dirichlet)
        T_next[0] = T_x_0
        T_next[-1] = T_x_e
        
        T_i = T_next.copy()
        T_i_n[n+1] = T_i
    
    return T_i_n, x_i, t_i

# Temps à afficher (en secondes)
temps_affiches = [0, 900, 1800, 2700, 3600]

# Lancement de la simulation
N_spatial = 100
T_i_n, x_i, t_i = Concentrations(N_spatial)

# Tracé
plt.figure()
for t in temps_affiches:
    idx = np.argmin(np.abs(t_i - t))  # Trouver l'index le plus proche du temps voulu
    plt.plot(x_i, T_i_n[idx], label=f't={t}s')

plt.title("Simulation de diffusion thermique au travers d'un matériau isotrope")
plt.xlabel('Position x (m)')
plt.ylabel('Température T (K)')
plt.legend()
plt.show()

