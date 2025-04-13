import numpy as np
import matplotlib.pyplot as plt
import config

# Définition des constantes
alpha=config.alpha
e=config.e
cp=config.cp
rho=config.rho
T_0=config.T_0
T_x_0=config.T_x_0
T_x_inf=config.T_x_inf
t_max=config.t_max
h=config.h
k = alpha * rho * cp  # Conductivité thermique

#N_temporel doit etre superieur a 3600
def Temperatures_Explicite(N_spatial,N_temporel):
    delta_x = e / (N_spatial - 1)
    delta_t = t_max/(N_temporel-1)
    

    x_i = np.linspace(0, e, N_spatial)
    t_i = np.linspace(0, t_max, N_temporel)

    T_i = np.ones(N_spatial) * T_0
    T_i_n = np.zeros((N_temporel, N_spatial))
    T_i_n[0] = T_i.copy()

    lambda_ = alpha * delta_t / delta_x**2

    for n in range(N_temporel - 1):
        T_next = T_i.copy()
        
        # Intérieur du domaine
        for i in range(1, N_spatial - 1):
            T_next[i] = T_i[i] + lambda_ * (T_i[i-1] - 2*T_i[i] + T_i[i+1])

        # Bord gauche : Dirichlet
        T_next[0] = T_x_0

        # Bord droit : Robin (flux -> interpolation directe)
        T_next[-1] = (k * T_i[-2]/delta_x + h * T_x_inf) / (k/delta_x + h)
        T_i = T_next.copy()
        T_i_n[n+1] = T_i

    return T_i_n, x_i, t_i


# --- Simulation ---

temps_affiches = [ 900, 1800, 2700, 3600]

# Lancement de la simulation
N_spatial = 100
N_temporel = 3600
T_i_n, x_i, t_i = Temperatures_Explicite(N_spatial,N_temporel)

# Affichage des résultats
plt.plot(x_i, T_i_n[10], label='t=0s')
plt.plot(x_i, T_i_n[125], label='t=900s')
plt.plot(x_i, T_i_n[250], label='t=1800s')
plt.plot(x_i, T_i_n[375], label='t=2700s')
plt.plot(x_i, T_i_n[499], label='t=3600s')
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