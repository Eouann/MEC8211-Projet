"""
Fichier de lancement de simulation de diffusion thermique au travers d'un matériau isotrope avec la méthode
de Crank-Nickolson
"""

# Importation des bibliothèques
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

# Calcul des températures pour N points
def Temperature_Crank_Nickolson(N_spatial):
    delta_x = e / (N_spatial - 1)
    delta_t = 2  # Pas de discrétisation temporelle
    N_temporel = int(t_max / delta_t)

    r = alpha * delta_t / (delta_x ** 2)
    k = alpha * rho * cp  # Conductivité thermique effective

    # Vecteurs et matrices
    T = np.ones(N_spatial) * T_0
    T[0] = T_x_0

    T_record = np.zeros((N_temporel, N_spatial))
    T_record[0, :] = T

    A = np.zeros((N_spatial, N_spatial))
    B = np.zeros((N_spatial, N_spatial))

    # Construction des matrices A et B
    for i in range(1, N_spatial - 1):
        A[i, i - 1] = -r / 2
        A[i, i] = 1 + r
        A[i, i + 1] = -r / 2

        B[i, i - 1] = r / 2
        B[i, i] = 1 - r
        B[i, i + 1] = r / 2

    # Condition de Dirichlet à gauche (x=0)
    A[0, 0] = 1
    B[0, 0] = 1

    # Condition de Robin à droite (x=e)
    coef_robin = (k / delta_x) + h
    A[-1, -2] = -k / delta_x
    A[-1, -1] = coef_robin
    B[-1, -2] = k / delta_x
    B[-1, -1] = coef_robin

    for n in range(0, N_temporel - 1):
        b = B @ T

        # Imposer les conditions aux limites
        b[0] = T_x_0  # Dirichlet
        b[-1] = h * T_x_inf  # Robin (second membre de l'équation)

        T = np.linalg.solve(A, b)
        T_record[n + 1, :] = T

    x_i = np.linspace(0, e, N_spatial)
    t_i = np.linspace(0, t_max, N_temporel)
    return T_record, x_i, t_i

# Choix des instants à tracer
temps_affiches = [900, 1800, 2700, 3600]

# Résolution Crank-Nicolson
N_spatial = 100
T_i_n, x_i, t_i = Temperature_Crank_Nickolson(N_spatial)

# Affichage
plt.figure()
for t in temps_affiches:
    idx = np.argmin(np.abs(t_i - t))
    plt.plot(x_i, T_i_n[idx], label=f't={t}s')

plt.title("Diffusion thermique - Méthode Crank-Nicolson")
plt.xlabel('Position x (m)')
plt.ylabel('Température T (K)')
plt.legend()
plt.show()
