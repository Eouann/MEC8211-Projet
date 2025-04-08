import numpy as np
import matplotlib.pyplot as plt
import config


# Définition des constantes
alpha=config.alpha
e=config.e
T_0=config.T_0
T_x_0=config.T_x_0
T_x_e=config.T_x_e
t_max=config.t_max

def Concentrations(N_spatial, N_temporel):
    """Calcul des températures en différences finies :
       - Ordre 4 en espace
       - Euler implicite en temps (ordre 1)
    """
    T_i = np.ones(N_spatial)*T_0
    T_i_n = np.zeros((N_temporel,N_spatial))
    matA = np.zeros((N_spatial,N_spatial))
    vectB = np.zeros(N_spatial)

    delta_x = e / (N_spatial - 1)
    delta_t = 1E-1
    x_i = np.linspace(0, e, N_spatial)
    t_i = np.linspace(0, t_max, N_temporel)
    coeff = alpha * delta_t / delta_x**2

    # Condition de Dirichlet aux bords
    matA[0, 0] = 1
    matA[-1, -1] = 1
    vectB[0] = T_x_0
    vectB[-1] = T_x_e

    # Schéma d’ordre 2 aux points 1 et N-2
    i = 1
    matA[i, i-1] = -coeff
    matA[i, i]   = 1 + 2*coeff
    matA[i, i+1] = -coeff

    i = N_spatial - 2
    matA[i, i-1] = -coeff
    matA[i, i]   = 1 + 2*coeff
    matA[i, i+1] = -coeff

    # Schéma d’ordre 4 pour les points intérieurs restants
    for i in range(2, N_spatial - 2):
        matA[i, i-2] = -coeff / 12
        matA[i, i-1] = 4 * coeff / 3
        matA[i, i]   = 1 + 5 * coeff / 2
        matA[i, i+1] = 4 * coeff / 3
        matA[i, i+2] = -coeff / 12

    # Boucle temporelle (Euler implicite)
    for n in range(len(t_i)):
        # On remplit le vecteur B à chaque pas de temps
        for j in range(1, N_spatial - 1):
            vectB[j] = T_i[j]

        # Conditions de Dirichlet
        vectB[0] = T_x_0
        vectB[-1] = T_x_e

        T_i = np.linalg.solve(matA, vectB)
        T_i_n[n] = T_i

    return T_i_n, x_i, t_i

# Lancement de la simulation
N_spatial=500
N_temporel=100
T_i_n,x_i,t_i = Concentrations(N_spatial,N_temporel)


# Affichage des résultats
#plt.plot(x_i,T_i_n[0],label='t=0s')
plt.plot(x_i,T_i_n[25],label='t=900s')
#plt.plot(x_i,T_i_n[50],label='t=1800s')
#plt.plot(x_i,T_i_n[75],label='t=2700s')
#plt.plot(x_i,T_i_n[99],label='t=3600s')
plt.title("Simulation de diffusion thermique au travers d'un matériau isotrope")
plt.xlabel('Position x (m)')
plt.ylabel('Température T (K)')
plt.legend()
plt.show()
