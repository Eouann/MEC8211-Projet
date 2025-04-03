"""
Fichier de lancement de simulation de diffusion thermique au travers d'un matériau isotrope avec la méthode
d'Euler implicite.
"""


# Importation des bibliothèques
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


# Calcul des températures pour N points
def Concentrations(N_spatial,N_temporel):
    """Fonction de calcul des N températures en différences finies"""
    T_i = np.ones(N_spatial)*T_0                # Vecteur des N températures numériques calculées T_i
    T_i_n = np.zeros((N_temporel,N_spatial))    # Matrice des N températures numériques calculées T_i pour chaque itération
    matA = np.zeros((N_spatial,N_spatial))      # Matrice A pour la résolution du système matriciel
    vectB = np.zeros(N_spatial)                 # Vecteur B pour la résolution du système matriciel

    delta_x=e/(N_spatial-1)                     # Pas de discrétisation
    x_i=np.linspace(0, e, N_spatial)            # Vecteur des N points r_i également espacées

    delta_t=t_max/N_temporel                    # Pas de discrétisation temporelle
    t_i=np.linspace(0, t_max, N_temporel)       # Vecteur des N points t_i également espacées

    # Condition de Dirichlet en x = 0
    matA[0,0] = 1
    vectB[0] = T_x_0

    # Condition de Dirichlet en x = e
    matA[-1,-1] = 1
    vectB[-1] = T_x_e


    # Algorithmes differences finies
    for i in range(1,N_spatial-1):
        matA[i,i-1] = -alpha*delta_t                # Coeff B devant T_i-1
        matA[i,i] = delta_x**2+2*alpha*delta_t      # Coeff A devant T_i
        matA[i,i+1] = -alpha*delta_t                # Coeff B devant T_i+1
    
    for i in range(len(t_i)):
        for j in range(1, N_spatial - 1):
            vectB[j] = T_i[j]*delta_x**2

        # Résolution du système matriciel
        T_i = np.linalg.solve(matA, vectB)
        T_i_n[i] = T_i

    return T_i_n,x_i,t_i


# Lancement de la simulation
N_spatial=100
N_temporel=100
T_i_n,x_i,t_i = Concentrations(N_spatial,N_temporel)


# Affichage des résultats
plt.plot(x_i,T_i_n[0],label='t=0s')
plt.plot(x_i,T_i_n[25],label='t=900s')
plt.plot(x_i,T_i_n[50],label='t=1800s')
plt.plot(x_i,T_i_n[75],label='t=2700s')
plt.plot(x_i,T_i_n[99],label='t=3600s')
plt.title("Simulation de diffusion thermique au travers d'un matériau isotrope")
plt.xlabel('Position x (m)')
plt.ylabel('Température T (K)')
plt.legend()
plt.show()
