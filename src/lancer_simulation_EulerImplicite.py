"""
Fichier de lancement de simulation de diffusion thermique au travers d'un matériau isotrope avec la méthode
d'Euler implicite.
"""


# Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
import config
import Result_Comsol
from erreurs import ErreurL1, ErreurL2



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
T_900 = Result_Comsol.T_900
T_1800 = Result_Comsol.T_1800
T_2700 = Result_Comsol.T_2700
T_3600 = Result_Comsol.T_3600
x_diff = Result_Comsol.x_diff


# Regroupe les arrays dans une seule matrice
T_all = np.vstack([
    Result_Comsol.T_900,
    Result_Comsol.T_1800,
    Result_Comsol.T_2700,
    Result_Comsol.T_3600
])

# Calcul des températures pour N points
def Temperatures(N_spatial,N_temporel):
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

    # Condition de Robin en x = e (Différentiation d'ordre 1)
    matA[-1, -1] = 1+delta_x * h / (alpha*rho*cp)
    matA[-1, -2] = -1
    vectB[-1] = h * delta_x * T_x_inf / (alpha*rho*cp)

    # Algorithmes differences finies
    for i in range(1,N_spatial-1):
        matA[i,i-1] = -alpha*delta_t                # Coeff B devant T_i-1
        matA[i,i] = delta_x**2+2*alpha*delta_t      # Coeff A devant T_i
        matA[i,i+1] = -alpha*delta_t                # Coeff B devant T_i+1
    
    T_i_n[0] = T_i
    for i in range(1,N_temporel):
        for j in range(1, N_spatial - 1):
            vectB[j] = T_i[j]*delta_x**2

        # Résolution du système matriciel
        T_i = np.linalg.solve(matA, vectB)
        T_i_n[i] = T_i
    
    return T_i_n , x_i, t_i

"""
# Lancement de la simulation
N_spatial=100
N_temporel=100
T_i_n,x_i,t_i = Temperatures(N_spatial,N_temporel)

# Affichage des résultats
plt.plot(x_i,T_i_n[25],label='t=900s')
plt.plot(x_i,T_i_n[50],label='t=1800s')
plt.plot(x_i,T_i_n[75],label='t=2700s')
plt.plot(x_i,T_i_n[99],label='t=3600s')
plt.title("Simulation de diffusion thermique au travers d'un matériau isotrope")
plt.xlabel('Position x (m)')
plt.ylabel('Température T (K)')
plt.legend()
plt.show()
"""

# Choix des paramètres de discrétisation
N_spatial = 16       # Doit être > len(x_points) pour interpolation précise
N_temporel = 100  # Doit être > 3600 pour respecter ta condition

def Result(N_spatial,N_temporel):
    # Appel de la fonction de simulation
    T_i_n, x_sim, t_sim = Temperatures(N_spatial, N_temporel)
    
    # Points où on veut extraire les températures (fixés par l'utilisateur)
    x_points = np.array([
        0.0, 0.006666666666666667, 0.013333333333333334, 0.020000000000000004,
        0.026666666666666672, 0.03333333333333335, 0.040000000000000015,
        0.04666666666666669, 0.05333333333333336, 0.06000000000000003,
        0.06666666666666671, 0.07333333333333339, 0.08000000000000007,
        0.08666666666666674, 0.09333333333333342, 0.1
    ])
    
    # Interpolation linéaire pour obtenir les températures aux points spécifiés
    T_interpolated = np.zeros((N_temporel, len(x_points)))
    
    for n in range(N_temporel):
        T_interpolated[n] = np.interp(x_points, x_sim, T_i_n[n])
        
    # Temps cibles en secondes
    target_times = [900, 1800, 2700, 3600]
    
    # Trouver les indices correspondants dans t_sim
    target_indices = [np.argmin(np.abs(t_sim - t)) for t in target_times]
    
    # Créer la matrice [4, 16]
    T_selected = T_interpolated[target_indices, :]

    return(T_selected)
    

def erreur_spatial(N_temporel, T_all):
    N_Spatiaux = [100,200,400]
    E = []
    delta_x = []
    for N_Sp in N_Spatiaux:
        A = Result(N_Sp,N_temporel)
        B = ErreurL1(T_all,A,N_Sp,N_temporel)
        E.append(B)
        delta_x.append(e/N_Sp)
    return(E,delta_x)

def erreur_temporelle(N_spatial, T_all):
    N_temporels = [200,400,800]
    E = []
    delta_t = []
    for N_Tp in N_temporels:
        A = Result(N_spatial,N_Tp)
        B = ErreurL1(T_all,A,N_spatial,N_Tp)
        E.append(B)
        delta_t.append(t_max/N_Tp)
    return(E,delta_t)

# Tracer l'ordre de convergence spatial
def plot_erreur_spatial(N_temporel, T_all):
    E, delta_x = erreur_spatial(N_temporel, T_all)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(delta_x, E, marker='o', label='Erreur spatiale')
    plt.xlabel('Pas de discrétisation (delta_x)')
    plt.ylabel('Erreur L1')
    plt.title('Ordre de convergence spatial')
    plt.grid(True)
    plt.legend()
    plt.show()

# Tracer l'ordre de convergence temporel
def plot_erreur_temporelle(N_spatial, T_all):
    E, delta_t = erreur_temporelle(N_spatial, T_all)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(delta_t, E, marker='o', label='Erreur temporelle')
    plt.xlabel('Pas de discrétisation (delta_t)')
    plt.ylabel('Erreur L1')
    plt.title('Ordre de convergence temporel')
    plt.grid(True)
    plt.legend()
    plt.show()
    

def calculer_ordre_convergence(E, delta_x):
    # Prenons deux points pour calculer l'ordre de convergence
    delta_x1, delta_x2 = delta_x[0], delta_x[1]
    E1, E2 = E[0], E[1]
    
    # Calcul de l'ordre de convergence
    p = np.log(E1 / E2) / np.log(delta_x1 / delta_x2)
    return p

# Exemple d'appel pour l'ordre de convergence spatial
E_spatial, delta_x = erreur_spatial(1000, T_all)  # On utilise 100 comme exemple pour N_temporel
ordre_spatial = calculer_ordre_convergence(E_spatial, delta_x)
print(f"Ordre de convergence spatial : {ordre_spatial}")

def calculer_ordre_convergence_temporel(E, delta_t):
    # Prenons deux points pour calculer l'ordre de convergence
    delta_t1, delta_t2 = delta_t[0], delta_t[1]
    E1, E2 = E[0], E[1]
    
    # Calcul de l'ordre de convergence
    p = np.log(E1 / E2) / np.log(delta_t1 / delta_t2)
    return p

# Exemple d'appel pour l'ordre de convergence temporel
E_temporel, delta_t = erreur_temporelle(16, T_all)  # On utilise 16 comme exemple pour N_spatial
ordre_temporel = calculer_ordre_convergence_temporel(E_temporel, delta_t)
print(f"Ordre de convergence temporel : {ordre_temporel}")


# Exemple d'utilisation
N_temporel = 100  # Exemple de paramètre temporel
plot_erreur_spatial(N_temporel, T_all)

N_spatial = 100  # Exemple de paramètre spatial
plot_erreur_temporelle(N_spatial, T_all)


        
        







