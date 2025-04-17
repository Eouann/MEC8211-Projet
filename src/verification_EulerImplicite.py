"""
Fichier de vérification du modèle numérique de diffusion thermique par MMS
"""


# Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.stats import linregress
import config
import erreurs


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


# Définition des variables symboliques
t, x = sp.symbols('t x')

# Solution manufacturée
T_MMS = T_0+T_x_0*sp.exp(-h*x/(alpha*rho*cp))*(1-sp.exp(-t/t_max))

# Calcul des dérivées
T_t = sp.diff(T_MMS, t)
T_x = sp.diff(T_MMS, x)
T_xx = sp.diff(T_MMS, x, x)

# Calcul du terme source S(t,x)
source = T_t - alpha * T_xx

# Conversion en fonctions Python
f_T_MMS = sp.lambdify([t, x], T_MMS, "numpy")
f_source = sp.lambdify([t, x], source, "numpy")

# Calcul des températures pour N points
def Temperatures(N_spatial, N_temporel, T_x_0=T_x_0, T_x_inf=T_x_inf, alpha=alpha, rho=rho, cp=cp, h=h):
    """Fonction de calcul des températures en différences finies avec terme source"""
    T_i = np.ones(N_spatial) * T_0
    T_i_n = np.zeros((N_temporel, N_spatial))
    matA = np.zeros((N_spatial, N_spatial))
    vectB = np.zeros(N_spatial)

    delta_x = e / (N_spatial - 1)
    x_i = np.linspace(0, e, N_spatial)

    delta_t = t_max / N_temporel
    t_i = np.linspace(0, t_max, N_temporel)

    # Condition de Dirichlet en x = 0
    matA[0, 0] = 1

    # Condition de Robin en x = e (ordre 1)
    matA[-1, -1] = 1 + delta_x * h / (alpha * rho * cp)
    matA[-1, -2] = -1
    vectB[-1] = h * delta_x * T_x_inf / (alpha * rho * cp)

    # Remplissage de la matrice A pour les points internes
    for i in range(1, N_spatial - 1):
        matA[i, i - 1] = -alpha * delta_t
        matA[i, i] = delta_x ** 2 + 2 * alpha * delta_t
        matA[i, i + 1] = -alpha * delta_t

    T_i_n[0] = T_i

    for n in range(1, N_temporel):
        # Mise à jour de vectB pour la condition de Dirichlet en x = 0
        vectB[0] = T_0 + T_x_0 * (1 - np.exp(-t_i[n] / t_max))
        
        for j in range(1, N_spatial - 1):
            S = f_source(t_i[n], x_i[j])  # Terme source
            vectB[j] = T_i[j] * delta_x ** 2 + delta_t * delta_x ** 2 * S  # Ajout du terme source
        
        T_i = np.linalg.solve(matA, vectB)
        T_i_n[n] = T_i

    return T_i_n, x_i, t_i


def convergence_et_erreur(N_spatial_list, N_temporel):
    """Calcule les erreurs L1, L2, L_infini et l'ordre de convergence"""
    erreurs_L1 = []
    erreurs_L2 = []
    erreurs_Linf = []
    dx_list = []

    for N in N_spatial_list:
        T_num, x_i, t_i = Temperatures(N, N_temporel)
        T_exact = f_T_MMS(t_i[-1], x_i)

        erreur = np.abs(T_num[-1] - T_exact)

        dx = e / (N - 1)
        dx_list.append(dx)

        erreur_L1 = np.sum(erreur) * dx
        erreur_L2 = np.sqrt(np.sum(erreur**2) * dx)
        erreur_Linf = np.max(erreur)

        erreurs_L1.append(erreur_L1)
        erreurs_L2.append(erreur_L2)
        erreurs_Linf.append(erreur_Linf)

    # Régression linéaire pour ordre de convergence
    slope_L1, _, _, _, _ = linregress(np.log(dx_list), np.log(erreurs_L1))
    slope_L2, _, _, _, _ = linregress(np.log(dx_list), np.log(erreurs_L2))
    slope_Linf, _, _, _, _ = linregress(np.log(dx_list), np.log(erreurs_Linf))

    # Affichage
    plt.figure(figsize=(10,5))
    
    plt.loglog(dx_list, erreurs_L1, 'o-', label=f"L1 (ordre = {abs(slope_L1):.2f})")
    plt.loglog(dx_list, erreurs_L2, 's-', label=f"L2 (ordre = {abs(slope_L2):.2f})")
    plt.loglog(dx_list, erreurs_Linf, 'd-', label=f"L∞ (ordre = {abs(slope_Linf):.2f})")
    plt.xlabel('dx')
    plt.ylabel('Erreur')
    plt.title("Erreurs L1, L2 et L∞ spatiales")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.show()


    return erreurs_L1, erreurs_L2, erreurs_Linf, dx_list, slope_L1, slope_L2, slope_Linf

N_spatial_list = [10, 20, 40, 80, 160]
N_temporel = 100

erreurs_L1, erreurs_L2, erreurs_Linf, dx_list, ordre_L1, ordre_L2, ordre_Linf= convergence_et_erreur(N_spatial_list, N_temporel)

print(f"Ordre de convergence L1 : {abs(ordre_L1):.2f}")
print(f"Ordre de convergence L2 : {abs(ordre_L2):.2f}")
print(f"Ordre de convergence temporel Linf : {abs(ordre_Linf):.2f}")

def convergence_temporelle(N_temporel_list, N_spatial):
    """Calcule les erreurs L1 et L2 en faisant varier le pas temporel"""
    erreurs_L1 = []
    erreurs_L2 = []
    erreurs_Linf = []
    dt_list = []

    for N_t in N_temporel_list:
        T_num, x_i, t_i = Temperatures(N_spatial, N_t)
        T_exact = f_T_MMS(t_i[-1], x_i)

        erreur = np.abs(T_num[-1] - T_exact)

        dx = e / (N_spatial - 1)
        dt = t_max / N_t
        dt_list.append(dt)

        erreur_L1 = np.sum(erreur) * dx
        erreur_L2 = np.sqrt(np.sum(erreur**2) * dx)
        erreur_Linf = np.max(erreur)

        erreurs_L1.append(erreur_L1)
        erreurs_L2.append(erreur_L2)
        erreurs_Linf.append(erreur_Linf)

    # R�gression pour ordre de convergence
    slope_L1, _, _, _, _ = linregress(np.log(dt_list), np.log(erreurs_L1))
    slope_L2, _, _, _, _ = linregress(np.log(dt_list), np.log(erreurs_L2))
    slope_Linf, _, _, _, _ = linregress(np.log(dx_list), np.log(erreurs_Linf))

    # Affichage
    plt.figure(figsize=(10,4))

    plt.loglog(dt_list, erreurs_L1, 'o-', label=f"L1 (ordre = {abs(slope_L1):.2f})")
    plt.loglog(dt_list, erreurs_L2, 's-', label=f"L2 (ordre = {abs(slope_L2):.2f})")
    plt.loglog(dx_list, erreurs_Linf, 'd-', label=f"L∞ (ordre = {abs(slope_Linf):.2f})")
    plt.xlabel('dt')
    plt.ylabel('Erreur')
    plt.title("Erreur L1 et L2 temporelle")
    plt.legend()
    plt.grid(True)
    plt.show()



    return erreurs_L1, erreurs_L2, erreurs_Linf, dt_list, slope_L1, slope_L2, slope_Linf

N_temporel_list = [10, 20, 40, 80, 160]
N_spatial = 500  # Assez grand pour figer l'erreur spatiale

erreurs_L1, erreurs_L2, erreurs_Linf, dt_list, ordre_L1, ordre_L2, ordre_Linf = convergence_temporelle(N_temporel_list, N_spatial)

print(f"Ordre de convergence temporel L1 : {abs(ordre_L1):.2f}")
print(f"Ordre de convergence temporel L2 : {abs(ordre_L2):.2f}")
print(f"Ordre de convergence temporel Linf : {abs(ordre_Linf):.2f}")
