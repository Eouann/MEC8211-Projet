"""
Fichier de validation du modèle numérique de diffusion thermique
"""


# Importation des bibliothèques
import numpy as np
import lhsmdu
import matplotlib.pyplot as plt
import config
from lancer_simulation_EulerImplicite import Temperatures


# Définition des constantes
alpha = config.alpha
e = config.e
T_0 = config.T_0
T_x_0 = config.T_x_0
T_x_e = config.T_x_e
t_max = config.t_max


# Définition de la fonction u_num
def calc_u_num(liste_f,r,p_f):
    """Fonction de calcl de u_num, l'incertitude numérique, grace au GCI (Grid Convergence Index)"""
    p_chapeau = np.log((liste_f[0]-liste_f[1])/(liste_f[1]-liste_f[2]))/np.log(r)
    if np.abs((p_chapeau-p_f)/p_f) <= 0.1:
        GCI = 1.25/(r**p_f-1)*np.abs(liste_f[1]-liste_f[2])
    else:
        p=min(max(0.5,p_chapeau),p_f)
        GCI = 3/(r**p-1)*np.abs(liste_f[1]-liste_f[2])
    u_num = GCI/2
    return u_num

# Définition de la fonction u_input de propagation des incertitudes d'entrée
def calc_u_input(incertitude_T_ext,incertitude_T_four,incertitude_alpha):
    """Fonction de calcul de u_input, l'incertitude de la validation, grace à la propagation des incertitudes"""
    # Nombre d'échantillons
    N_samples = 100

    # Générer un hypercube latin
    lhs_samples = lhsmdu.sample(3, N_samples)

    # Intervalles des entrées
    a1, b1 = T_x_e-incertitude_T_ext, T_x_e+incertitude_T_ext
    a2, b2 = T_x_0-incertitude_T_four, T_x_0+incertitude_T_four
    a3, b3 = alpha-incertitude_alpha, alpha+incertitude_alpha

    # Échelle des échantillons pour correspondre aux intervalles d'incertitude
    scaled_samples = np.zeros_like(lhs_samples)
    scaled_samples[:, 0] = a1 + (b1 - a1) * lhs_samples[:, 0]
    scaled_samples[:, 1] = a2 + (b2 - a2) * lhs_samples[:, 1]
    scaled_samples[:, 2] = a3 + (b3 - a3) * lhs_samples[:, 2]

    # Propager les incertitudes
    results = Temperatures(scaled_samples[:, 0], scaled_samples[:, 1], scaled_samples[:, 2])

    # Analyser les résultats
    mean_result = np.mean(results)
    std_result = np.std(results)

    return mean_result, std_result


# Résultats de simulations pour u_num
r = 2 # Choisi arbitrairement
list_Nx = np.array([])
liste_dx = np.array([])
liste_T = np.array([]) # f_r^2.h, f_r.h, f_h
p_f = 2

# Définition des incertitudes à propager
incertitude_T_ext = 1       # En K
incertitude_T_four = 5      # En K
incertitude_alpha = 0.02e-6 # En m²/s
'''
# Importation des données
T_simu = np.array([])       # Listes de resultats de simulations
S = np.mean(T_simu)         # Valeur moyenne des simulations
D = XXX                     # Valeur moyenne des mesures expérimentales
u_num = calc_u_num(liste_T,r,p_f) # Incertitude numérique
u_input = calc_u_input(incertitude_T_ext,incertitude_T_four,incertitude_alpha)  # Incertitude des parametres d'entrée
u_D = XXX                   # Incertitude des mesures experimentales
k = 2                       # Car u_num a ete déterminé grace au GCI


# Définition de la fonction E
def Calcul_E(S,D):
    """Fonction de calcul de E l'erreur de simulation"""
    E = S-D
    return E


# Execution du calcul
E=Calcul_E(S,D)
print("La valeur de E est : ",E)


# Calcul de u_val, l'incertitude de validation
def Calcul_u_val(u_num,u_input,u_D):
    """Fonction de calcul de u_val l'incertitude de validation"""
    u_val = np.sqrt(u_num**2+u_input**2+u_D**2)
    return u_val


# Intervalle de delta_model
intervalle_sup=E+k*Calcul_u_val(u_num,u_input,u_D)
intervalle_inf=E-k*Calcul_u_val(u_num,u_input,u_D)

print("Delta_model, l'erreur du modèle se trouve dans l'intervalle [",intervalle_inf,";",intervalle_sup,"] avec une confiance de 95,4 %")
'''
