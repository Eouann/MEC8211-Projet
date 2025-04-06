"""
Fichier de validation du modèle numérique de diffusion thermique
"""


# Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt


# Résultats de simulations
r = 2 # Choisi arbitrairement
list_Nx = np.array([])
liste_dx = np.array([])
liste_T = np.array([]) # f_r^2.h, f_r.h, f_h
p_f = 2


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


# Importation des données
T_simu = np.array([])       # Listes de resultats de simulations
S = np.mean(T_simu)         # Valeur moyenne des simulations
D = XXX                     # Valeur moyenne des mesures expérimentales
u_num = calc_u_num(liste_T,r,p_f) # Incertitude numérique
u_input = XXX               # Incertitude des parametres d'entrée
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
