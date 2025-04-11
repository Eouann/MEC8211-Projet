"""
Fichier de validation du modèle numérique de diffusion thermique
"""


# Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
import config
from lancer_simulation_EulerImplicite import Temperatures


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
def calc_u_input(incertitude_T_ext,incertitude_T_four,incertitude_alpha,incertitude_cp,incertitude_rho,incertitude_h):
    """Fonction de calcul de u_input, l'incertitude de la validation, grace à la méthode des sensibilités"""
    S=np.zeros(6) # Liste des sensibilités
    # Calcul des sensibilités
    S1 = (Temperatures(T_x_inf=T_x_inf+incertitude_T_ext)-Temperatures(T_x_inf=T_x_inf-incertitude_T_ext))/2*incertitude_T_ext
    S[0] = S1
    S2 = (Temperatures(T_x_0=T_x_0+incertitude_T_four)-Temperatures(T_x_0=T_x_0-incertitude_T_four))/2*incertitude_T_four
    S[1] = S2
    S3 = (Temperatures(alpha=alpha+incertitude_alpha)-Temperatures(alpha=alpha-incertitude_alpha))/2*incertitude_alpha
    S[2] = S3
    S4 = (Temperatures(cp=cp+incertitude_cp)-Temperatures(cp=cp-incertitude_cp))/2*incertitude_cp
    S[3] = S4
    S5 = (Temperatures(rho=rho+incertitude_rho)-Temperatures(rho=rho-incertitude_rho))/2*incertitude_rho
    S[4] = S5
    S6 = (Temperatures(h=h+incertitude_h)-Temperatures(h=h-incertitude_h))/2*incertitude_h
    S[5] = S6
    
    u_input = np.sqrt(np.sum((S)**2))
    return u_input


# Définition de la fonction u_D de calcul de l'incertitude de mesure
def calc_u_D(T_exp,incertitude_ThermoCouple):
    """Fonction de calcul de u_D, l'incertitude de mesure"""
    # Calcul de l'écart-type des mesures expérimentales
    std_exp = np.std(T_exp)
    # Calcul de l'incertitude de mesure
    u_D = np.sqrt(incertitude_ThermoCouple**2+std_exp**2)
    return u_D


# Résultats de simulations pour u_num
r = 2 # Choisi arbitrairement
list_Nx = np.array([])
liste_dx = np.array([])
liste_T = np.array([]) # f_r^2.h, f_r.h, f_h
p_f = 2


# Définition des incertitudes pour l'analyse des sensibilités
incertitude_T_ext = 1       # En K
incertitude_T_four = 20     # En K
incertitude_alpha = 0.1e-6  # En m²/s
incertitude_cp = 100        # En J/(kg.K)
incertitude_rho = 50        # En kg/m³
incertitude_h = 10          # En W/(m².K)


# Définition des resultas expérimentaux
T_exp = np.array([290.3023615,296.5679481,293.3943294,301.7692329,309.2371422,329.1088,341.6838723,364.5406292,374.9108509,385.4515811]) # Liste des resultats expérimentaux
incertitude_ThermoCouple = 1 # En K

'''
# Importation des données
T_simu = np.array([])       # Listes de resultats de simulations
S = np.mean(T_simu)         # Valeur moyenne des simulations
D = np.mean(T_exp)          # Valeur moyenne des mesures expérimentales
u_num = calc_u_num(liste_T,r,p_f) # Incertitude numérique
u_input = calc_u_input(incertitude_T_ext,incertitude_T_four,incertitude_alpha)  # Incertitude des parametres d'entrée
u_D = calc_u_D(T_exp,incertitude_ThermoCouple)                   # Incertitude des mesures experimentales
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
