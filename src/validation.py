"""
Fichier de validation du modèle numérique de diffusion thermique
"""


# Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
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
    S_Xi=np.zeros(6) # Liste des sensibilités
    u_Xi = np.zeros(6) # Liste des incertitudes standards des paramètres d'entrée

    # Calcul des sensibilités et des incertitudes standards
    S_Xi[0] = (Temperatures(T_x_inf=T_x_inf+0.01*T_x_inf)[-1,-1]-Temperatures(T_x_inf=T_x_inf-0.01*T_x_inf)[-1,-1])/2*incertitude_T_ext
    u_Xi[0] = incertitude_T_ext
    S_Xi[1] = (Temperatures(T_x_0=T_x_0+0.01*T_x_0)[-1,-1]-Temperatures(T_x_0=T_x_0-0.01*T_x_0)[-1,-1])/2*incertitude_T_four
    u_Xi[1] = incertitude_T_four
    S_Xi[2] = (Temperatures(alpha=alpha+0.01*alpha)[-1,-1]-Temperatures(alpha=alpha-0.01*alpha)[-1,-1])/2*incertitude_alpha
    u_Xi[2] = incertitude_alpha
    S_Xi[3] = (Temperatures(cp=cp+0.01*cp)[-1,-1]-Temperatures(cp=cp-0.01*cp)[-1,-1])/2*incertitude_cp
    u_Xi[3] = incertitude_cp
    S_Xi[4] = (Temperatures(rho=rho+0.01*rho)[-1,-1]-Temperatures(rho=rho-0.01*rho)[-1,-1])/2*incertitude_rho
    u_Xi[4] = incertitude_rho
    S_Xi[5] = (Temperatures(h=h+0.01*h)[-1,-1]-Temperatures(h=h-0.01*h)[-1,-1])/2*incertitude_h
    u_Xi[5] = incertitude_h

    # Calcul de l'incertitude des paramètres d'entrée
    print("S_Xi : ",S_Xi)
    u_input = np.sqrt(np.sum((S_Xi*u_Xi)**2))
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
p_f = 1 # Ordre de convergence du schéma temporel
list_N_temporel = np.zeros(3)
liste_delta_t = np.array([18*r*r,18*r,18])
liste_T = np.zeros(3) # Dans l'ordre f_r^2.h, f_r.h, f_h
for i in range(3):
    list_N_temporel[i] = int(t_max/liste_delta_t[i])
    liste_T[i] = Temperatures(N_temporel=int(list_N_temporel[i]))[-1,-1] # Dernière température de la simulation


# Définition des incertitudes pour l'analyse des sensibilités
incertitude_T_ext = 0.5     # En K
incertitude_T_four = 5      # En K
incertitude_alpha = 0.03e-6 # En m²/s
incertitude_cp = 5          # En J/(kg.K)
incertitude_rho = 2.5       # En kg/m³
incertitude_h = 1.5         # En W/(m².K)


# Définition des resultas expérimentaux
T_exp = np.array([320.8568087,323.3207646,315.8538349,322.2963,325.6347422,323.3933496,316.7395012,324.2282505,322.6564081,317.5701481]) # Liste des resultats expérimentaux
incertitude_ThermoCouple = 0.5 # En K


# Importation des données
S = Temperatures()[-1,-1]      # Listes de resultats de simulations
D = np.mean(T_exp)                  # Valeur moyenne des mesures expérimentales
u_num = calc_u_num(liste_T,r,p_f)   # Incertitude numérique
u_input = calc_u_input(incertitude_T_ext,incertitude_T_four,incertitude_alpha,incertitude_cp,incertitude_rho,incertitude_h)  # Incertitude des parametres d'entrée
u_D = calc_u_D(T_exp,incertitude_ThermoCouple)                   # Incertitude des mesures experimentales
k = 2                               # Car u_num a ete déterminé grace au GCI


# Définition de la fonction E
def Calcul_E(S,D):
    """Fonction de calcul de E l'erreur de simulation"""
    E = S-D
    return E


# Execution du calcul
E=Calcul_E(S,D)


# Calcul de u_val, l'incertitude de validation
def calc_u_val(u_num,u_input,u_D):
    """Fonction de calcul de u_val l'incertitude de validation"""
    u_val = np.sqrt(u_num**2+u_input**2+u_D**2)
    return u_val


# Intervalle de delta_model
intervalle_sup=E+k*calc_u_val(u_num,u_input,u_D)
intervalle_inf=E-k*calc_u_val(u_num,u_input,u_D)


# Graphique de convergence asymptotique
liste_delta_t_conv_asymp = np.array([300,200,100,72,50,36,30,20,18,10,5,2])
liste_T_conv_asymp = np.zeros(len(liste_delta_t_conv_asymp))
for i in range(len(liste_T_conv_asymp)):
    N_temporel = int(t_max/liste_delta_t_conv_asymp[i])
    liste_T_conv_asymp[i] = Temperatures(N_temporel=int(N_temporel))[-1,-1]
liste_T_normalisé = (liste_T_conv_asymp[-1]-liste_T_conv_asymp)/liste_T_conv_asymp[-1] # Normalisation des températures pour la convergence asymptotique

plt.plot(liste_delta_t_conv_asymp,liste_T_normalisé,'o', color='red')
slope, intercept, r_value, p_value, std_err = linregress(np.log(liste_delta_t_conv_asymp[:-1]), np.log(liste_T_normalisé[:-1]))
y_pred =  np.exp(intercept) * liste_delta_t_conv_asymp[:-1]**slope
plt.plot(liste_delta_t_conv_asymp[:-1], y_pred, '--', color='red', label=f'Ordre de convergence : {slope}')
plt.xlabel('Delta t [s]')
plt.ylabel('Température [K]')
plt.title('Vérification de la convergence asymptotique')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.savefig('results/validation-analyse-de-convergence-asymptotique.png')
plt.show()


# Affichage des résultats
print("La valeur de S est : ",S)
print("La valeur de D est : ",D)
print("La valeur de E est : ",E)
print("L'incertitude numérique est : ",u_num)
print("L'incertitude d'entrée est : ",u_input)
print("L'incertitude de mesure est : ",u_D)
print("La valeur de u_val est : ",calc_u_val(u_num,u_input,u_D))
print("Delta_model, l'erreur du modèle se trouve dans l'intervalle [",intervalle_inf,";",intervalle_sup,"] avec une confiance de 95,4 %")
