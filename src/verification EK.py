"""
Fichier de vérification par MMS
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


# Calcul des températures pour N points
def Temperatures(N_spatial,N_temporel,T_x_0=T_x_0,T_x_inf=T_x_inf,alpha=alpha,rho=rho,cp=cp,h=h):
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
        vectB[0] = T_0+T_x_0*(1-np.exp(-t_i[i]/t_max))
        for j in range(1, N_spatial - 1):
            vectB[j] = T_i[j]*delta_x**2

        # Résolution du système matriciel
        T_i = np.linalg.solve(matA, vectB)
        T_i_n[i] = T_i
    
    return T_i_n , x_i, t_i


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

'''
# Tracé de la solution manufacturée
plt.figure()
t_values = np.linspace(0, t_max, 100)
x_values = np.linspace(0, e, 100)
z_MMS = f_T_MMS(t_values[:, None], x_values[None, :])
plt.contourf(x_values, t_values, z_MMS, levels=100)
plt.colorbar(label='Température')
plt.xlabel('x (m)')
plt.ylabel('t (s)')
plt.title('Solution manufacturée')
#plt.savefig('results/solution_manufacturee.png')
plt.show()

plt.figure()
t_values = np.linspace(0, t_max, 100)
x_values = np.linspace(0, e, 100)
z_MMS = f_T_MMS(t_values[:, None], x_values[None, :])
plt.plot(x_values, z_MMS[0], label='t=0 s')
plt.plot(x_values, z_MMS[25], label='t=900 s')
plt.plot(x_values, z_MMS[50], label='t=1800 s')
plt.plot(x_values, z_MMS[75], label='t=2700 s')
plt.plot(x_values, z_MMS[99], label='t=3600 s')
plt.xlabel('Position x (m)')
plt.ylabel('Température T (K)')
plt.title("Solution manufacturée")
plt.legend()
#plt.savefig('results/solution_manufacturee.png')
plt.show()


# Tracé du terme source
plt.figure()
t_values = np.linspace(0, t_max, 100)
x_values = np.linspace(0, e, 100)
z_source = f_source(t_values[:, None], x_values[None, :])
plt.plot(x_values, z_source[0], label='t=0 s')
plt.plot(x_values, z_source[25], label='t=900 s')
plt.plot(x_values, z_source[50], label='t=1800 s')
plt.plot(x_values, z_source[75], label='t=2700 s')
plt.plot(x_values, z_source[99], label='t=3600 s')
plt.xlabel('Position x (m)')
plt.ylabel('Température T (K)')
plt.title("Solution manufacturée")
plt.legend()
#plt.savefig('results/solution_manufacturee.png')
plt.show()
'''

# Analyse de convergence spatiale
listN_spatial=[3, 4, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000]
vectDelta_x = np.zeros(len(listN_spatial))
vectL1 = np.zeros(len(listN_spatial))
vectL2 = np.zeros(len(listN_spatial))
vectLinf = np.zeros(len(listN_spatial))
N_temporel = 1000

for i in (range(len(listN_spatial))):
    N = listN_spatial[i]
    T_i_n , x_i, t_i=Temperatures(N_spatial=N, N_temporel=N_temporel)
    vectDelta_x[i]=e/(N-1)
    x_MMS_values = np.linspace(0, e, N)
    t_MMS_values = np.linspace(0, t_max, N_temporel)
    y_MMS_values=f_T_MMS(t_MMS_values[:, None], x_MMS_values[None, :])
    L1=erreurs.ErreurL1(T_i_n,y_MMS_values,N,N_temporel)
    vectL1[i]=L1
    L2=erreurs.ErreurL2(T_i_n,y_MMS_values,N,N_temporel)
    vectL2[i]=L2
    Linf=erreurs.ErreurLinf(T_i_n,y_MMS_values)
    vectLinf[i]=Linf

plt.figure()

# L1
plt.plot(vectDelta_x, vectL1, 'o', color='red', label='L1')
slope_L1, intercept_L1, r_value_L1, p_value_L1, std_err_L1 = linregress(np.log(vectDelta_x[:3]), np.log(vectL1[:3]))
y_pred_L1 =  np.exp(intercept_L1) * vectDelta_x[:3]**slope_L1
plt.plot(vectDelta_x[:3], y_pred_L1, '--', color='red', label=f'Ordre de convergence L1: {slope_L1}')

# L2
plt.plot(vectDelta_x, vectL2, 'o', color='green', label='L2')
slope_L2, intercept_L2, r_value_L2, p_value_L2, std_err_L2 = linregress(np.log(vectDelta_x[:3]), np.log(vectL2[:3]))
y_pred_L2 = np.exp(intercept_L2) * vectDelta_x[:3]**slope_L2
plt.plot(vectDelta_x[:3], y_pred_L2, '--', color='green', label=f'Ordre de convergence L2: {slope_L2}')

# Linf
plt.plot(vectDelta_x, vectLinf, 'o', color='blue', label='Linf')
slope_L3, intercept_L3, r_value_L3, p_value_L3, std_err_L3 = linregress(np.log(vectDelta_x[:3]), np.log(vectLinf[:3]))
y_pred_Linf = np.exp(intercept_L3) * vectDelta_x[:3]**slope_L3
plt.plot(vectDelta_x[:3], y_pred_Linf, '--', color='blue', label=f'Ordre de convergence Linf: {slope_L3}')

plt.title('Erreurs L1, L2 et Linf en fonction du nombre de points N spatiaux')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Delta x (m)')
plt.ylabel('Température (K)')
plt.grid(True, which="both", ls="--")
plt.show()
'''
# Analyse de convergence temporel
listN_temporel=[3, 4, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 3000, 10000, 30000, 100000]
vectDelta_t = np.zeros(len(listN_temporel))
vectL1 = np.zeros(len(listN_temporel))
vectL2 = np.zeros(len(listN_temporel))
vectLinf = np.zeros(len(listN_temporel))
N_spatial = 300

for i in (range(len(listN_temporel))):
    N = listN_temporel[i]
    T_i_n , x_i, t_i=Temperatures(N_spatial=100, N_temporel=N)
    vectDelta_t[i]=t_max/(N-1)
    x_MMS_values = np.linspace(0, e, 100)
    t_MMS_values = np.linspace(0, t_max, N)
    y_MMS_values=f_T_MMS(t_MMS_values[:, None], x_MMS_values[None, :])
    L1=erreurs.ErreurL1(T_i_n,y_MMS_values,100,N)
    vectL1[i]=L1
    L2=erreurs.ErreurL2(T_i_n,y_MMS_values,100,N)
    vectL2[i]=L2
    Linf=erreurs.ErreurLinf(T_i_n,y_MMS_values)
    vectLinf[i]=Linf

plt.figure()

# L1
plt.plot(vectDelta_t, vectL1, 'o', color='red', label='L1')
slope_L1, intercept_L1, r_value_L1, p_value_L1, std_err_L1 = linregress(np.log(vectDelta_t[:3]), np.log(vectL1[:3]))
y_pred_L1 =  np.exp(intercept_L1) * vectDelta_t[:3]**slope_L1
plt.plot(vectDelta_t[:3], y_pred_L1, '--', color='red', label=f'Ordre de convergence L1: {slope_L1}')

# L2
plt.plot(vectDelta_t, vectL2, 'o', color='green', label='L2')
slope_L2, intercept_L2, r_value_L2, p_value_L2, std_err_L2 = linregress(np.log(vectDelta_t[:3]), np.log(vectL2[:3]))
y_pred_L2 = np.exp(intercept_L2) * vectDelta_t[:3]**slope_L2
plt.plot(vectDelta_t[:3], y_pred_L2, '--', color='green', label=f'Ordre de convergence L2: {slope_L2}')

# Linf
plt.plot(vectDelta_t, vectLinf, 'o', color='blue', label='Linf')
slope_L3, intercept_L3, r_value_L3, p_value_L3, std_err_L3 = linregress(np.log(vectDelta_t[:3]), np.log(vectLinf[:3]))
y_pred_Linf = np.exp(intercept_L3) * vectDelta_t[:3]**slope_L3
plt.plot(vectDelta_t[:3], y_pred_Linf, '--', color='blue', label=f'Ordre de convergence Linf: {slope_L3}')

plt.title('Erreurs L1, L2 et Linf en fonction du nombre de points N temporel')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Delta t (s)')
plt.ylabel('Température (K)')
plt.grid(True, which="both", ls="--")
plt.show()
'''
