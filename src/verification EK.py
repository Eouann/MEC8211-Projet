"""
Fichier de v√©rification par MMS
"""


# Importation des biblioth√®ques
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.stats import linregress
import config
import erreurs


# D√©finition des constantes
alpha=config.alpha
e=config.e
cp=config.cp
rho=config.rho
T_0=config.T_0
T_x_0=config.T_x_0
T_x_inf=config.T_x_inf
t_max=config.t_max
h=config.h


# D√©finition des variables symboliques
t, x = sp.symbols('t x')

# Solution manufactur√©e
T_MMS = T_0+T_x_0*sp.exp(-h*x/(alpha*rho*cp))*(1-sp.exp(-t/t_max))

# Calcul des d√©riv√©es
T_t = sp.diff(T_MMS, t)
T_x = sp.diff(T_MMS, x)
T_xx = sp.diff(T_MMS, x, x)

# Calcul du terme source S(t,x)
source = T_t - alpha * T_xx

# Conversion en fonctions Python
f_T_MMS = sp.lambdify([t, x], T_MMS, "numpy")
f_source = sp.lambdify([t, x], source, "numpy")

# Calcul des temp√©ratures pour N points
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
        
        # La condition de Robin est déjà gérée hors de la boucle
        T_i = np.linalg.solve(matA, vectB)
        T_i_n[n] = T_i

    return T_i_n, x_i, t_i


def convergence_et_erreur(N_spatial_list, N_temporel):
    """Calcule les erreurs L1 et L2 et l'ordre de convergence"""
    erreurs_L1 = []
    erreurs_L2 = []
    dx_list = []

    for N in N_spatial_list:
        T_num, x_i, t_i = Temperatures(N, N_temporel)
        T_exact = f_T_MMS(t_i[-1], x_i)

        erreur = np.abs(T_num[-1] - T_exact)

        dx = e / (N - 1)
        dx_list.append(dx)

        erreur_L1 = np.sum(erreur) * dx
        erreur_L2 = np.sqrt(np.sum(erreur**2) * dx)

        erreurs_L1.append(erreur_L1)
        erreurs_L2.append(erreur_L2)

    # Régression linéaire pour ordre de convergence
    slope_L1, _, _, _, _ = linregress(np.log(dx_list), np.log(erreurs_L1))
    slope_L2, _, _, _, _ = linregress(np.log(dx_list), np.log(erreurs_L2))

    # Affichage
    plt.figure(figsize=(10,4))
    
    plt.loglog(dx_list, erreurs_L1, 'o-', label=f"L1 (ordre ≈ {abs(slope_L1):.2f})")
    plt.loglog(dx_list, erreurs_L2, 's-', label=f"L2 (ordre ≈ {abs(slope_L2):.2f})")
    plt.xlabel('dx')
    plt.ylabel('Erreur')
    plt.title("Erreurs L1 et L2 spatiale")
    plt.legend()
    plt.grid(True)
    plt.show()


    return erreurs_L1, erreurs_L2, dx_list, slope_L1, slope_L2

N_spatial_list = [10, 20, 40, 80, 160]
N_temporel = 100

erreurs_L1, erreurs_L2, dx_list, ordre_L1, ordre_L2 = convergence_et_erreur(N_spatial_list, N_temporel)

print(f"Ordre de convergence L1 : {abs(ordre_L1):.2f}")
print(f"Ordre de convergence L2 : {abs(ordre_L2):.2f}")

def convergence_temporelle(N_temporel_list, N_spatial):
    """Calcule les erreurs L1 et L2 en faisant varier le pas temporel"""
    erreurs_L1 = []
    erreurs_L2 = []
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

        erreurs_L1.append(erreur_L1)
        erreurs_L2.append(erreur_L2)

    # Régression pour ordre de convergence
    slope_L1, _, _, _, _ = linregress(np.log(dt_list), np.log(erreurs_L1))
    slope_L2, _, _, _, _ = linregress(np.log(dt_list), np.log(erreurs_L2))

    # Affichage
    plt.figure(figsize=(10,4))

    plt.loglog(dt_list, erreurs_L1, 'o-', label=f"L1 (ordre ≈ {abs(slope_L1):.2f})")
    plt.loglog(dt_list, erreurs_L2, 's-', label=f"L2 (ordre ≈ {abs(slope_L2):.2f})")
    plt.xlabel('dt')
    plt.ylabel('Erreur')
    plt.title("Erreur L1 et L2 temporelle")
    plt.legend()
    plt.grid(True)
    plt.show()



    return erreurs_L1, erreurs_L2, dt_list, slope_L1, slope_L2

N_temporel_list = [10, 20, 40, 80, 160]
N_spatial = 500  # Assez grand pour figer l'erreur spatiale

erreurs_L1, erreurs_L2, dt_list, ordre_L1, ordre_L2 = convergence_temporelle(N_temporel_list, N_spatial)

print(f"Ordre de convergence temporel L1 : {abs(ordre_L1):.2f}")
print(f"Ordre de convergence temporel L2 : {abs(ordre_L2):.2f}")


'''
# Trac√© de la solution manufactur√©e
plt.figure()
t_values = np.linspace(0, t_max, 100)
x_values = np.linspace(0, e, 100)
z_MMS = f_T_MMS(t_values[:, None], x_values[None, :])
plt.contourf(x_values, t_values, z_MMS, levels=100)
plt.colorbar(label='Temp√©rature')
plt.xlabel('x (m)')
plt.ylabel('t (s)')
plt.title('Solution manufactur√©e')
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
plt.ylabel('Temp√©rature T (K)')
plt.title("Solution manufactur√©e")
plt.legend()
#plt.savefig('results/solution_manufacturee.png')
plt.show()


# Trac√© du terme source
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
plt.ylabel('Temp√©rature T (K)')
plt.title("Solution manufactur√©e")
plt.legend()
#plt.savefig('results/solution_manufacturee.png')
plt.show()
'''
"""
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
plt.ylabel('Temp√©rature (K)')
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
plt.ylabel('Temp√©rature (K)')
plt.grid(True, which="both", ls="--")
plt.show()
'''
"""
