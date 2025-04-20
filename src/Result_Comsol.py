"""
Comparaison Comsol et Python
"""


# Importation des bibliothèques
import numpy as np 
import matplotlib.pyplot as plt
import config

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
def Temperatures(N_spatial=100,N_temporel=100,T_x_0=T_x_0,T_x_inf=T_x_inf,alpha=alpha,rho=rho,cp=cp,h=h):
    """Fonction de calcul des N températures en différences finies"""
    T_i = np.ones(N_spatial)*T_0                # Vecteur des N températures numériques calculées T_i
    T_i_n = np.zeros((N_temporel,N_spatial))    # Matrice des N températures numériques calculées T_i pour chaque itération
    matA = np.zeros((N_spatial,N_spatial))      # Matrice A pour la résolution du système matriciel
    vectB = np.zeros(N_spatial)                 # Vecteur B pour la résolution du système matriciel

    delta_x=e/(N_spatial-1)                     # Pas de discrétisation
           # Vecteur des N points r_i également espacées

    delta_t=t_max/N_temporel                    # Pas de discrétisation temporelle
      # Vecteur des N points t_i également espacées

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
    
    return T_i_n 

T_900 = np.array([1873, 1519.5365195543675, 1192.2006851483898, 912.0039794573862, 691.0896138361082,
                  531.343185409435, 425.7414560546323, 361.93937838181216,326.57452303034984, 308.458770585503,
                  299.80143151939876,295.9007688496783,294.2234736299725, 293.52302576635975, 293.2297761886693,
                  293.0995109397809])

T_1800 = np.array([1873,
    1621.3970415165056,
      1379.2892737270786,
     1155.218391432206,
      955.9491149402465,
      785.8875268678893,
      646.819456171123,
     538.0052424083274,
      456.6163441591961,
     398.4311738719759,
      358.63524206428355,
      332.5395693587561,
      316.0753132875159,
       306.0187216526329,
      299.9893729893522,
      296.3056035282268])



T_3600 = np.array([1873,
     1693.544839167548,
    1517.636762658667,
     1348.6205478769243,
    1189.4539296000107,
     1042.5558986666326,
     909.6995934451292,
       791.9561990077003,
     689.6905788561623,
     602.6038349060043,
      529.8132652483205,
     469.95680379453626,
      421.3073944910532,
     381.88313838064045,
      349.5415176206239,
      322.05032429767806])

x_diff = np.array ([0,                    
0.006666666666666667,     
0.013333333333333334,     
0.020000000000000004,     
0.026666666666666672,      
0.03333333333333335 ,     
0.040000000000000015,      
0.04666666666666669 ,      
0.05333333333333336 ,    
0.06000000000000003 ,      
0.06666666666666671 ,      
0.07333333333333339 ,     
0.08000000000000007 ,      
0.08666666666666674,
0.09333333333333342 ,      
0.1    ])



# Lancement de la simulation
N_spatial=100
N_temporel=100
T_i_n = Temperatures(N_spatial,N_temporel)
delta_x=e/(N_spatial-1) 
delta_t=t_max/N_temporel
x_i = np.linspace(0,e,N_spatial)
t_i = np.linspace(0,e,N_temporel)

# Affichage des résultats de la simulation + COMSOL
plt.figure(figsize=(10,6))

# Simulation Python
plt.plot(x_i, T_i_n[25], label='Python t=900s', linestyle='--')
plt.plot(x_i, T_i_n[50], label='Python t=1800s', linestyle='--')
plt.plot(x_i, T_i_n[99], label='Python t=3600s', linestyle='--')

# Données COMSOL
plt.plot(x_diff, T_900, 'o', label='COMSOL t=900s')
plt.plot(x_diff, T_1800, 'o', label='COMSOL t=1800s')
plt.plot(x_diff, T_3600, 'o', label='COMSOL t=3600s')

# Mise en forme du graphe
plt.title("Comparaison Simulation Python vs COMSOL")
plt.xlabel('Position x (m)')
plt.ylabel('Température T (K)')
plt.legend()
plt.grid(True)
plt.show()

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
