"""
Fichier de vérification du modèle numérique de diffusion thermique
""" 

# Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import config 
import erreurs


# Importation des constantes
alpha = config.alpha        # m²/s (matière réfractaire)
e = config.e                # m (épaisseur du mur du four)
cp = config.cp              # J/(kg.K) (capacité thermique massique de la matière réfractaire)
rho = config.rho            # kg/m³ (masse volumique de la matière réfractaire)
h = config.h                # W/m2.K (coefficient de transfert thermique convectif)
k = alpha * rho * cp                 #   (conductivité thermique de l'isolant)
T_0 = config.T_0            # K (température initiale du mur)
T_x_0 = config.T_x_0        # K (température de la flamme du four)
T_x_inf = config.T_x_inf    # K (température de l'air ambiant dans l'atelier)
t_max = config.t_max        # s (durée de la simulation)

x,t = sp.symbols('x t')

# Solution MMS
T_MMS = T_0 + (T_x_0 - T_x_inf) * sp.exp(-h*x /k) * (1 - sp.exp(- t /t_max))

# Transformation de la solution manufacturée sympy en fonction traçable par matplotlib
f_T_MMS = sp.lambdify([x,t], T_MMS, modules=['numpy'])

# Terme Source
def terme_source (x,t) :
    terme1 = 1/t_max * np.exp(-t/t_max)
    terme2 = alpha * (h/k)**2 * (1 - np.exp(- t /t_max))
    facteur = (T_x_0 - T_x_inf) * np.exp(-h*x/k)
    return facteur * (terme1 - terme2)

# Tracé de la Solution MMS
x_values = np.linspace (0,e,500)
plt.figure()

for t in ([0, t_max/4, t_max/2, t_max]):
    y_values = f_T_MMS(x_values,t)
    plt.plot(x_values, y_values, label=f't={t} s')
    
plt.title("Solution manufacturée fonction de x à différents instants")
plt.xlabel('x (m)')
plt.ylabel('Solution manufacturée')
plt.legend()
plt.grid()
plt.savefig('results/solution_manufacturée.png')
plt.show()

# Tracé du Terme Source  
plt.figure()

for t in ([0, t_max/4, t_max/2, t_max]):
    S_values = np.zeros(len(x_values))
    for i in range(len(x_values)):
        S_values[i] = terme_source(x_values[i],t)
    plt.plot(x_values, S_values, label=f't={t} s')

plt.title("Terme source en fonction de x à différents instants")
plt.xlabel('x (m)')
plt.ylabel('Terme source')
plt.legend()
plt.grid()
plt.savefig('results/terme_source.png')
plt.show()

# Analyse de convergence spatiale et temporelle
def Temperatures(N_spatial, N_temporel):
   """Fonction de calcul des N températures en différences finies avec terme source MMS"""
   T_i = np.ones(N_spatial) * T_0                  # Vecteur des N températures numériques calculées T_i
   T_i_n = np.zeros((N_temporel, N_spatial))       # Matrice des N températures numériques T_i pour chaque itération
   matA = np.zeros((N_spatial, N_spatial))         # Matrice A pour la résolution du système matriciel
   vectB = np.zeros(N_spatial)                     # Vecteur B (second membre)

   delta_x = e / (N_spatial - 1)                   # Pas spatial
   x_i = np.linspace(0, e, N_spatial)              # Points en x

   delta_t = t_max / N_temporel                    # Pas temporel
   t_i = np.linspace(0, t_max, N_temporel)         # Points en t

   # Condition de Dirichlet en x = 0
   matA[0, 0] = 1
   vectB[0] = T_x_0

   # Condition de Robin en x = e (différenciation d’ordre 1)
   matA[-1, -1] = 1 + delta_x * h / k
   matA[-1, -2] = -1
   vectB[-1] = h * delta_x * T_x_inf / k
   
   # Remplissage de la matrice A (ne dépend pas du temps)
   for i in range(1, N_spatial - 1):
       matA[i, i - 1] = -alpha * delta_t
       matA[i, i]     = delta_x**2 + 2 * alpha * delta_t
       matA[i, i + 1] = -alpha * delta_t

   T_i_n[0] = T_i

   for i in range(1, N_temporel):
       t = t_i[i]
       for j in range(1, N_spatial - 1):
           x = x_i[j]
           S = terme_source(x, t)
           vectB[j] = T_i[j] * delta_x**2 + delta_t * delta_x**2 * S  # ajout du terme source


       # Résolution du système
       T_i = np.linalg.solve(matA, vectB)
       T_i_n[i] = T_i

   return T_i_n, x_i, t_i


# Lancement de la simulation
N_spatial=100
N_temporel=100
T_i_n,x_i,t_i = Temperatures(N_spatial,N_temporel)


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
plt.savefig('results/simulation diff.png')
plt.show()


def Calcul_Matrice_Terme_Exact(x_i,t_i):
    T_ex = []
    for i in t_i :
        A = []
        for j in x_i :
            A.append(f_T_MMS(j,i))
        T_ex.append(A)
    return np.array(T_ex)
            
      
def Erreur_Temporel():
    """Étude de l'erreur en fonction du nombre de points temporels (résolution temporelle)"""
    erreurs_L1, erreurs_L2, taille_maillage = [], [], []
    N_spatial = 100 #On fixe le pas spatial
    N_temp_list = [200, 500, 1000, 1500, 2000, 2500] # Prise du nombre de point temporel a étudier
    delta_t_liste = [t_max / (i-1) for i in N_temp_list]

    for N_temp in N_temp_list:
        T_num, x_i, t_i = Temperatures(N_spatial, N_temp) #Calcul des valeurs de la MMS ainsi que des listes de pas
        T_ana = Calcul_Matrice_Terme_Exact(x_i,t_i)
        diff_totale = (1/(N_temp*N_spatial))*np.sum(np.abs(T_num - T_ana))
        diff_totale2 = np.sqrt((1/(N_temp*N_spatial))*np.sum((T_num - T_ana)**2))
        erreurs_L1.append(diff_totale)
        erreurs_L2.append(diff_totale2)
        
        print(erreurs_L1)
        
        
    
    return(erreurs_L1,erreurs_L2,delta_t_liste)

        
err_L1, err_L2, delta_t_liste = Erreur_Temporel()

plt.loglog(delta_t_liste, err_L1, 'o-', label="Erreur L1")
plt.loglog(delta_t_liste, err_L2, 's-', label="Erreur L2")
plt.loglog(delta_t_liste, [err_L1[0]*(dt/delta_t_liste[0]) for dt in delta_t_liste], '--', label='Ordre 1')
plt.xlabel('Delta t')
plt.ylabel('Erreur')
plt.legend()
plt.grid(True)
plt.title("Étude de l’ordre de convergence temporel")
plt.savefig('results/erreurs.png')
plt.show()
pente = np.polyfit(np.log(delta_t_liste), np.log(err_L1), 1)
print("Ordre de convergence estimé (L1) :", pente[0])
