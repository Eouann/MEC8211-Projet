"""
Fichier de vérification du modèle numérique de diffusion thermique
""" 

# Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import config 

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
plt.savefig('solution_manufacturée.png')
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
plt.savefig('terme_source.png')
plt.show()
