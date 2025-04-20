"""
Fichier de création des graphiques heat map de la solution manufacturée et du terme source
""" 

# Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import config


# Importation des constantes
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

# Définition des paramètres
tmin, tmax = 0, t_max
xmin, xmax = 0, e
nt, nx = 100, 100

# Création des maillages temporel et spatial
tdom = np.linspace(tmin, tmax, nt)
xdom = np.linspace(xmin, xmax, nx)
ti, xi = np.meshgrid(tdom, xdom, indexing='ij')

# Évaluation des fonctions sur le maillage
z_MMS = f_T_MMS(ti, xi)
z_source = f_source(ti, xi)

# Tracé des résultats
plt.figure()
plt.contourf(xi, ti, z_MMS, levels=50)
plt.colorbar()
plt.title('Solution Manufacturée')
plt.xlabel('X')
plt.ylabel('t')
plt.savefig('results/verification-solution-manufacturee.png')
plt.show()

plt.figure()
plt.contourf(xi, ti, z_source, levels=50)
plt.colorbar()
plt.title('Terme Source')
plt.xlabel('X')
plt.ylabel('t')
plt.savefig('results/verification-terme source.png')
plt.show()
