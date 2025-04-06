"""
Fichier de vérification du modèle numérique de diffusion thermique
""" 

# Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
import config 

# Importation des constantes
alpha = config.alpha        # m²/s (matière réfractaire)
e = config.e                # m (épaisseur du mur du four)
T_0 = config.T_0            # K (température initiale du mur)
T_x_0 = config.T_x_0        # K (température de la flamme du four)
T_x_e = config.T_x_e        # K (température de l'air ambiant dans l'atelier)
t_max = config.t_max        # s (durée de la simulation)

# === Discrétisation ===
Nx = 100          # points d'espace
Nt = 500          # pas de temps
x = np.linspace(0, e, Nx)
dx = x[1] - x[0]
dt = t_max / Nt
t = np.linspace(0, t_max, Nt+1)


# Définition de la solution exacte MMS 
def T_exact (x,t):
    return T_0 + (T_x_0 - T_x_e) * (1 - x/e)**2 * (1 - np.exp(- t /t_max))


# === Terme source MMS ===
def source(x, t):
    term1 = (1 - x/e)**2 * 1/t_max * np.exp(- t / t_max)
    term2 = -2 * alpha / e**2 * (1 - np.exp(- t / t_max))
    return (T_x_0 - T_x_e) * (term1 + term2)

# === Matrice Crank-Nicolson ===
r = alpha * dt / (dx**2)
A = np.zeros((Nx, Nx))
B = np.zeros((Nx, Nx))

for i in range(1, Nx-1):
    A[i, i-1] = -r / 2
    A[i, i]   = 1 + r
    A[i, i+1] = -r / 2
    B[i, i-1] = r / 2
    B[i, i]   = 1 - r
    B[i, i+1] = r / 2

# Conditions de Dirichlet : T(0,t)=1600, T(e,t)=20
A[0, 0] = A[-1, -1] = 1
B[0, 0] = B[-1, -1] = 1

# === Initialisation ===
Tn = np.ones(Nx) * 20  # T(x,0) = 20
T_all = [Tn.copy()]

# === Boucle en temps ===
for n in range(1, Nt+1):
    tn = t[n]
    tn1 = t[n-1]

    S_n = source(x, tn)
    S_n1 = source(x, tn1)

    b = B @ Tn + 0.5 * dt * (S_n + S_n1)

    # Imposer CLs :
    b[0] = T_x_0
    b[-1] = T_x_e
    A[0, :] = 0
    A[-1, :] = 0
    A[0, 0] = 1
    A[-1, -1] = 1

    # Résolution du système linéaire
    Tn1 = np.linalg.solve(A, b)
    T_all.append(Tn1.copy())
    Tn = Tn1

# === Affichage ===
plt.figure(figsize=(10,6))
for i in [0, int(Nt/4), int(Nt/2), Nt]:
    plt.plot(x, T_all[i], label=f't = {t[i]:.2f}')
plt.plot(x, T_exact(x, t[-1]), 'k--', label='T exact (final)')
plt.xlabel("x")
plt.ylabel("T(x,t)")
plt.title("Équation de la chaleur avec solution MMS (Crank-Nicolson)")
plt.legend()
plt.grid()
plt.show()
