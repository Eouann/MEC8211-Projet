"""
Fichier de calcul des erreurs L1, L2 et Linf discrètes
"""


#Importation de bibliothèques
import numpy as np


#Erreur L1 discrète
def ErreurL1(T_i_n,T_i_n_manufacturee,N_spatial,N_temporel):
    """Fonction de calcul de l'erreur L1 discrète"""
    L1=0
    for t in range(N_temporel):
        for x in range(N_spatial):
            L1+=np.abs(T_i_n[t,x]-T_i_n_manufacturee[t,x])
    normeL1 = 1/(N_spatial*N_temporel)*L1
    return normeL1


#Erreur L2 discrète
def ErreurL2(T_i_n,T_i_n_manufacturee,N_spatial,N_temporel):
    """Fonction de calcul de l'erreur L2 discrète"""
    L2=0
    for t in range(N_temporel):
        for x in range(N_spatial):
            L2+=np.abs(T_i_n[t,x]-T_i_n_manufacturee[t,x])**2
    normeL2 = np.sqrt(1/(N_spatial*N_temporel)*L2)
    return normeL2


#Erreur Linf discrète
def ErreurLinf(T_i_n,T_i_n_manufacturee):
    """Fonction de calcul de l'erreur Linf discrète"""
    normeLinf = np.max(np.abs(T_i_n-T_i_n_manufacturee))
    return normeLinf
