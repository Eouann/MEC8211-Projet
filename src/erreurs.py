"""
Fichier de calcul des erreurs L1, L2 et Linf discrètes
"""


#Importation de bibliothèques
import numpy as np


#Erreur L1 discrète
def ErreurL1(Ti,Ti_exact,N_spatial,N_temporel):
    """Fonction de calcul de l'erreur L1 discrète"""
    normeL1 = 1/(N_spatial*N_temporel)*np.sum(np.abs(Ti-Ti_exact))
    return normeL1


#Erreur L2 discrète
def ErreurL2(Ti,Ti_exact,N_spatial,N_temporel):
    """Fonction de calcul de l'erreur L2 discrète"""
    normeL2 = np.sqrt(1/(N_spatial*N_temporel)*np.sum(np.abs(Ti-Ti_exact)**2))
    return normeL2


#Erreur Linf discrète
def ErreurLinf(Ti,Ti_exact):
    """Fonction de calcul de l'erreur Linf discrète"""
    normeLinf = np.max(np.abs(Ti-Ti_exact))
    return normeLinf
