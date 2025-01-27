import numpy as np
import scipy.linalg as lin
  
# on définit la matrice A
A = np.array([[1, 2],
              [2, 1],
              [1,-1]])
  
# pour la DVS ``économique'' : U est (m,K), s est 1d de taille K et VT est (K,n)
U, s, VT = lin.svd(A, full_matrices=False)

print("\n Exercice 4")

print(f"\n DVS de la matrice : A = \n {A}")
print(f"\n U = \n {U}")
print(f"\n valeurs singulières : s = \n {s}")
print(f"\n V = \n {VT.T}")

# pour la DVS complète : U est (m,m), s est 1d de taille K et VT est (n,n)
# on utilise full_matrices=True ou simplement U, s, VT = lin.svd(A)
  
# calcul de la pseudo-inverse à partir de la svd
Apinv =  ((U/s)@VT).T  # ou (VT.T/s)@U.T  

print(f"\n Pseudo inverse via DVS : Apinv = \n {Apinv}")

# calcul direct (les opérations précédentes sont effectuées en fait)
Apinv_bis = lin.pinv(A)
print(f"\n Pseudo inverse via pinv : Apinv_bis = \n {Apinv_bis}")

print(f"\n Apinv - Apinv_bis = \n {Apinv-Apinv_bis}")

b = np.array([1,2,3])
x = ((b@U)/s)@VT   # ou x =  VT.T@((U.T@b)/s) 
x_bis = Apinv@b

print("\n Résolution de Ax = b (au sens mdc) :") 
print(f"\n via DVS : x = \n {x}")
print(f"\n via pinv : x_bis = \n {x_bis}")

print(f"\n")
