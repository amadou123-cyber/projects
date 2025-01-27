import numpy as np
import scipy.linalg as lin

print("\n Test 1:")

A = np.array([[1,0],
              [1,1],
              [1,2],
              [1,3]])
b = np.array([-2,-1,0.5,1])

                          
# on construit la matrice G = A^T A et le vecteur h = A^T b
G = A.T @ A
h = A.T @ b

# 2 possibilités :

# (i) calcul de x solution de G x = h ``directement''
x = lin.solve(G, h, assume_a='pos')
res = A@x-b
print(f" solution : x = {x}")
print(f" norme résidu = || Ax - b ||/||b|| = {lin.norm(res)/lin.norm(b)}")

# (ii) on peut aussi d'abord factoriser G par Cholesky G = C*C^T
# puis résoudre les 2 systèmes triangulaires ensuite
# (utile dans certains cas, par exemple en cas de différents b
# pas connus en même temps)
Gfact = lin.cho_factor(G)   # factorisation
xb = lin.cho_solve(Gfact, h) # résolution
resb = A@xb-b
print(f" solution : xb = {xb}")
print(f" norme résidu = || A xb - b ||/||b|| = {lin.norm(resb)/lin.norm(b)}")
print(f" ||x-xb||/||xb|| = {lin.norm(x-xb)/lin.norm(xb)} \n")



