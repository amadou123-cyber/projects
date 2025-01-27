import numpy as np
import scipy.linalg as lin

A = np.array([[1,0],
              [1,1],
              [1,2],
              [1,3]])
b = np.array([-2,-1,0.5,1])

                          
# via une factorisation QR économique
Q, R = lin.qr(A, mode='economic')
# projection orthogonale de b sur Im(A) avec
proj_b = Q.T@b
# résolution du système triangulaire sup Rx=proj_b
x = lin.solve_triangular(R, proj_b) # plein d'autres options
res = A@x - b
print("\n Exercice 2")
print("\n Résolution via une factorisation QR économique, on obtient :")
print(f" x = {x}")
print(f" || Ax - b ||/||b|| = {lin.norm(res)/lin.norm(b)}")

# via une factorisation QR complète
Q, R = lin.qr(A)
# projection orthogonale de b sur Im(A) avec
proj_b = Q[:,:2].T@b
# résolution du système triangulaire sup Rx=proj_b
xb = lin.solve_triangular(R[:2,:], proj_b) # plein d'autres options
print("\n Résolution via une factorisation QR complète, on vérifie que :")
print(f" x - xb = {x-xb}")

#
B = A.transpose()
KerB = Q[:,2:]
print("\n Une b.o.n. du noyau de B est constituée des vecteurs :")
print(f" v1 = {KerB[:,0]} \n v2 = {KerB[:,1]}")

# qq essais
coord = np.random.random(2)
y1 = KerB@coord
By1 = B @ y1
print(f"\n Avec y1 = {coord[0]} v1 + {coord[1]} v2, on obtient :") 
print(f" By1 = {By1}")
print(f" ||By1||/||y1|| = {lin.norm(By1)/lin.norm(y1)}")

y2 = 2*KerB[:,0] - KerB[:,1]
By2 = B @ y2
print("\n Avec y2 = 2 v1 - v2, on obtient :") 
print(f" By2 = {By2}")
print(f" ||By2||/||y2|| = { lin.norm(By2)/lin.norm(y2)}")
