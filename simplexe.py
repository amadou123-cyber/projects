from numpy import zeros, argmin, inf, outer

def simplexe(c, A, b, J, K, itermax, B) :
    """
    Le code retournera plusieurs arguments :
    — un entier (qu’on appelera stat) donnant le status du calcul :
    1 si le minimum est atteint (couts réduits positifs), 
    2 si la fonction est non minor´ee sur le domaine D = {x ∈ Rn : Ax = b,x ≥ 0},
    3 si on a d´epass´e le nombre maximum d’it´erations (sans donc avoir détecté l’optimalité ou la non minoration de f);
   — la dernière solution de base obtenue (qui est donc la solution lorsque stat = 1);
   — la valeur de la fonction pour la derni`ere solution de base obtenue;
   — la dernière base obtenue (les indices de base J et ceux hors-base K);
   — l’inverse de la dernière matrice AJ ;
  — le nombre d’itérations
 """
    m, n = A.shape
    it = 0
    J = J.copy()
    K = K.copy()
    B = B.copy()
    
    while True :
        it += 1
        # étape 1 : calcul des coûts réduits
        cr = c[K] - (c[J]@B)@A[:,K]
        kstar = argmin(cr)
        if cr[kstar] >= 0.0 :
            # minimum atteint
            x = zeros(n)
            xJ = B @ b
            x[J] = xJ
            fmin = c[J] @ xJ
            return 1, x, fmin, J, K, B, it
        # la variable entrante est r = K[kstar]
        r = K[kstar]

        # étape 2 : calcul de la variable sortante
        xJ = B @ b  # calcul x[J]
        z = B @ A[:,r] 

        rap_min = inf; istar = None
        for i in range(m) :
            if z[i] > 0 :
                rap = xJ[i]/z[i]
                if rap < rap_min :
                    rap_min = rap; istar = i
        if istar is None :
            # fonction coût non minorée
            x = zeros(n)
            x[J] = xJ
            fmin = -inf
            return 2, x, fmin, J, K, B, it
        # la variable sortante est s = J[istar]
        s = J[istar]

        # étape 3 : mises à jour
        J[istar] = r; K[kstar] = s
        # calcul du nouvel inverse
        w = B@(A[:,s] - A[:,r])
        w = w/(1.0-w[istar])
        B = B + outer(w,B[istar,:])
                
        if it > itermax :
            # on renvoie là où on en est...
            x = zeros(n)
            xJ = B @ b
            x[J] = xJ
            fmin = c[J] @ xJ
            return 3, x, fmin, J, K, B, it
