import numpy as np
from scipy.optimize import minimize_scalar
import pdb

def affichage(titre, xc, fc, gc, cvg, tolg, tolx, norm_gc, errx, it):
    """ fonction utilitaire pour résumer une optimisation

    titre : chaîne de caractères
    xc : point terminal obtenu par optimisation
    fc : valeur de la fonction en xc
    gc : gradient de la fonction en xc
    cvg : booléen (succès ou échec de l'optimisation)
    tolg : tolérance sur ||gc||
    tolx : tolérance sur la stationnarité de 2 itérés successifs
    norm_gc : ||gc||
    errx :  || xc - x(it-1) ||/ max(||xc||, ||x(it-1)||, 1e-3) 
    it : nb d'itérations
    """
    print("\n*******************************************")
    print(titre)
    print("-------------------------------------------")
    print("'Minimum' obtenu : xc= ", xc)
    print(f"f(xc) = {fc}")
    print("f'(xc) = ",gc)    
    if cvg:
        print(f"-> convergence obtenue en {it} itérations")
    else:
        print(f"-> pas de convergence en {it} itérations")        
    print(f" || f'(xc) || = {norm_gc} (tolg = {tolg})")
    print(f" || xc - x(it-1) ||/ max(||xc||, ||x(it-1)||, 1e-3) = {errx} (tolx = {tolx})")
    print("*******************************************\n")    
    
def quad(x, A, b):
    """ retourne 1/2 <Ax,x> - <b,x>
         x et b doivent être des tableaux numpy de profil (n,)
         A un tableau numpy de profil (n,n)
    """
    return 0.5*((A@x)@x) - b@x

def grd_quad(x, A, b):
    """ retourne le gradient de quad
         x et b doivent être des tableaux numpy de profil (n,)
         A un tableau numpy de profil (n,n)
    """
    return A@x - b

def rosenbrock(x, C):
    """ fonction de Rosenbrock à deux variables avec paramètre C """
    return C*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def grd_rosenbrock(x, C):
    """ gradient de la fonction précédente """
    temp = 2*C*(x[1]- x[0]**2)
    return np.array( [-2*(temp*x[0] + (1 - x[0])), temp] ) 
    
def eval_sur_grille(a, b, nx, c, d, ny, f, fparam=()):
    """ fonction utilitaire pour représenter la fonction f sur le rectangle [a,b]x[c,d]
    a, b, c, d : scalaires réels
    nx : nombre de points à prendre sur [a,b] (= nb d'intervalles plus 1)
    ny : nombre de points à prendre sur [c,d] (= nb d'intervalles plus 1)
    f : fonction à représenter graphiquement
    fparam : paramètres supplémentaires de la fonction f
   
    retourne : 
    x : maillage de [a,b]
    y : maillage de [c, d]
    Z : valeurs de f sur les points de la grille ainsi définie (Z[j,i] = f(x[i],y[j])
    """
    x = np.linspace(a,b,nx)
    y = np.linspace(c,d,ny)
    Z = np.empty((ny,nx), dtype=float)
    xy = np.empty(2, dtype=float)
    for j, yj in enumerate(y):
        xy[1] = yj
        for i, xi in enumerate(x):
            xy[0] = xi 
            Z[j,i] = f(xy, *fparam)
    return x, y, Z

def gradient_pas_fixe(f, grd_f, x0, tau, tolg=1e-3, tolx=1e-6, itermax=100,
                      verb=0, record=False, fparam=()):
    """ méthode de gradient à pas fixe
        f : fonction à minimiser
        grd_f : son gradient
        f_param : tuple (éventuels paramètres de f et grd_f)
        x0 : point de départ
        tau : pas
        tolg : tolérance sur le gradient
        tolx : tolérance sur la stationnarité de deux itérés successifs
        itermax : nb max d'itérations
        verb : affichage de f(xc) et ||grad_f(xc)|| toutes les verb itérations
               (aucun affichage si verb <= 0)
        record : si True on enregistre les itérés successifs dans une liste

    La fonction doit retourner les arguments suivants : xc, fc, gc, cvg, it, err_g, err_x, L
        xc : valeur finale obtenue à la suite des itérations
        fc : valeur de la fonction f en xc
        gc : gradient de f en xc
        cvg : booléen True si au moins l'un des 2 critères de convergence est obtenu
        it : nombre d'itérations effectuées
        norm_gc : ||gc||
        err_x : ||xc-xp|| /  max(|xc||, ||xp||, 1e-3) (xp étant l'itéré précédent)
        L : si record est True liste des itérés (x0 compris) sinon une liste vide
    """
    it = 0   # numéro d'itération
    xc = x0  # xc itéré courant
    norm_xc = np.linalg.norm(xc)  # ||xc|| utile pour le 2 ème test
    gc = grd_f(xc, *fparam)  # gradient de f en xc
    norm_gc = np.linalg.norm(gc)
    
    L = []   # liste des itérés
    if record:     
        L.append(xc)
        
    while True:

        # affichage éventuel
        if verb > 0 and it % verb == 0 :
            fc = f(xc,*fparam)
            print(f"itération n° {it}: f = {fc}, ||g|| = {norm_gc}") 

        # l'itération de gradient à pas fixe
        xn = xc - tau*gc
        it += 1
        
        if record:  # ajout de xn
            L.append(xn)
        
        gc = grd_f(xn, *fparam)  # calcul du nouveau gradient
        
        # tests de cvg
        norm_gc = np.linalg.norm(gc)
        norm_xn = np.linalg.norm(xn) 
        err_x = np.linalg.norm(xc - xn)/max(1e-3, norm_xc, norm_xn)
        # mise à jour de xc et norm_xc
        xc, norm_xc = xn, norm_xn
        if norm_gc <= tolg or err_x <= tolx :
            cvg = True
            break
        elif it >= itermax:
            cvg = False
            break

    fc = f(xc, *fparam)
    return xc, fc, gc, cvg, it, norm_gc, err_x, L

def gradient_pas_optimal(f, grd_f, x0, f_quad=False, tolg=1e-3, tolx=1e-6, itermax=100,
                         verb=0, record=False, fparam=()):
    """ méthode de gradient à pas optimal
        f : fonction à minimiser
        grd_f : son gradient
        f_param : tuple (éventuels paramètres de f et grd_f)
        x0 : point de départ
        f_quad : True si la fct à minimiser est quadratique
        tolg : tolérance sur le gradient
        tolx : tolérance sur la stationnarité de deux itérés successifs
        itermax : nb max d'itérations
        verb : affichage de f(xc) et ||grad_f(xc)|| toutes les verb itérations
               (aucun affichage si verb <= 0)
        record : si True on enregistre les itérés successifs dans une liste

    La fonction doit retourner les arguments suivants : xc, fc, gc, cvg, it, err_g, err_x, L
        xc : valeur finale obtenue à la suite des itérations
        fc : valeur de la fonction f en xc
        gc : gradient de f en xc
        cvg : booléen True si au moins l'un des 2 critères de convergence est obtenu
        it : nombre d'itérations effectuées
        norm_gc : ||gc||
        err_x : ||xc-xp|| /  max(|xc||, ||xp||, 1e-3) (xp étant l'itéré précédent)
        L : si record est True liste des itérés (x0 compris) sinon une liste vide
    """
    it = 0   # numéro d'itération
    xc = x0  # xc itéré courant
    norm_xc = np.linalg.norm(xc)  # ||xc|| utile pour le 2 ème test
    gc = grd_f(xc, *fparam)  # gradient de f en xc
    norm_gc = np.linalg.norm(gc)
    err_x = np.nan    # pas encore défini

    if not f_quad:
        # on définit la fonction unidimensionnelle à fournir à minimize_scalar
        def hfunc(theta, x, d, fparam):
            return f(x + theta*d, *fparam)

    L = []   # liste des itérés
    if record:     
        L.append(xc)
    # il faut définir err_g et  
        
    while True:

        # affichage éventuel
        if verb > 0 and it % verb == 0 :
            fc = f(xc,*fparam)
            print(f"itération n° {it}: f = {fc}, ||g|| = {norm_gc}") 

        # l'itération de gradient à pas optimal
        if f_quad :  # fct quadratique
            A = fparam[0]
            tau = gc @ gc / ((A@gc)@gc)
        else:        # on utilise minimize scalar
            res = minimize_scalar(hfunc, args=(xc, -gc, fparam))
            if res.success :
                tau = res.x
            else:
                cvg = False
                print(res.message)
                break
            
        xn = xc - tau*gc
        it += 1
        
        if record:  # ajout de xn
            L.append(xn)
        
        gc = grd_f(xn, *fparam)  # calcul du nouveau gradient
        
        # tests de cvg
        norm_gc = np.linalg.norm(gc)
        norm_xn = np.linalg.norm(xn) 
        err_x = np.linalg.norm(xc - xn)/max(1e-3, norm_xc, norm_xn)
        # mise à jour de xc et norm_xc
        xc, norm_xc = xn, norm_xn
        if norm_gc <= tolg or err_x <= tolx :
            cvg = True
            break
        elif it >= itermax:
            cvg = False
            break

    fc = f(xc, *fparam)
    return xc, fc, gc, cvg, it, norm_gc, err_x, L


def gradient_conjugue(f, grd_f, x0, f_quad=False, tolg=1e-3, tolx=1e-6, itermax=100,
                      verb=0, record=False, fparam=()):
    """ méthode de gradient conjugué
        f : fonction à minimiser
        grd_f : son gradient
        f_param : tuple (éventuels paramètres de f et grd_f)
        x0 : point de départ
        f_quad : True si la fct à minimiser est quadratique
        tolg : tolérance sur le gradient
        tolx : tolérance sur la stationnarité de deux itérés successifs
        itermax : nb max d'itérations
        verb : affichage de f(xc) et ||grad_f(xc)|| toutes les verb itérations
               (aucun affichage si verb <= 0)
        record : si True on enregistre les itérés successifs dans une liste

    La fonction doit retourner les arguments suivants : xc, fc, gc, cvg, it, err_g, err_x, L
        xc : valeur finale obtenue à la suite des itérations
        fc : valeur de la fonction f en xc
        gc : gradient de f en xc
        cvg : booléen True si au moins l'un des 2 critères de convergence est obtenu
        it : nombre d'itérations effectuées
        norm_gc : ||gc||
        err_x : ||xc-xp|| /  max(|xc||, ||xp||, 1e-3) (xp étant l'itéré précédent)
        L : si record est True liste des itérés (x0 compris) sinon une liste vide
    """
    it = 0   # numéro d'itération
    xc = x0  # xc itéré courant
    norm_xc = np.linalg.norm(xc)  # ||xc|| utile pour le 2 ème test
    gc = grd_f(xc, *fparam)  # gradient de f en xc
    norm_gc = np.linalg.norm(gc)
    err_x = np.nan    # pas encore défini
    d = -gc  # direction de descente initiale
    
    if not f_quad:
        # on définit la fonction unidimensionnelle à fournir à minimize_scalar
        def hfunc(theta, x, d, fparam):
            return f(x + theta*d, *fparam)

    L = []   # liste des itérés
    if record:     
        L.append(xc)
        
    while True:

        # 0- affichage éventuel
        if verb > 0 and it % verb == 0 :
            fc = f(xc,*fparam)
            print(f"itération n° {it}: f = {fc}, ||g|| = {norm_gc}") 

        # 1- phase de minimisation unidimensionnelle
        if f_quad :  # fct quadratique
            A = fparam[0]
            tau = - d @ gc / ((A@d)@d)
        else:        # on utilise minimize scalar
            res = minimize_scalar(hfunc, args=(xc, d, fparam))
            if res.success :
                tau = res.x
            else:
                cvg = False
                print(res.message)
                break
            
        # 2- calcul du nouvel itéré    
        xn = xc + tau*d
        it += 1
        
        if record:  # ajout de xn
            L.append(xn)

        # 3- calcul du nouveau gradient et de la direction de descente
        gn = grd_f(xn, *fparam)
        beta = (gn - gc)@gn / norm_gc**2
        d = -gn + beta*d
        gc = gn
        
        # 4- tests de cvg
        norm_gc = np.linalg.norm(gc)
        norm_xn = np.linalg.norm(xn) 
        err_x = np.linalg.norm(xc - xn)/max(1e-3, norm_xc, norm_xn)
        # mise à jour de xc et norm_xc
        xc, norm_xc = xn, norm_xn
        if norm_gc <= tolg or err_x <= tolx :
            cvg = True
            break
        elif it >= itermax:
            cvg = False
            break

    fc = f(xc, *fparam)
    return xc, fc, gc, cvg, it, norm_gc, err_x, L


def penalisation(f, grd_f, c, J_c, x0, Mu, meth = gradient_pas_optimal, tolg=1e-3,
                 tolx=1e-6, itermax=100, verb=0, fparam=(), cparam=()):
    """ méthode de pénalisation pour résoudre le problème d'optimisation

        Min_{ x \in D } f(x) où D = { x \in R^n : c_i(x) <= 0, i = 0, ..., m-1 }

        Le domaine est déterminé par m fonctions c_i.
    
        On remplace le pb d'optimisation avec contrainte, par une succession de pb d'optimisation sans contraintes :

          Min_x fmu(x) := f(x) + (1/2) mu || ct(x) ||^2 où ct_i(x) = max(0, c_i(x))

        avec mu = Mu[0], puis mu = Mu[1], puis mu = Mu[2], etc.... (cf paramètre Mu)
    
        Le gradient de fmu(x) est :

             grad_fmu = grad_f +  mu (J_c)^T ct(x)
    
        f : fonction à minimiser
        grd_f : son gradient
        f_param : tuple (éventuels paramètres de f et grd_f)
        c : contraintes (on considère que le domaine est défini par c_i(x) <= 0, forall i=0,.., m-1)
        J_c : jacobienne (la ligne i de J_c est le gradient de c_i(x)) J_c(x) est une matrice (m,n)
        cparam : tuple (éventuels paramètres de c et de J_c)
        x0 : point de départ
        Mu : liste ou tuple ou tableau numpy : suite des paramètres mu à utiliser (donne aussi le nombre
             d'itérations pour cette méthode)
        meth : fonction à utiliser pour résoudre les pbs d'optimisation sans contraintes successsifs
        tolg : tolérance sur le gradient (pour meth)
        tolx : tolérance sur la stationnarité de deux itérés successifs (pour meth)
        itermax : nb max d'itérations
        verb : pour affichage dans meth  (affichage de fmu(xc) et ||grad_fmu(xc)|| toutes les verb itérations
               (aucun affichage si verb <= 0)

    La fonction doit retourner pour le moment : xc
        xc : valeur finale obtenue à la suite des itérations
        fc : valeur de la fonction f en xc
        gc : gradient de f en xc
        c : valeur des contraintes
    """
    # définition de la fonction à minimiser pour les pbs sans contraintes
    # param devra être formé par (mu, f, grd_f, fparam, c, J_c, cparam)
    def fmu(x, mu, f, grd_f, fparam, c, J_c, cparam):
        return f(x, *fparam) + 0.5*mu*p(x, c, cparam)
    # définition de son gradient
    def grd_fmu(x, mu, f, grd_f, fparam, c, J_c, cparam):
        return grd_f(x, *fparam) + mu*( np.maximum(c(x, *cparam), 0) @ J_c(x, *cparam))
    # la fonction de pénalisation
    def p(x, c, cparam):
        return np.sum(np.maximum(c(x, *cparam), 0)**2)

    
    # tous les paramètres qui ne changent pas dans un tuple 
    t = (f, grd_f, fparam, c, J_c, cparam)

    print("\n Point de départ :")
    print(" ----------------")
    print(f"  x0 = {x0}")
    print(f"  f(x0) = {f(x0,*fparam)}")
    print(f"  p(x0) = {p(x0, c, cparam)}")                

    xc = x0
    for mu in Mu:
        param = (mu, *t)
        print(f"\n Itération avec mu = {mu}")
        print(" -------------------------")
        xc, fc, gc, cvg, it, norm_gc, err_x, _ = meth(fmu, grd_fmu, xc, f_quad=False, tolg=tolg, tolx=tolx,
                                                      itermax=itermax, verb=verb, record=False, fparam=param)
        print(f"  cvg = {cvg}, norm_gc = {norm_gc}, err_x = {err_x}, it = {it}")
        print(f"  xc = {xc}")
        print(f"  f_mu(xc) = {fc}")
        print(f"  p(xc) = {p(xc, c, cparam)}")                
        print(f"  f(xc) = {f(xc,*fparam)}")

    return xc, f(xc, *fparam), grd_f(xc, *fparam), c(xc, *cparam)

def pc2_cstr(xx, r):
    x, y = xx
    return np.array([1-x, y - 1, r**2 - (x-1)**2 - (y-1)**2])

def pc2_J_cstr(xx, r):
    x, y = xx
    return np.array([[-1, 0],
                     [0,  1],
                     [-2*(x-1), -2*(y-1)]])

x0 = np.array([-1, 1])
Mu = [1, 10, 100, 1000, 10000, 100000, 1000000]


C = 10
r = 0.05

print("\n********************************")
print(" Exemple 2 : test 1")

x1, fx1, gx1, cx1 = penalisation(rosenbrock, grd_rosenbrock, pc2_cstr, pc2_J_cstr, x0,
                             Mu, fparam=(C,), cparam=(r,), meth=gradient_conjugue)


print("\n********************************")
print(" Exemple 2 : test 2 (où on repart à partir du point obtenu suite au test 1")
print("                       avec 3 valeurs du paramètre de pénalisation plus élevée)")
# on continue le calcul avec des valeurs plus élevée
Mu = [1e7, 1e8, 1e9]
x2, fx2, gx2, cx2 = penalisation(rosenbrock, grd_rosenbrock, pc2_cstr, pc2_J_cstr, x1,
                             Mu, fparam=(C,), cparam=(r,), meth=gradient_conjugue)

print("\n********************************")
print(" Exemple 2 : test 2 bis (idem au test 2 mais avec le paramètre tolx plus petit)")
x2bis, fx2bis, gx2bis, cx2bis = penalisation(rosenbrock, grd_rosenbrock, pc2_cstr, pc2_J_cstr, x1,
                             Mu, fparam=(C,), cparam=(r,), meth=gradient_conjugue, tolx=1e-16)
