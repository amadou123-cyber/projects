import random
import time
import matplotlib.pyplot as plt



def tri_insertion(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def tri_bulles(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def tri_rapide(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    gauche = [x for x in arr if x < pivot]
    milieu = [x for x in arr if x == pivot]
    droite = [x for x in arr if x > pivot]
    return tri_rapide(gauche) + milieu + tri_rapide(droite)

def tri_fusion(arr):
    if len(arr) <= 1:
        return arr
    
    milieu = len(arr) // 2
    gauche = tri_fusion(arr[:milieu])
    droite = tri_fusion(arr[milieu:])
    
    return fusion(gauche, droite)

def fusion(gauche, droite):
    resultat = []
    i = j = 0
    
    while i < len(gauche) and j < len(droite):
        if gauche[i] < droite[j]:
            resultat.append(gauche[i])
            i += 1
        else:
            resultat.append(droite[j])
            j += 1
    
    resultat.extend(gauche[i:])
    resultat.extend(droite[j:])
    
    return resultat

def tri_tas(arr):
    n = len(arr)
    
    for i in range(n // 2 - 1, -1, -1):
        entasser(arr, n, i)
    
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        entasser(arr, i, 0)
    
    return arr

def entasser(arr, n, i):
    plus_grand = i
    gauche = 2 * i + 1
    droite = 2 * i + 2
    
    if gauche < n and arr[gauche] > arr[plus_grand]:
        plus_grand = gauche
    
    if droite < n and arr[droite] > arr[plus_grand]:
        plus_grand = droite
    
    if plus_grand != i:
        arr[i], arr[plus_grand] = arr[plus_grand], arr[i]
        entasser(arr, n, plus_grand)
        

# Données aléatoires
def donnees_aleatoires(n):
    return [random.randint(0, 1000) for _ in range(n)]

# Données déjà triées
def donnees_triees(n):
    return list(range(n))

# Données triées en ordre décroissant
def donnees_triees_decroissant(n):
    return list(range(n, 0, -1))

# Données partiellement triées
def donnees_partiellement_triees(n):
    arr = list(range(n))
    for _ in range(n // 10):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def mesurer_temps(tri_fonction, arr):
    debut = time.time()
    tri_fonction(arr.copy())  # Utiliser une copie pour ne pas modifier l'original
    fin = time.time()
    return fin - debut

tailles = [100, 1000, 10000, 20000, 30000]
algorithmes = [tri_insertion, tri_bulles, tri_rapide, tri_fusion, tri_tas]
noms = ["Tri par Insertion", "Tri à Bulles", "Tri Rapide", "Tri Fusion", "Tri par Tas"]

resultats = {nom: [] for nom in noms}

for taille in tailles:
#    arr = donnees_aleatoires(taille)
#    arr = donnees_triees(taille)
    arr = donnees_triees_decroissant(taille)
#    arr = donnees_partiellement_triees(taille)
    
    for algo, nom in zip(algorithmes, noms):
        print(f"Algo {nom}\t Taille : {taille}")
        temps = mesurer_temps(algo, arr)
        resultats[nom].append(temps)

# Affichage des résultats
for nom, temps in resultats.items():
    print(f"{nom}: {temps}")
    

# Exemple de données
#tailles = [100, 1000, 10000]
temps_insertion = resultats["Tri par Insertion"] #[0.01, 0.5, 50]
temps_bulles = resultats["Tri à Bulles"]
temps_rapide = resultats["Tri Rapide"]
temps_fusion = resultats["Tri Fusion"]
temps_tas = resultats["Tri par Tas"]

# Tracer le graphique
plt.plot(tailles, temps_insertion, label="Tri par Insertion")
plt.plot(tailles, temps_bulles, label="Tri à Bulles")
plt.plot(tailles, temps_rapide, label="Tri Rapide")
plt.plot(tailles, temps_fusion, label="Tri Fusion")
plt.plot(tailles, temps_tas, label="Tri par Tas")
plt.xlabel("Taille des données")
plt.ylabel("Temps d'exécution (s)")
plt.legend()
plt.show()
