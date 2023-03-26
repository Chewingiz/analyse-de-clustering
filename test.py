from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, DBSCAN
import pandas as pd
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, accuracy_score

    
def display_clusters(model, X, ax):
    # Récupérer les labels et les centres des clusters
    labels = model.labels_
    
    if hasattr(model, "cluster_centers_"): #vérifie si l'instance contient l'attribut "cluster_centers_"
        centers = model.cluster_centers_
    else:
        centers = None

    # Générer les couleurs pour chaque cluster
    num_clusters = len(np.unique(labels))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    color_map = {i: colors[i] for i in range(num_clusters)}

    # Assigner la couleur à chaque point
    point_colors = [color_map[labels[i]] for i in range(len(X))]

    # Afficher les points
    ax.scatter(X[:, 0], X[:, 1], c=point_colors, s=50)

    # Vérifier que les centres ont été calculés et les afficher
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], marker='*', c='black', s=100)

    return ax
    
    
def test_kmean(X, nb_cluster):
    start = time.time()
    kmeans = KMeans(n_clusters=nb_cluster, random_state=0, n_init="auto").fit(X)
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution kmean : {elapsed:.2}ms')
    #display_clusters(kmeans, X)
    return kmeans
    
def agglo(X, nb_cluster):
    start = time.time()
	# Instanciation de la classe AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=nb_cluster)

	# Entraînement du modèle sur les données
    model.fit(X)
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution agglo : {elapsed:.2}ms')   
    #display_agglomerative_clustering(model, X)
    return model
    
def MeanShift_test(X, bw):
    start = time.time()
	# Créer une instance de la classe MeanShift
    #ms = MeanShift(bandwidth=25) # donne 6 groupes
    #ms = MeanShift(bandwidth=30) # donne 4 groupes
    #ms = MeanShift(bandwidth=35) # donne 3 groupes très bon
    #ms = MeanShift(bandwidth=40) # donne 3 groupes
    ms = MeanShift(bandwidth=bw)
	# Entraînement du modèle sur les données
    ms.fit(X)
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution MeanShift : {elapsed:.2}ms')
    print(f'Temps d\'exécution MeanShift : {elapsed:.2f}s')
    #display_clusters(ms, X)
    return ms
	
	
def DBSCAN_test(X, eps, ms):
    start = time.time()
	# Création d'un objet DBSCAN
    #dbscan = DBSCAN(eps=20, min_samples=2) # marche bien
	#dbscan = DBSCAN(eps=25, min_samples=2) # 2 clsuters coller
    dbscan = DBSCAN(eps=eps, min_samples=ms)
    dbscan.fit(X)
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution DBSCAN : {elapsed:.2}ms')
    
    #display_agglomerative_clustering(dbscan, X)
    return dbscan
    
def nb_cluster_optimal(X):
    silhouette_scores = []
    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        labels = kmeans.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        silhouette_scores.append(silhouette)
    # Recherche du nombre optimal de clusters
    optimal_n_clusters = np.argmax(silhouette_scores) + 2
    print(f"Le nombre optimal de clusters est {optimal_n_clusters}")

    plt.plot(range(2, 11), silhouette_scores)
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Coefficient de silhouette")
    plt.title("Courbe de silhouette pour KMeans clustering")
    plt.show()

    return optimal_n_clusters


def main(title):
    data = pd.read_csv(title)
    X = data[['x', 'y']].values
    Color = data[['color']].values
    Color = [5 if x == 2 else x for x in Color]
    Color = [2 if x == 0 else x for x in Color]
    Color = [0 if x == 5 else x for x in Color]
    liste_flottants = list(map(float, Color))
    nb_cluster = nb_cluster_optimal(X)

    DBSCAN_1 = DBSCAN_test(X, 20, 2)#10,15 créé erreur affichage
    DBSCAN_2 = DBSCAN_test(X, 25, 2)
    DBSCAN_3 = DBSCAN_test(X, 30, 2)
    DBSCAN_4 = DBSCAN_test(X, 20, 3)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.canvas.manager.set_window_title("Comparaisons des résultats de DBSCAN en fonction de la valeur de l'eps")
    # Afficher le premier tableau de clustering k-means
    ax = axes[0, 0]
    ax.set_title("DBSCAN (eps = 20 et min_samples = 2)")
    display_clusters(DBSCAN_1, X, ax)

    # Afficher le deuxième tableau de clustering k-means
    ax = axes[0, 1]
    ax.set_title("DBSCAN (eps = 25 et min_samples = 2)")
    display_clusters(DBSCAN_2, X, ax)

    # Afficher le troisième tableau de clustering agglomératif
    ax = axes[1, 0]
    ax.set_title("DBSCAN (eps = 30 et min_samples = 2)")
    display_clusters(DBSCAN_3, X, ax)

    # Afficher le quatrième tableau de clustering agglomératif
    ax = axes[1, 1]
    ax.set_title("DBSCAN (eps = 20 et min_samples = 3)")
    display_clusters(DBSCAN_4, X, ax)

    plt.tight_layout()

    # Afficher la figure
    plt.show()
    


    kmeans = test_kmean(X, nb_cluster)
    agglomerative = agglo(X, nb_cluster)
    MeanShift = MeanShift_test(X, 35)
    DBSCAN = DBSCAN_test(X, 20, 2)

    silouette_kmeans = silhouette_score(X, kmeans.fit_predict(X))
    silouette_agglomerative = silhouette_score(X, agglomerative.labels_)
    silouette_MeanShift = silhouette_score(X, MeanShift.fit_predict(X))
    silouette_DBSCAN = silhouette_score(X, DBSCAN.fit_predict(X))

    print("silouette = " + str(silouette_kmeans) )
    print("silouette = " + str(silouette_agglomerative))
    print("silouette = " + str(silouette_MeanShift) )
    print("silouette = " + str(silouette_DBSCAN) )

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.canvas.manager.set_window_title("Comparaisons des méthodes de clusterisation")
    # Afficher le premier tableau de clustering k-means
    ax = axes[0, 0]
    ax.set_title(f"K-Means (k = {nb_cluster})")
    #print(kmeans.predict(X))
    #print("Score =" + str(accuracy_score(kmeans.predict(X) ,liste_flottants)))
    display_clusters(kmeans, X, ax)
    

    # Afficher le deuxième tableau de clustering k-means
    ax = axes[0, 1]
    ax.set_title("MeanShift (bandwidth = 35)")
    display_clusters(MeanShift, X, ax)
    #print(MeanShift.predict(X))
    #print("Score =" + str(accuracy_score(MeanShift.predict(X) ,liste_flottants)))

    # Afficher le troisième tableau de clustering agglomératif
    ax = axes[1, 0]
    ax.set_title(f"Agglomerative Clustering (k = {nb_cluster})")
    display_clusters(agglomerative, X, ax)
    #print(agglomerative.labels_)
    #print("Score =" + str(accuracy_score(agglomerative.labels_ ,liste_flottants)))

    # Afficher le quatrième tableau de clustering agglomératif
    ax = axes[1, 1]
    ax.set_title("DBSCAN  (eps = 20 et min_samples = 2)")
    display_clusters(DBSCAN, X, ax)
    #print(DBSCAN.fit_predict(X))
    #print("Score =" + str(accuracy_score(DBSCAN.fit_predict(X) ,liste_flottants)))

    plt.tight_layout()

    # Afficher la figure
    plt.show()
   



    MeanShift_1 = MeanShift_test(X, 25)
    MeanShift_2 = MeanShift_test(X, 30)
    MeanShift_3 = MeanShift_test(X, 35)
    MeanShift_4 = MeanShift_test(X, 40)
   

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.canvas.manager.set_window_title("Comparaisons des résultats de MeanShift en fonction de la valeur de bandwidth")
    # Afficher le premier tableau de clustering k-means
    ax = axes[0, 0]
    ax.set_title("MeanShift (bandwidth = 25)")
    display_clusters(MeanShift_1, X, ax)

    # Afficher le deuxième tableau de clustering k-means
    ax = axes[0, 1]
    ax.set_title("MeanShift (bandwidth = 30)")
    display_clusters(MeanShift_2, X, ax)

    # Afficher le troisième tableau de clustering agglomératif
    ax = axes[1, 0]
    ax.set_title("MeanShift (bandwidth = 35)")
    display_clusters(MeanShift_3, X, ax)

    # Afficher le quatrième tableau de clustering agglomératif
    ax = axes[1, 1]
    ax.set_title("MeanShift (bandwidth = 40)")
    display_clusters(MeanShift_4, X, ax)

    plt.tight_layout()

    # Afficher la figure
    plt.show()

    return 0

    


if __name__ == "__main__":
    try:
        print(sys.argv[1])
    except:
        print(f"Warning no file specified.")
        exit(1)
    main(sys.argv[1])

"""

    KMeans : cet algorithme divise les données en K clusters en minimisant la somme des distances au carré entre chaque point de données et le centroïde de son cluster. 
    C'est l'un des algorithmes de clustering les plus couramment utilisés.

    AgglomerativeClustering : cet algorithme commence par attribuer chaque point de données à son propre cluster, 
    puis fusionne de manière récursive les paires de clusters les plus proches jusqu'à ce que tous les points appartiennent à un seul cluster. 
    Il est souvent utilisé pour les données de type arborescentes.

    DBSCAN : cet algorithme est un algorithme de clustering de densité qui divise les données en clusters de haute densité séparés par des zones de faible densité. 
    Il est particulièrement utile pour les données avec des formes complexes et de tailles différentes.

    Birch : cet algorithme utilise une structure hiérarchique en forme d'arbre pour diviser les données en clusters. 
    Il est capable de gérer des ensembles de données volumineux et est souvent utilisé pour des tâches de clustering en temps réel.

    MeanShift : cet algorithme est un algorithme de clustering non paramétrique qui se base sur la densité pour trouver les centres de cluster. 
    Il est souvent utilisé pour des données de type image et pour des tâches de segmentation d'image.

    SpectralClustering : cet algorithme utilise les propriétés spectrales de la matrice de similarité pour diviser les données en K clusters. 
    Il est particulièrement utile pour les données non linéaires.

    OPTICS : cet algorithme est similaire à DBSCAN, mais il ne nécessite pas de spécifier à l'avance le nombre de clusters à créer. 
    Il est souvent utilisé pour des données de type spatial.
    
"""


"""from sklearn.cluster import KMeans
import pandas as pd
import sys

import numpy as np
import matplotlib.pyplot as plt

    
def display_clusters(kmeans_model, X):
    # Récupérer les labels et les centres des clusters
    labels = kmeans_model.labels_
    centers = kmeans_model.cluster_centers_

    # Générer les couleurs pour chaque cluster
    num_clusters = len(np.unique(labels))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    color_map = {i: colors[i] for i in range(num_clusters)}

    # Assigner la couleur à chaque point
    point_colors = [color_map[labels[i]] for i in range(len(X))]

    # Afficher les points et les centres de chaque cluster
    plt.scatter(X[:, 0], X[:, 1], c=point_colors, s=50)
    plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='black', s=1000)
    plt.show()
    
    
def test_kmean(data, nb_cluster):
    kmeans = KMeans(n_clusters=nb_cluster, random_state=0, n_init="auto").fit(data)
  	
    display_clusters(kmeans, data)
    return 0
    

    
def main( title ):
    data = pd.read_csv(title)
    test_kmean(data, 3)
    return 0
	#comment afficher les clusters trouver avec sklearn.cluster avec  des couleurs differente ?
if __name__ == "__main__":
    try:
        print(sys.argv[1])
    except:
        print(f"Warning no file specified.")
        exit(1)
    main(sys.argv[1])"""
