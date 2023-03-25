from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, DBSCAN
import pandas as pd
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

    
def display_kmeans(kmeans_model, X):
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
    

def display_agglomerative_clustering(model, X):
    # Récupérer les labels des clusters
    labels = model.fit_predict(X)

    # Générer les couleurs pour chaque cluster
    num_clusters = len(np.unique(labels))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    color_map = {i: colors[i] for i in range(num_clusters)}

    # Assigner la couleur à chaque point
    point_colors = [color_map[labels[i]] for i in range(len(X))]

    # Afficher les points de chaque cluster
    plt.scatter(X[:, 0], X[:, 1], c=point_colors, s=50)
    plt.show()

    
def test_kmean(X, nb_cluster):
    start = time.time()
    kmeans = KMeans(n_clusters=nb_cluster, random_state=0, n_init="auto").fit(X)
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution kmean : {elapsed:.2}ms')
    display_kmeans(kmeans, X)
    return 0
    
def aglo(X, nb_cluster):
    start = time.time()
	# Instanciation de la classe AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=nb_cluster)

	# Entraînement du modèle sur les données
    model.fit(X)
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution aglo : {elapsed:.2}ms')   
    display_agglomerative_clustering(model, X)
    
def MeanShift_test(X):
    start = time.time()
	# Créer une instance de la classe MeanShift
    #ms = MeanShift(bandwidth=25) # donne 6 groupes
    #ms = MeanShift(bandwidth=30) # donne 4 groupes
    ms = MeanShift(bandwidth=35) # donne 3 groupes
    #ms = MeanShift(bandwidth=40) # donne 3 groupes
	# Entraînement du modèle sur les données
    ms.fit(X)
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution MeanShift : {elapsed:.2}ms')
    print(f'Temps d\'exécution MeanShift : {elapsed:.2f}s')
    display_kmeans(ms, X)
	
	
def DBSCAN_test(X):
    start = time.time()
	# Création d'un objet DBSCAN
    dbscan = DBSCAN(eps=20, min_samples=2) # marche bien
	#dbscan = DBSCAN(eps=25, min_samples=2) # 2 clsuters coller
    dbscan.fit(X)
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution DBSCAN : {elapsed:.2}ms')
    
    display_agglomerative_clustering(dbscan, X)
    
def main(title):
    data = pd.read_csv(title)
    X = data[['x', 'y']].values
    #test_kmean(X, 3)
    #aglo(X, 3)
    MeanShift_test(X)
    #DBSCAN_test(X)
    return 0


if __name__ == "__main__":
    try:
        print(sys.argv[1])
    except:
        print(f"Warning no file specified.")
        exit(1)
    main(sys.argv[1])
"""

    KMeans : cet algorithme divise les données en K clusters en minimisant la somme des distances au carré entre chaque point de données et le centroïde de son cluster. C'est l'un des algorithmes de clustering les plus couramment utilisés.

    AgglomerativeClustering : cet algorithme commence par attribuer chaque point de données à son propre cluster, puis fusionne de manière récursive les paires de clusters les plus proches jusqu'à ce que tous les points appartiennent à un seul cluster. Il est souvent utilisé pour les données de type arborescentes.

    DBSCAN : cet algorithme est un algorithme de clustering de densité qui divise les données en clusters de haute densité séparés par des zones de faible densité. Il est particulièrement utile pour les données avec des formes complexes et de tailles différentes.

    Birch : cet algorithme utilise une structure hiérarchique en forme d'arbre pour diviser les données en clusters. Il est capable de gérer des ensembles de données volumineux et est souvent utilisé pour des tâches de clustering en temps réel.

    MeanShift : cet algorithme est un algorithme de clustering non paramétrique qui se base sur la densité pour trouver les centres de cluster. Il est souvent utilisé pour des données de type image et pour des tâches de segmentation d'image.

    SpectralClustering : cet algorithme utilise les propriétés spectrales de la matrice de similarité pour diviser les données en K clusters. Il est particulièrement utile pour les données non linéaires.

    OPTICS : cet algorithme est similaire à DBSCAN, mais il ne nécessite pas de spécifier à l'avance le nombre de clusters à créer. Il est souvent utilisé pour des données de type spatial.
    
"""


"""from sklearn.cluster import KMeans
import pandas as pd
import sys

import numpy as np
import matplotlib.pyplot as plt

    
def display_kmeans(kmeans_model, X):
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
  	
    display_kmeans(kmeans, data)
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
