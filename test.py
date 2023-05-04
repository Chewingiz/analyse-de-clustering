from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, DBSCAN
import pandas as pd
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import silhouette_score, accuracy_score, calinski_harabasz_score

    
def display_clusters(model, X, ax):
    # Récupérer les labels et les centres des clusters
    labels = model.labels_
    
    if hasattr(model, "cluster_centers_"): #vérifie si l'instance contient l'attribut "cluster_centers_"
        centers = model.cluster_centers_
    else:
        centers = None

    # Générer les couleurs pour chaque cluster
    num_clusters = len(np.unique(labels))

    colors_generation = mpl.colormaps.get_cmap('tab10')  # , num_clusters
    colors = list(colors_generation.colors)
    color_map = {i: colors[i] for i in range(num_clusters)}
    
    # Assigner la couleur à chaque point
    #point_colors = [color_map[labels[i]] for i in range(len(X))]
    point_colors = []
    for i in range(len(X)):
        v = labels[i]
        if v == -1 : 
            # permet d'afficher les points non affecter à des clusters en noir
            point_colors.append("black")
        else :
            point_colors.append(color_map[v])

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
    return model
    
def MeanShift_test(X, bw):
    start = time.time()
	# Créer une instance de la classe MeanShift
    ms = MeanShift(bandwidth=bw)
	# Entraînement du modèle sur les données
    ms.fit(X)
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution MeanShift : {elapsed:.2}ms')
    print(f'Temps d\'exécution MeanShift : {elapsed:.2f}s')
    return ms
	
	
def DBSCAN_test(X, eps, ms):
    start = time.time()
	# Création d'un objet DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=ms)
    dbscan.fit(X)
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution DBSCAN : {elapsed:.2}ms')
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

def qualite_clusters_silhouette():
	silhouette_score(X, labels)
def main(title, bw = 35 , eps = 20, ms = 2):
    data = pd.read_csv(title)
    X = data[['x', 'y']].values
    nb_cluster = nb_cluster_optimal(X)
 
    print("\nComparaisons des méthodes de clusterisation :")
    print("...Attendez...")
    kmeans = test_kmean(X, nb_cluster)
    agglomerative = agglo(X, nb_cluster)
    MeanShift = MeanShift_test(X, bw)
    DBSCAN = DBSCAN_test(X, eps, ms)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.canvas.manager.set_window_title("Comparaisons des méthodes de clusterisation")

    ax = axes[0, 0]
    ax.set_title(f"K-Means (k = {nb_cluster})")
    display_clusters(kmeans, X, ax)
    
    ax = axes[0, 1]
    ax.set_title(f"MeanShift (bandwidth = {bw})")
    display_clusters(MeanShift, X, ax)

    ax = axes[1, 0]
    ax.set_title(f"Agglomerative Clustering (k = {nb_cluster})")
    display_clusters(agglomerative, X, ax)

    ax = axes[1, 1]
    ax.set_title(f"DBSCAN  (eps = {eps} et min_samples = {ms})")

    display_clusters(DBSCAN, X, ax)
    plt.tight_layout()
    plt.show()
    
    print("Qualité des clusters:")
    print("Silhouette :")
    print("Kmeans :"        + str(silhouette_score(X, kmeans.fit_predict(X)          )))
    print("agglomerative :" + str(silhouette_score(X, agglomerative.fit_predict(X)   )))
    print("MeanShift :"     + str(silhouette_score(X, MeanShift.fit_predict(X)       )))
    #print("DBSCAN : "       + str(silhouette_score(X, DBSCAN.fit_predict(X)          )))

    print("Calinski Harabasz :")
    print("Kmeans :"        + str(calinski_harabasz_score(X, kmeans.fit_predict(X)          )))
    print("agglomerative :" + str(calinski_harabasz_score(X, agglomerative.fit_predict(X)   )))
    print("MeanShift :"     + str(calinski_harabasz_score(X, MeanShift.fit_predict(X)       )))
   # print("DBSCAN : "       + str(calinski_harabasz_score(X, DBSCAN.fit_predict(X)          )))

    print("\nComparaisons des résultats de DBSCAN en fonction de la valeur de l'eps :")
    print("...Attendez...")
    DBSCAN_1 = DBSCAN_test(X, eps, ms)
    DBSCAN_2 = DBSCAN_test(X, eps+5, ms)
    DBSCAN_3 = DBSCAN_test(X, eps+10, ms)
    DBSCAN_4 = DBSCAN_test(X, eps, ms+1)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.canvas.manager.set_window_title("Comparaisons des résultats de DBSCAN en fonction de la valeur de l'eps")
    
    ax = axes[0, 0]
    ax.set_title(f"DBSCAN (eps = {eps} et min_samples = {ms})")
    display_clusters(DBSCAN_1, X, ax)
 
    ax = axes[0, 1]
    ax.set_title(f"DBSCAN (eps = {eps+5} et min_samples = {ms})")
    display_clusters(DBSCAN_2, X, ax)
 
    ax = axes[1, 0]
    ax.set_title(f"DBSCAN (eps = {eps+10} et min_samples = {ms})")
    display_clusters(DBSCAN_3, X, ax)

    ax = axes[1, 1]
    ax.set_title(f"DBSCAN (eps = {eps} et min_samples = {ms+1})")
    display_clusters(DBSCAN_4, X, ax)

    plt.tight_layout()
    plt.show()
    
    print("\nComparaisons des résultats de MeanShift en fonction de la valeur de bandwidth :")
    print("...Attendez...")
    MeanShift_1 = MeanShift_test(X, bw-10)
    MeanShift_2 = MeanShift_test(X, bw-5)
    MeanShift_3 = MeanShift_test(X, bw)
    MeanShift_4 = MeanShift_test(X, bw+5)
   

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.canvas.manager.set_window_title("Comparaisons des résultats de MeanShift en fonction de la valeur de bandwidth")

    ax = axes[0, 0]
    ax.set_title(f"MeanShift (bandwidth = {bw-10})")
    display_clusters(MeanShift_1, X, ax)

    ax = axes[0, 1]
    ax.set_title(f"MeanShift (bandwidth = {bw-5})")
    display_clusters(MeanShift_2, X, ax)

    ax = axes[1, 0]
    ax.set_title(f"MeanShift (bandwidth = {bw})")
    display_clusters(MeanShift_3, X, ax)

    ax = axes[1, 1]
    ax.set_title(f"MeanShift (bandwidth = {bw+5})")
    display_clusters(MeanShift_4, X, ax)

    plt.tight_layout()
    plt.show()

    return 0

    


if __name__ == "__main__":
    try:
        print(sys.argv[1])
    except:
        print(f"Warning no file specified.")
        exit(1)

    if (len(sys.argv) == 5):
        main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
    else :  
        main(sys.argv[1])
