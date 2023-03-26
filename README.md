# analyse-de-clustering
Bienvenue dans ce programme Python d'analyse de clustering ! Ce programme est conçu pour vous permettre de comparer quatre méthodes de clustering populaires en Python : KMeans, AgglomerativeClustering, MeanShift et DBSCAN. Il offre des informations sur le temps d'exécution et teste la qualité des groupes obtenus grâce à la méthode de silouette. Pour une meilleure visualisation des résultats, le programme affiche également des graphiques clairs et intuitifs.

## Bibliothèques nécessaires

Les bibliothèques nécessaires pour exécuter le code sont les suivantes :

[sklearn](https://scikit-learn.org/stable/) pour les méthodes de clustering
[pandas](https://pandas.pydata.org/) pour importer les données
[numpy](https://numpy.org/) pour certaines manipulations de données
[matplotlib](https://matplotlib.org/) pour afficher les graphiques
[time](https://docs.python.org/fr/3/library/time.html) pour calculer le temps d'exécution
[sklearn.metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) pour calculer le nombre optimal de cluster, le coefficient de silhouette et la précision
    
## Méthodes de clustering 
* [KMeans](https://fr.wikipedia.org/wiki/K-moyennes) : C'est l'une des méthodes de clustering les plus populaires. Elle consiste à diviser les données en k groupes distincts en minimisant la somme des distances entre chaque point et le centre de son cluster. C'est une méthode de clustering par partition, ce qui signifie que chaque point de données appartient à un seul cluster.
* [Agglomerative Clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) : Il s'agit d'une méthode de clustering hiérarchique qui commence avec chaque point de données comme son propre cluster, puis fusionne les clusters les plus similaires jusqu'à ce qu'il n'en reste qu'un. C'est une méthode de clustering par hiérarchie, ce qui signifie que les points peuvent appartenir à plusieurs clusters à différents niveaux de l'arbre.
* [MeanShift](https://en.wikipedia.org/wiki/Mean_shift) : Cette méthode de clustering utilise une approche de densité pour trouver les centres de cluster. Elle commence par sélectionner un point aléatoire comme centre de cluster, puis calcule la densité de points autour de ce centre. Les points sont ensuite déplacés vers des zones de densité plus élevée, jusqu'à ce qu'ils atteignent un maximum local de densité. Cela se répète jusqu'à ce que tous les points se trouvent dans une zone de densité maximale. C'est une méthode de clustering par densité, ce qui signifie que les points peuvent appartenir à différents clusters en fonction de leur densité de voisinage.
* [DBSCAN](https://fr.wikipedia.org/wiki/DBSCAN) : Cette méthode de clustering utilise également une approche de densité pour trouver les clusters, mais elle est plus robuste que MeanShift. DBSCAN divise les données en trois types : les points centraux, les points limites et les points de bruit. Les points centraux ont un nombre minimum de voisins dans un rayon spécifié, les points limites ont moins de voisins que le minimum, mais appartiennent à un cluster avec des points centraux, tandis que les points de bruit n'ont pas suffisamment de voisins pour appartenir à un cluster. C'est une méthode de clustering par densité, qui est efficace pour les ensembles de données avec des clusters de forme arbitraire et de taille variable.



## Fonctions

    display_clusters(model, X, ax): cette fonction permet d'afficher les clusters sur un graphique en 2D. Elle prend en paramètres le modèle de clustering, les données et le graphique sur lequel afficher les résultats.
    test_kmean(X, nb_cluster): cette fonction effectue le clustering des données avec la méthode KMeans. Elle prend en paramètres les données et le nombre de clusters souhaité.
    agglo(X, nb_cluster): cette fonction effectue le clustering des données avec la méthode AgglomerativeClustering. Elle prend en paramètres les données et le nombre de clusters souhaité.
    MeanShift_test(X, bw): cette fonction effectue le clustering des données avec la méthode MeanShift. Elle prend en paramètres les données et la bande passante.
    DBSCAN_test(X, eps, ms): cette fonction effectue le clustering des données avec la méthode DBSCAN. Elle prend en paramètres les données, la valeur de epsilon et le nombre minimum de points.
    nb_cluster_optimal(X): cette fonction permet de déterminer le nombre optimal de clusters avec la méthode KMeans en calculant le coefficient de silhouette. Elle prend en paramètre les données.

##Exécution du code

Le code importe des données à partir d'un fichier CSV. Pour exécuter le code, il suffit de renseigner le nom de ce fichier à l'exécution.

```
python3 test.py <fichier.csv>
```

La fonction main() appelle les différentes fonctions pour effectuer les clusterings et afficher les graphiques un par un.

Attention! Il faut parfois attendre un moment car MeanShift est plutôt lent sur de petites bases de données.




