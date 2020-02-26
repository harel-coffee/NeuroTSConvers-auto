from sklearn.cluster import KMeans


def kmeans_auto (data, max_k =  8):

    best_k = 2
    best_model = KMeans (n_clusters = 2). fit (data)
    score = best_model. inertia_

    for k in range (3, max_k + 1):
        clustering = KMeans (n_clusters = k). fit (data)
        #clustering = DBSCAN (eps=3, min_samples=2).fit (data)
        if clustering. inertia_ < score:
            score = clustering. inertia_
            best_model = clustering

    return best_model, k
