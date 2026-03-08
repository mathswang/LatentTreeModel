import numpy as np
from sklearn.cluster import KMeans
from dmicc.tools.metrics import metrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def check_clustering_metrics(npc, train_loader):
    trainFeatures = npc.memory
    z = trainFeatures.cpu().numpy()
    y = np.array(train_loader.dataset.labels)
    n_clusters = len(np.unique(y))
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z)
    sil = silhouette_score(z, y_pred)
    ch = calinski_harabasz_score(z, y_pred)
    db = davies_bouldin_score(z, y_pred)
    return sil, ch, db, metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred)
