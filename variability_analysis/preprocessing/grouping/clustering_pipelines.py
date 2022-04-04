import pandas as pd

from copy import deepcopy

import matplotlib.pyplot as plt

from kneed import KneeLocator

import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, Normalizer

from scipy.cluster.hierarchy import dendrogram

from sklearn_extra.cluster import KMedoids


from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes


def get_optimal_k(
    clusterer,
    features,
    min_k=2,
    max_k=10,
    K_MODES_ELBOW=0,
    score_calculator=silhouette_score,
    curve="convex",
    direction="decreasing",
):
    cluster_range = range(min_k, max_k)

    if K_MODES_ELBOW > 0:
        return K_MODES_ELBOW
    scores = []
    for k in cluster_range:
        labels = clusterer(k).fit_predict(features)
        scores.append(score_calculator(features, labels))

    print(f"scores: {scores}")

    plt.plot(cluster_range, scores)
    plt.show()

    # find elbow / best cluster nr.
    kl = KneeLocator(cluster_range, scores, curve=curve, direction=direction)
    print(f"best nr. of clusters: {kl.elbow}")
    return kl.elbow


def clust_kproto(
    _features,
    cat_idx,
    num_cols,
    K_MODES_ELBOW=0,
    min_k=2,
    max_k=11,
    score_calculator=None,
):
    features = deepcopy(_features)
    scaler = MinMaxScaler()
    features[num_cols] = scaler.fit_transform(features[num_cols])
    sc_features = features

    cluster_range = range(min_k, max_k)
    valid_k = []

    if K_MODES_ELBOW == 0:
        scores = []
        for k in cluster_range:
            try:
                km = KPrototypes(n_clusters=k, init="Cao", n_init=10, n_jobs=-1).fit(
                    sc_features, categorical=cat_idx
                )
                valid_k.append(k)
                if score_calculator != None:
                    scores.append(score_calculator(sc_features, km.labels_))
                else:
                    scores.append(km.cost_)
            except:
                continue

        print(f"scores: {scores}")

        plt.plot(valid_k, scores)
        plt.show()

        # find elbow / best cluster nr.
        kl = KneeLocator(valid_k, scores, curve="convex", direction="decreasing")
        K_MODES_ELBOW = kl.elbow
        print(f"best nr. of clusters: {K_MODES_ELBOW}")

    km = KPrototypes(n_clusters=K_MODES_ELBOW, init="Cao", n_init=10, n_jobs=-1)

    clusters = km.fit_predict(sc_features, categorical=cat_idx)

    if score_calculator != None:
        print(f"score: {score_calculator(sc_features, km.labels_)} // Cost: {km.cost_}")
    else:
        print(f"Cost: {km.cost_}")

    return clusters, km.cluster_centroids_


def clust_pipeline(
    features, n_pca=2, _eps=0.125, _min_samples=4, K_MEANS_ELBOW=0, k_max=2
):

    if K_MEANS_ELBOW == 0:
        cost = []
        for k in range(1, k_max):
            pipe = Pipeline(
                [
                    (
                        "preprocessor",
                        Pipeline(
                            [
                                ("scaler", MinMaxScaler()),
                                # ("pca", PCA(n_components=n_pca)),
                            ]
                        ),
                    ),
                    (
                        "clusterer",
                        Pipeline(
                            [
                                (
                                    "kmeans",
                                    KMeans(n_clusters=k, init="k-means++"),
                                ),
                            ]
                        ),
                    ),
                ]
            )

            clusters = pipe.fit_predict(features)
            inertia = pipe["clusterer"]["kmeans"].inertia_
            cost.append(inertia)

        # find elbow / best cluster nr.
        kl = KneeLocator(range(1, k_max), cost, curve="convex", direction="decreasing")
        K_MEANS_ELBOW = kl.elbow
        print(f"best nr. of clusters: {K_MEANS_ELBOW}")

    ### Pipeline setup
    preprocessor = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            # ("pca", PCA(n_components=n_pca)),
        ]
    )

    clusterer = Pipeline(
        [
            (
                "kmeans",
                KMeans(n_clusters=K_MEANS_ELBOW, init="k-means++"),
                # DBSCAN( eps=_eps, min_samples=_min_samples, n_jobs=-1 ),
            ),
        ]
    )

    pipe = Pipeline([("preprocessor", preprocessor), ("clusterer", clusterer)])

    clusters = pipe.fit_predict(features)

    ### Pipeline metrics

    preprocessed_data = pipe["preprocessor"].transform(features)

    predicted_labels = pipe["clusterer"]["kmeans"].labels_
    inertia = pipe["clusterer"]["kmeans"].inertia_
    centers = pipe["clusterer"]["kmeans"].cluster_centers_

    ssc = silhouette_score(
        preprocessed_data, predicted_labels
    )  # closer to 0: more overlap;  closer to 1: well-separed clusters

    print(f"silhouette_score: {ssc}, inertia:{inertia}")

    # Visualization
    """if n_pca == 2:
        pcadf = pd.DataFrame(
            pipe["preprocessor"].transform(features),
            columns=["component_1", "component_2"],
        )
        pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_

        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(8, 8))

        scat = sns.scatterplot(
            x="component_1",
            y="component_2",
            s=100,
            data=pcadf,
            hue="predicted_cluster",
            palette="Set2",
        )

        scat.set_title("Clustering results from Event Log")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

        plt.show()
    """

    return clusters, centers


def tune_pca(features, n_max, elbow=4, _eps=0.125, _min_samples=4):

    # Empty lists to hold evaluation metrics
    silhouette_scores = []
    preprocessor = Pipeline(
        [
            ("scaler", MinMaxScaler()),
        ]
    )

    clusterer = Pipeline(
        [
            (
                "kmeds",
                # KMeans( n_clusters=elbow, init='k-means++' ),
                # KMedoids( n_clusters=elbow, init='k-medoids++' ),
                DBSCAN(eps=_eps, min_samples=_min_samples),
            ),
        ]
    )

    pipe = Pipeline([("preprocessor", preprocessor), ("clusterer", clusterer)])

    for n in range(0, n_max):
        # This set the number of components for pca,
        # but leaves other steps unchanged
        if n > 1:
            pipe = Pipeline(
                [
                    (
                        "preprocessor",
                        Pipeline(
                            [("scaler", MinMaxScaler()), ("pca", PCA(n_components=n))]
                        ),
                    ),
                    ("clusterer", clusterer),
                ]
            )

        pipe.fit(features)

        silhouette_coef = silhouette_score(
            pipe["preprocessor"].transform(features),
            pipe["clusterer"]["kmeds"].labels_,
        )

        # Add metrics to their lists
        silhouette_scores.append(silhouette_coef)

        plt.style.use("fivethirtyeight")

    plt.figure(figsize=(6, 6))
    plt.plot(
        range(0, n_max),
        silhouette_scores,
        c="#008fd5",
        label="Silhouette Coefficient",
    )

    plt.xlabel("n_components")
    plt.legend()
    plt.title("Clustering Performance as a Function of n_components")
    plt.tight_layout()
    plt.show()
