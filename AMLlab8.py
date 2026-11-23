
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

st.title("Hierarchical Clustering")

# 1. Load Dataset
X, y = make_blobs(n_samples=100, centers=3, random_state=42)

df = pd.DataFrame(X, columns=["Income", "Spending Score"])

st.header("1. Dataset")
st.dataframe(df.head())
st.write("Records:", df.shape[0])
st.write("Attributes:", df.shape[1])
st.write(df.dtypes)

# 2. Preprocessing
st.header("2. Data Preprocessing")

df = df.fillna(df.mean())

scaler = StandardScaler()
data = scaler.fit_transform(df)

st.write("Standardized Data")
st.dataframe(pd.DataFrame(data, columns=df.columns).head())

# 3. Distance Matrices
st.header("3. Distance Matrices")

for distance in ["euclidean", "cityblock", "cosine"]:
    matrix = squareform(pdist(data, metric=distance))
    st.write(distance.capitalize())
    st.dataframe(pd.DataFrame(matrix[:5, :5]).round(2))

# 4. Agglomerative Clustering
st.header("4. Agglomerative Clustering")

methods = ["single", "complete", "average", "ward"]

results = {}

for method in methods:
    model = AgglomerativeClustering(
        n_clusters=3,
        linkage=method
    )
    labels = model.fit_predict(data)
    results[method] = labels

    st.write(method.capitalize(), "Linkage")
    st.write("Silhouette Score:", round(silhouette_score(data, labels), 3))

# 5. Dendrogram
st.header("5. Dendrogram")

for method in methods:
    Z = linkage(data, method=method)

    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(Z, ax=ax)
    ax.set_title(method.capitalize() + " Linkage")
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Distance")

    st.pyplot(fig)
    plt.close(fig)

# 6. Divisive Clustering
st.header("6. Divisive Clustering")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
divisive_labels = kmeans.fit_predict(data)

st.write("Divisive Cluster Labels:")
st.write(divisive_labels)
st.write("Silhouette Score:",
         round(silhouette_score(data, divisive_labels), 3))

# 7. Different Number of Clusters
st.header("7. Different Number of Clusters")

for k in range(2, 6):
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(data)

    score = silhouette_score(data, labels)

    st.write("Clusters:", k, "Score:", round(score, 3))

# 8. PCA Visualization
st.header("8. PCA Visualization")

pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)

fig, ax = plt.subplots()

ax.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=results["ward"],
    cmap="viridis"
)

ax.set_title("Agglomerative Clustering")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")

st.pyplot(fig)
plt.close(fig)

# 9. Distance and Linkage Comparison
st.header("9. Comparison")

for distance in ["euclidean", "manhattan", "cosine"]:
    model = AgglomerativeClustering(
        n_clusters=3,
        metric=distance,
        linkage="average"
    )

    labels = model.fit_predict(data)

    score = silhouette_score(data, labels)

    st.write(distance, "Score:", round(score, 3))

# 10. Cluster Interpretation
st.header("10. Cluster Interpretation")

df["Cluster"] = results["ward"]

st.write("Cluster Groups:")
st.dataframe(df)

st.write("Cluster Average:")
st.dataframe(df.groupby("Cluster").mean())

st.subheader("Conclusion")

st.write("""
Agglomerative clustering starts with individual data points
and combines them into groups.

Divisive clustering starts with one large group and divides
it into smaller groups.

Different distance measures and linkage methods produce
different clusters. Dendrograms help understand cluster
formation, and PCA helps visualize the clusters.
""")

st.success("Lab Exercise 8 Completed!")
