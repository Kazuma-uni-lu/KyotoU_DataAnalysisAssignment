import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
from sklearn.decomposition import PCA
import parse_data
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

df = parse_data.parse()

X = df[[
        "Hour",
        "Temperature",
        "Humidity",
        "Wind Speed",
        "Visibility",
        "Solar Radiation",
        "Rainfall"
    ]]
y = df["Bike Count"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#KN regression
def KN_reg_result():
    k = 10
    knr = neighbors.KNeighborsRegressor(n_neighbors=k, weights="distance")
    knr.fit(X_train_scaled, y_train)

    y_pred = knr.predict(X_test_scaled)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2   = r2_score(y_test, y_pred)

    print("KNN Regression")
    print("k =", k)
    print("RMSE =", rmse)
    print("R^2  =", r2)
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.35, s=12)
    m0, m1 = y_test.min(), y_test.max()
    plt.plot([m0, m1], [m0, m1], "r--", linewidth=2)

    plt.xlabel("True Bike Count")
    plt.ylabel("Predicted Bike Count")
    plt.title(f"KNN Regression (k={k})\nRMSE={rmse:.1f}, R²={r2:.2f}")
    plt.grid(alpha=0.3)
    plt.show()
    return rmse, r2

#SVR regression
def SVR_reg_result():
    svr = svm.SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
    svr.fit(X_train_scaled, y_train)
    y_pred_svr = svr.predict(X_test_scaled)
    rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
    r2_svr   = r2_score(y_test, y_pred_svr)
    print("SVR Regression (RBF)")
    print("RMSE =", rmse_svr)
    print("R^2  =", r2_svr)
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred_svr, alpha=0.35, s=12)

    m0, m1 = y_test.min(), y_test.max()
    plt.plot([m0, m1], [m0, m1], "r--", linewidth=2)

    plt.xlabel("True Bike Count")
    plt.ylabel("Predicted Bike Count")
    plt.title(f"SVR (RBF)\nRMSE={rmse_svr:.1f}, R²={r2_svr:.2f}")
    plt.grid(alpha=0.3)
    plt.show()
    return rmse_svr, r2_svr

#KNN_reggg = KN_reg_result()
#SVR_regg = SVR_reg_result()


#PCA + K-means
X2 = df[
    [
        "Hour",
        "Temperature",
        "Humidity",
        "Wind Speed",
        "Visibility",
        "Solar Radiation",
        "Rainfall"
    ]
].values
X_scaled = scaler.fit_transform(X2)

def PCA_result():
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total explained variance:", pca.explained_variance_ratio_.sum())

    inertias = []
    K = range(1, 9)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_pca)
        inertias.append(km.inertia_)
    plt.figure(figsize=(6,4))
    plt.plot(K, inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for K-means")
    plt.grid(alpha=0.3)
    plt.show()

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    sil_score = silhouette_score(X_pca, labels)
    print("Silhouette score: ", sil_score)
    plt.figure(figsize=(7,6))
    plt.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=labels,
        cmap="tab10",
        alpha=0.35,
        s=12
    )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("K-means Clustering of Bike Usage Patterns (PCA space)")
    plt.grid(alpha=0.3)
    plt.show()
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels
    a = df_clustered.groupby("Cluster")[[
    "Bike Count",
    "Temperature",
    "Hour",
    "Humidity",
    "Rainfall"
    ]].mean()
    return a


#PCA_result()