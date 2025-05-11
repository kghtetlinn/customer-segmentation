import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_data():
    df = pd.read_csv("data/marketing_campaign.csv", sep="\t")
    df.dropna(inplace=True)
    return df

def preprocess_data(df):
    selected = df[["Income", "Recency", "MntWines", "MntFruits", "MntMeatProducts", "MntGoldProds"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(selected)
    return scaled, df

def apply_kmeans(data, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters
