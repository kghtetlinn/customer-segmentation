import streamlit as st
import matplotlib.pyplot as plt
from model import load_data, preprocess_data, apply_kmeans

st.title("Customer Segmentation using K-Means")

df = load_data()
data_scaled, df_original = preprocess_data(df)
clusters = apply_kmeans(data_scaled)

df_original["Cluster"] = clusters
st.write("Clustered Data", df_original.head())

# Plotting
st.subheader("Cluster Distribution")
fig, ax = plt.subplots()
df_original["Cluster"].value_counts().plot(kind="bar", ax=ax)
st.pyplot(fig)
