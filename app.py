import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.title("Clustering & Regresi Pelanggan Online Retail")

# =====================
# LOAD DATA (FIX)
# =====================
df = pd.read_excel("Online_Retail_1002_rows.xlsx")

st.subheader("Contoh Data")
st.dataframe(df.head())

# =====================
# PREPROCESSING
# =====================
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

customer_df = df.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum'
}).reset_index()

st.subheader("Data Setelah Agregasi Customer")
st.dataframe(customer_df.head())

# =====================
# CLUSTERING
# =====================
X = customer_df[['Quantity', 'TotalPrice']]

kmeans = KMeans(n_clusters=3, random_state=42)
customer_df['Cluster'] = kmeans.fit_predict(X)

st.subheader("Hasil Clustering")
st.dataframe(customer_df.head())

# =====================
# VISUALISASI CLUSTER
# =====================
fig, ax = plt.subplots()
scatter = ax.scatter(
    customer_df['Quantity'],
    customer_df['TotalPrice'],
    c=customer_df['Cluster']
)
ax.set_xlabel("Total Quantity")
ax.set_ylabel("Total Price")
ax.set_title("Hasil Clustering Pelanggan Online Retail")
st.pyplot(fig)

# =====================
# JUMLAH ANGGOTA CLUSTER
# =====================
st.subheader("Jumlah Anggota Tiap Cluster")
cluster_counts = customer_df['Cluster'].value_counts().sort_index()
st.write(cluster_counts)

# =====================
# REGRESI LINEAR
# =====================
st.subheader("Regresi Linear (Quantity → Total Price)")

X_reg = customer_df[['Quantity']]
y_reg = customer_df['TotalPrice']

model = LinearRegression()
model.fit(X_reg, y_reg)

y_pred = model.predict(X_reg)

r2 = r2_score(y_reg, y_pred)
mse = mean_squared_error(y_reg, y_pred)

st.write("R2 Score:", r2)
st.write("MSE:", mse)

coef_df = pd.DataFrame({
    "Variabel": ["Quantity"],
    "Koefisien": model.coef_
})

st.subheader("Koefisien Regresi")
st.dataframe(coef_df)
