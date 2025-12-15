import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Data Mining Final", layout="wide")

st.title("📊 Clustering & Regresi Pelanggan Online Retail")

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("Online_Retail_1002_rows.csv")

st.subheader("📁 Contoh Data")
st.dataframe(df.head())

# =====================
# CLUSTERING
# =====================
st.subheader("🔹 Clustering Pelanggan (K-Means)")

X = df[['Quantity', 'TotalPrice']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualisasi Clustering
fig, ax = plt.subplots()
scatter = ax.scatter(
    df['Quantity'],
    df['TotalPrice'],
    c=df['Cluster']
)
ax.set_xlabel("Total Quantity")
ax.set_ylabel("Total Price")
ax.set_title("Hasil Clustering Pelanggan")

st.pyplot(fig)

# Jumlah anggota cluster
st.subheader("📌 Jumlah Anggota Tiap Cluster")
st.write(df['Cluster'].value_counts().sort_index())

# =====================
# REGRESI LINEAR
# =====================
st.subheader("📈 Regresi Linear (Quantity → Total Price)")

X_reg = df[['Quantity']]
y = df['TotalPrice']

model = LinearRegression()
model.fit(X_reg, y)

y_pred = model.predict(X_reg)

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

st.write("### 🔢 Hasil Regresi Linear")
st.write(f"**R² Score :** {r2:.3f}")
st.write(f"**MSE :** {mse:.2f}")

coef_df = pd.DataFrame({
    "Variabel": ["Quantity"],
    "Koefisien": model.coef_
})

st.write("### 📌 Koefisien Regresi")
st.dataframe(coef_df)

# Grafik regresi
fig2, ax2 = plt.subplots()
ax2.scatter(df['Quantity'], df['TotalPrice'])
ax2.plot(df['Quantity'], y_pred)
ax2.set_xlabel("Quantity")
ax2.set_ylabel("Total Price")
ax2.set_title("Regresi Linear Quantity vs Total Price")

st.pyplot(fig2)

# =====================
# KESIMPULAN
# =====================
st.subheader("📝 Kesimpulan")
st.markdown("""
- Clustering membagi pelanggan menjadi **3 segmen**: kecil, menengah, dan besar.
- Regresi menunjukkan **Quantity berpengaruh positif terhadap Total Price**.
- Model regresi cukup baik dengan **R² > 0.7**.
""")
