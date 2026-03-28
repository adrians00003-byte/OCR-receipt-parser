# csv_dashboard.py
import streamlit as st
import pandas as pd
from manager import (
    load_data,
    filter_data,
    top_products,
    monthly_spending,
    export_excel,
)

def show_dashboard():
    st.header("📊 Historia zakupów")

    df = load_data()
    if df.empty:
        st.info("Brak zapisanych paragonów.")
        return

    # -------------------------
    # FILTRY
    # -------------------------
    st.subheader("🔎 Filtruj dane")

    store = st.selectbox("Sklep:", ["", "Biedronka", "Kaufland"])
    product = st.text_input("Produkt:")
    date = st.date_input("Data:", value=None)

    df_filtered = filter_data(df, store, date.strftime("%Y-%m-%d") if date else None, product)

    st.dataframe(df_filtered, use_container_width=True)

    # -------------------------
    # STATYSTYKI
    # -------------------------
    st.subheader("📈 Statystyki")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### 💰 Najdroższe produkty")
        st.bar_chart(top_products(df_filtered))

    with col2:
        st.write("### 📅 Wydatki miesięczne")
        st.line_chart(monthly_spending(df))

    # -------------------------
    # EXPORT
    # -------------------------
    if st.button("📥 Eksport do Excel"):
        path = export_excel()
        st.success(f"Zapisano jako {path}")
