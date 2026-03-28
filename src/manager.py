# csv_manager.py
import pandas as pd
import os
from datetime import datetime
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "paragony.csv"

# ======================
# 1. Zapis danych
# ======================
def save_receipt(df: pd.DataFrame, store: str):
    """
    Zapisuje pozycje paragonu:
    - dodaje kolumnę 'store'
    - dodaje kolumnę 'date'
    - dopisuje dane do CSV
    """
    df["store"] = store
    df["date"] = datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, index=False)
        return

    old = pd.read_csv(CSV_PATH)
    merged = pd.concat([old, df], ignore_index=True)
    merged.to_csv(CSV_PATH, index=False)


# ======================
# 2. Wczytanie pliku
# ======================
def load_data() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()
    return pd.read_csv(CSV_PATH)


# ======================
# 3. Filtry danych
# ======================
def filter_data(df: pd.DataFrame, store=None, date=None, product=None):
    if store:
        df = df[df["store"].str.contains(store, case=False, na=False)]
    if date:
        df = df[df["date"] == date]
    if product:
        df = df[df["Produkt"].str.contains(product, case=False, na=False)]
    return df


# ======================
# 4. Statystyki
# ======================
def top_products(df: pd.DataFrame, n=10):
    """Najdroższe produkty."""
    return df.groupby("Produkt")["Suma"].sum().sort_values(ascending=False).head(n)


def monthly_spending(df: pd.DataFrame):
    """Wydatki miesięczne."""
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
    return df.groupby("month")["Suma"].sum()


# ======================
# 5. Eksport
# ======================
def export_excel(path="paragony.xlsx"):
    df = load_data()
    df.to_excel(path, index=False)
    return path
