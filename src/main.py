import streamlit as st
from PIL import Image
import pandas as pd
from typing import List, Dict
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser.biedronka import parse_biedronka, group_lines, cut_header, clean_ocr_text
from parser.kaufland import parse_kaufland
from manager import save_receipt
from dashboard import show_dashboard
from prepare import detect_receipt_corners, run_ocr
def normalize_items(items: List[Dict]) -> List[Dict]:
    normalized = []
    for it in items:
        normalized.append({
            "name": it.get("name", None),
            "qty": it.get("qty", None),
            "unit_price": it.get("unit_price", None),
            "weight": it.get("weight", None),
            "price_per_kg": it.get("price_per_kg", None),
            "total": it.get("total", None),
        })
    return normalized

# 🔍 AUTOMATYCZNE WYKRYWANIE SKLEPU
def detect_store(res):
    raw_text = " ".join(txt.lower() for (_, txt, _) in res)

    if "biedronka" in raw_text or "biedronk" in raw_text:
        return "biedronka"

    if "kaufland" in raw_text:
        return "kaufland"

    return "unknown"
# --- UI ---
st.sidebar.title("Menu")

choice = st.sidebar.radio(
    "Wybierz opcję:",
    ["📤 Wczytaj paragon", "📊 Historia zakupów"]
)


# =====================================================
# 🔥 2. TRYB HISTORII (DASHBOARD)
# =====================================================
if choice == "📊 Historia zakupów":
    show_dashboard()
    st.stop()


# =====================================================
# 🔥 3. TRYB WCZYTYWANIA PARAGONU (OCR + PARSER)
# =====================================================
st.title("Asystent Paragonów")


# --- wczytywanie pliku ---
upload = st.file_uploader("Wczytaj paragon", type=["png","jpg","jpeg"])
    
if upload is None:
    st.stop()

pil = Image.open(upload).convert("RGB")
st.image(pil, caption="Oryginalny paragon")

# --- 1. Prostowanie obrazu ---
corners, warped = detect_receipt_corners(pil)
if warped is None:
    warped = pil



# --- 2. OCR ---
res = run_ocr(warped)

# --- 2.5. Wykrywanie sklepu z surowego OCR ---
store = detect_store(res)
st.subheader(f"Wykryty sklep: {store.upper()}")

# --- 3. Grupowanie i czyszczenie ---
# --- 3. Grupowanie i czyszczenie ---
merged = group_lines(res)
merged = cut_header(merged)

cleaned = []
for m in merged:
    t = clean_ocr_text(m["text"])
    if t:
        cleaned.append({**m, "text": t})

merged = cleaned

# --- 4. Wybór parsera ---
if store == "biedronka":
    items = normalize_items(parse_biedronka(merged))


elif store == "kaufland":
    items = normalize_items(parse_kaufland(merged))


else:
    st.warning("Nie rozpoznano sklepu — używam parsera Biedronki.")
    items = normalize_items(parse_biedronka(merged))

    
# --- 5. TABELA ---
if items:
    df = pd.DataFrame(items)

    df = df.rename(columns={
        "name": "Produkt",
        "qty": "Ilość",
        "unit_price": "Cena szt.",
        "weight": "Waga (kg)",
        "price_per_kg": "Cena/kg",
        "total": "Suma"
        })

    # Wymuszona kolejność (zawsze taka sama)
    df = df[["Produkt", "Ilość", "Cena szt.", "Waga (kg)", "Cena/kg", "Suma"]]

    st.dataframe(df, use_container_width=True)
    # 🔥 6. Zapis do CSV
    if st.button("💾 Zapisz do historii (CSV)"):
        save_receipt(df, store)
        st.success(f"Zapisano paragon ({store}) do historii.")

else:
    st.warning("Nie udało się rozpoznać pozycji.")

