from rapidocr_onnxruntime import RapidOCR
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import streamlit as st
import re
import pandas as pd
import cv2
from typing import List, Dict
HEADER_JUNK = re.compile(
    r'(kaufland|polska|markety|sp\.z|armii|krajowej|wrocław|bdo|ul\.|maja|ruda|śląska|nip|nr[: ]|\d{5}-\d{3})',
    re.IGNORECASE
)
MONEY = r'\d{1,3}(?:[ \u00A0]?\d{2})*(?:[.,]\d{2})'  # 47,64  3,753  1 234,56
_LETTERS_PL = "A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż"
MULT = r'[xX×*@Aa]'
NUM = r'\d+(?:[.,]\d{2})'  # 12,34  | 0,555 | 24,60
RE_ITEM_BOX =  re.compile(
    rf'''
    (?P<name>.+?)                 # nazwa produktu
    \s+
    (?P<qty>\d+(?:[.,]\d+)?)      # ilość
    \s*{MULT}\s*
    (?P<unit>{NUM})               # cena jednostkowa
    (?:\s*\|\s*|\s+)              # dopuszczamy: spacja LUB " | "
    (?P<total>{NUM})[A-Za-z]?     # total + ewentualna litera A/C/F
    ''',
    re.IGNORECASE | re.VERBOSE
) 
RE_ITEM_WEIGHT = re.compile(
    r'''
    (?:(?P<name>.+?)\s+)?                 # opcjonalna nazwa produktu
    (?P<weight>\d+(?:[.,]\d+))            # 0.276
    \s*KG                                 # KG / kg / Kg
    \s*[xX×]\s*
    (?P<perkg>\d+[.,]\d{2})               # 34.89
    (?:\s*\|\s*|\s+)                      # | lub spacja
    (?P<total>\d+[.,]\d{2})               # 9.63
    ''',
    re.IGNORECASE | re.VERBOSE
)

@st.cache_resource
def _0cr() -> RapidOCR:
    return RapidOCR(
        det_db_thresh =0.25,
        det_db_box_thresh=0.45,
        det_db_unclip_ratio=1.8,
        rec_char_dict_path="latin_pl.txt",
        rec_img_h=48,
        rec_img_w=320,
        rec_score_thresh=0.5,
        use_angle_cls=False
) 
def cut_header(merged):
    cleaned = []
    for m in merged:
        t = m["text"].strip()

        # Wyrzuć tylko oczywiste nagłówkowe śmieci
        if HEADER_JUNK.search(t):
            continue

        # Wyrzuć kod pocztowy
        if re.fullmatch(r'\d{2}-\d{3}', t):
            continue

        cleaned.append(m)

    return cleaned

def _box_w_h(box):
    xs = [p[0] for p in box]; ys = [p[1] for p in box]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return max(w, 1), max(h, 1)

def _should_drop_token(box, text, img_w=None, gray_arr=None):
    t = text.strip()
    if not t:
        return True
    w, h = _box_w_h(box)
    if h > 2 * w and t.isalpha() and len(t) <=2:
        return  True
    if t.isalpha() and set(t) <= {'A', 'B', 'C'} and len(t) <=6:
        return True
    if img_w is not None:
        x_right = max(p[0] for p in box)
        if x_right > 0.8 * img_w and t.isalpha() and len(t) <= 4:
            return True
    if img_w is not None:
        xs = [p[0] for p in box]
        x_left, x_right = min(xs), max(xs)
        if x_right < 0.05 *img_w or x_left > 0.9 *img_w:
            return True
    
    return False

def clean_merged_line(line: str) -> str:
    """
    Czyści ostateczną zmergowaną linię:
    – usuwa samotne cyfry typu „3”
    – usuwa pojedyncze kwoty typu „24,32” (VAT/PTU)
    – usuwa śmieciowe końcówki
    """
    t = line.strip()
    
    # 1. Usuń samotną cyfrę
    if re.fullmatch(r"\d", t):
        return ""

    # 2. Usuń linię, która wygląda jak pojedyncza kwota (np. 24,32)
    if "x" not in t.lower()and re.fullmatch(r"\d+[.,]\d{2}", t):
        return ""

    # 3. Usuń linię, która zawiera tylko liczby + przecinki/kropki
    if "x" not in t.lower() and re.fullmatch(r"[0-9., ]+", t):
        return ""

    return t


def filter_roi(res, top_st=None, bottom_st="suma"):
    """
    Lepsza wersja ROI:
    - NIE wymaga znalezienia 'paragon fiskalny'
    - PRZEPUŚCI zawsze górne 15% tekstów, żeby nie uciąć nagłówka (np. Kaufland)
    - Odetnie dopiero przy SUMIE
    """
    if not res:
        return []

    # Wszystkie Y
    ys = [min(p[1] for p in box) for box,_,_ in res]
    y_min, y_max = min(ys), max(ys)
    height = y_max - y_min

    # --- 1. Zawsze przepuść górne 15% tekstu
    safe_top = y_min + height * 0.15

    # --- 2. Znajdź SUMĘ (jeśli jest)
    y_bottom = None
    for box, txt, _ in res:
        if re.search(r"suma", txt.lower()):
            y_bottom = min(p[1] for p in box)
            break

    if y_bottom is None:
        y_bottom = y_max + 9999  # nic nie ucinamy

    keep = []
    for box, txt, score in res:
        y = min(p[1] for p in box)

        # przepuszczamy górne 15% lub wszystko między safe_top a SUMA
        if y <= safe_top or (safe_top < y < y_bottom):
            keep.append((box, txt, score))

    return keep


def _sanitize_token(t: str) -> str:
    if not t:
        return ""
    s = t.strip()

    # 1) „47,64C” / „47,64 PLN” -> „47,64”
    s = re.sub(rf'({MONEY})\s*[{_LETTERS_PL}]+$', r'\1', s)

    # 2) „1 , 99” -> „1,99”
    s = re.sub(r'(\d)\s+([.,]\d{2,3}\b)', r'\1\2', s)

    # 3) USUWAJ ogon liter TYLKO jeśli token zawiera jakąś cyfrę
    #    (czyli "47,64C" -> "47,64", ale "SokolowSchab" zostaje)
    if any(ch.isdigit() for ch in s):
        s = re.sub(rf'[{_LETTERS_PL}]+$', '', s).strip()
    return s    

def _post_fix_prices(line: str) -> str:
    # 3,753,75  ->  3,753 | 3,75
    # 1x4,494,49 -> 1x4,49 | 4,49
    line = re.sub(rf'({MONEY})\s*(?={MONEY}\b)', r'\1 | ', line)
    # jeszcze raz na wszelki wypadek usuń ogon liter
    line = re.sub(rf'({MONEY})\s*[{_LETTERS_PL}]+$', r'\1', line)
    return line

def group_lines(res, line_tol=60):
    rows = []
    sorted_res = sorted(res, key=lambda r: min(p[1] for p in r[0]))
    cur = []
    cur_y = None

    for box, text, score in sorted_res:
        ys = [p[1] for p in box]
        y_mid = sum(ys) / len(ys)   # 🔥 ŚREDNIE Y ZAMIAST MIN

        if cur and abs(y_mid - cur_y) > line_tol:
            rows.append(cur)
            cur = []

        cur.append((box, text, score))
        cur_y = y_mid if cur_y is None else (cur_y*0.5 + y_mid*0.5)

    if cur:
        rows.append(cur)

    merged = []
    for items in rows:
        items = sorted(items, key=lambda r: min(p[0] for p in r[0]))
        text = " ".join(_sanitize_token(t) for _, t, _ in items)
        text = _post_fix_prices(text)
        text = text.strip()

        # jeśli cały wiersz wyczyścił się do pustego – pomiń go
        if not text:
            continue
        min_x = min(min(p[0] for p in b) for b, _, _ in items)
        max_x = max(max(p[0] for p in b) for b,_ , _ in items)
        y = min(min(p[1] for p in b) for b, _, _ in items)
        merged.append({"text": text, "min_x": min_x, "max_x": max_x, "y": y})

    return merged


def clean_ocr_text(t: str) -> str:
    t = t.replace('@','x').replace('×','x').replace('X','x').replace('A','x')
    t = t.replace(',','.')  # żeby było 1.99, nie 1,99
    t = t.replace('..','.')
    return t

def _parse_price(s: str) -> float:
    s = s.replace(' ', '').replace('\u00A0','')
    if ',' in s and '.' not in s: s = s.replace(',','.')
    return float(s)

def _parse_items(merged):
    items = []
    pending = None
    last_text = None

    # -----------------------------------------------------------
    # Pomocnicze funkcje do rozróżniania nazwy od linii cenowej
    # -----------------------------------------------------------
    def is_price_line(t):
        """
        Linia cenowa zawiera cenę AND wygląda jak operacja x:
        1x3.75 | 3.75
        3x5.99 | 17.97
        0.276KG x34.89 | 9.63
        """
        return (
            re.search(r"\d+[.,]\d{2}", t)
            and ("x" in t.lower() or "kg" in t.lower())
        )

    def is_qty_only(t):
        """
        Rozpoznaje np:
        2x1, 3x2, i2x1, 12x1
        Ale NIE jeśli w linii występuje cena.
        """
        clean = t.replace(" ", "")
        return (
            re.fullmatch(r"[A-Za-z]*\d+x\d+[A-Za-z]*", clean) 
            and not re.search(r"\d+[.,]\d{2}", t)
        )

    def is_clear_name(t):
        """
        Uznajemy nazwę, jeśli:
        - zawiera litery
        - NIE jest linią cenową
        - NIE jest czystym 2x1
        """
        return (
            re.search(r"[A-Za-z]", t)
            and not is_price_line(t)
            and not is_qty_only(t)
        )

    # -----------------------------------------------------------
    # Główna pętla
    # -----------------------------------------------------------
    for ln in merged:
        t = ln["text"].strip()

        # odfiltrowanie oczywistych śmieci
        if re.fullmatch(r"\d{2}-\d{3}", t):
            last_text = None
            continue
        if re.search(r"(NIP|SPRZEDAZ|PTU|KASA|KASJER|PARAGON)", t, re.I):
            last_text = None
            continue

        # -------------------------------------------------------
        # 1) dopasowanie wagi lub qty z nazwą → pełna linia produktu
        # -------------------------------------------------------
        m = RE_ITEM_BOX.search(t) or RE_ITEM_WEIGHT.search(t)
        if m:
            raw_name = m.groupdict().get("name")

            if raw_name:
                name = raw_name.strip("-•. ")
            elif last_text and is_clear_name(last_text):
                name = last_text.strip("-•. ")
            else:
                name = "UNKNOWN"

            data = {"name": name}

            # ilość sztuk
            if m.groupdict().get("qty"):
                data["qty"] = _parse_price(m.group("qty"))

            # cena szt.
            if m.groupdict().get("unit"):
                data["unit_price"] = _parse_price(m.group("unit"))

            # waga
            if m.groupdict().get("weight"):
                data["weight"] = _parse_price(m.group("weight"))

            # cena za kg
            if m.groupdict().get("perkg"):
                data["price_per_kg"] = _parse_price(m.group("perkg"))

            # suma
            if m.groupdict().get("total"):
                data["total"] = _parse_price(m.group("total"))
            else:
                data["total"] = None

            items.append(data)
            pending = None
            last_text = None
            continue

        # -------------------------------------------------------
        # 2) linia czysto cenowa, np:
        #    1x3.75 | 3.75
        # -------------------------------------------------------
        m2 = re.search(r'(?P<qty>\d+)\s*[xX×]\s*(?P<unit>\d+[.,]\d{2}).*\|\s*(?P<total>\d+[.,]\d{2})', t)
        if m2:
            pending = {
                "name": None,
                "qty": int(m2.group("qty")),
                "unit_price": _parse_price(m2.group("unit")),
                "total": _parse_price(m2.group("total"))
            }
            continue

        # -------------------------------------------------------
        # 3) nazwa produktu poprzedzająca pending
        # -------------------------------------------------------
        if pending and is_clear_name(t):
            pending["name"] = t.strip("-•. ")
            items.append(pending)
            pending = None
            last_text = None
            continue

        # -------------------------------------------------------
        # 4) samodzielna nazwa produktu → zapamiętaj ją
        # -------------------------------------------------------
        if is_clear_name(t):
            last_text = t.strip("-•. ")
            continue

        # -------------------------------------------------------
        # 5) fallback — zapamiętaj linię
        # -------------------------------------------------------
        last_text = t

    return items
def parse_biedronka(merged_lines: List[Dict]) -> List[Dict]:
    """
    Przyjmuje merged_lines (już po OCR), zwraca listę pozycji.
    Jest to jedyna funkcja, którą importuje główny program.
    """
    return _parse_items(merged_lines)