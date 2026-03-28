# kaufland.py
import re
from typing import List, Dict

# KG
RE_WEIGHT = re.compile(
    r"(?P<weight>\d+(?:[.,]\d+))\s*KG\s*[xX×]\s*(?P<perkg>\d+[.,]\d{2})\s*\|\s*(?P<total>\d+[.,]\d{2})",
    re.IGNORECASE
)

# SZT – ulepszony
RE_SZT = re.compile(
    r"[tT]?(?P<qty>\d+)[tT]?\s*SZT\s*[xX×]\s*(?P<unit>\d+[.,]\d{2})\s*\|\s*(?P<total>\d+[.,]\d{2})",
    re.IGNORECASE
)

def _clean(x: str) -> float:
    return float(x.replace(",", "."))

def is_name(t: str) -> bool:
    s = t.strip()
    u = s.upper()

    if RE_WEIGHT.search(s) or RE_SZT.search(s):
        return False


    # musi mieć litery (nazwa)
    if not re.search(r"[A-ZĄĆĘŁŃÓŚŹŻ]", u):
        return False

    # odrzuć ewidentne nagłówki
    if "PXRXGON" in u or "FISKXLNY" in u or "PARAGON" in u or "FISKALNY" in u:
        return False


    # odrzuć linie będące tylko liczbą/ceną
    if re.fullmatch(r"\d+", s) or re.fullmatch(r"\d+[.,]\d{2}", s):
        return False
    

    # odrzuć linię SZT (to nie nazwa)
    if "SZT" in u:
        return False

    # opcjonalnie: minimalna długość
    if len(s) < 3:
        return False
    if re.fullmatch(r"[A-Za-z]", s):
        return False
    
    return True


def parse_kaufland(merged: List[Dict]) -> List[Dict]:
    items = []
    last_name = None
    pending_szt = None

    for ln in merged:
        t = ln["text"].strip()

        if "SZT" in t.upper():
            print(">> SZT:", t)

        if pending_szt:
            print(".. pending_szt czeka, next line:", t, "is_name:", is_name(t))

        # 1 — nazwa po sztukowej
        if pending_szt and is_name(t):
            pending_szt["name"] = t
            items.append(pending_szt)
            pending_szt = None
            last_name = None
            continue

        # 2 — nazwa
        if is_name(t):
            last_name = t
            continue

        # 3 — waga KG
        m = RE_WEIGHT.search(t)
        if m:
            name = last_name or "UNKNOWN"
            items.append({
                "name": name,
                "weight": _clean(m.group("weight")),
                "price_per_kg": _clean(m.group("perkg")),
                "total": _clean(m.group("total"))
            })
            last_name = None
            continue

        # 4 — sztukowe
        m2 = RE_SZT.search(t)
        if m2:
            if last_name:
                items.append({
                    "name": last_name,
                    "qty": int(m2.group("qty")),
                    "unit_price": _clean(m2.group("unit")),
                    "total": _clean(m2.group("total"))
                })
                last_name = None
            else:
                pending_szt = {
                    "name": "UNKNOWN",
                    "qty": int(m2.group("qty")),
                    "unit_price": _clean(m2.group("unit")),
                    "total": _clean(m2.group("total"))
                }
            continue

    if pending_szt:
        items.append(pending_szt)

    return items
