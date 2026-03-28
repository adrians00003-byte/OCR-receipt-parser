# OCR Receipt Parser

A modular OCR pipeline for transforming raw receipt images into structured financial data.

This project is designed to handle real-world, noisy receipts using store-specific parsing logic.

## Features
- OCR preprocessing
- Store-specific parsers (Biedronka, Kaufland)
- CSV export
- Streamlit dashboard

## Architecture
Image → OCR → Text → Store detection → Parser → CSV

## Example output
WODx CIS LG 1.5 6x2.39 | 14.34
OsheeUitamin1100 xCCCC 1x4.49 | 4.49
SledNaRazKurki100 1x3.75 | 3.75
SledNaRazSuszPom100 1x3.75 | 3.75
Schab/Szynka 100 3x5.99 | 17.97
ColaZerFanSpri2x1 1x9.98 | 9.98
## Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Run
streamlit run src/main.py

## Requirements
Tested on Python 3.11.8