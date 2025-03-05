# Adverse Drug Event Detection System

This project aims to detect adverse drug events (ADEs) from conversations by extracting medicine names and symptoms, then matching them with the FAERS dataset to identify known adverse events.

## Project Structure

```
├── README.md                 # Project documentation
├── data/                    # Directory for storing datasets
│   ├── raw/                 # Raw FAERS datasets
│   └── processed/           # Processed datasets
├── src/                     # Source code
│   ├── data_processing/     # Scripts for data processing
│   │   ├── extract_faers.py # Extract relevant FAERS data
│   │   └── preprocess.py    # Preprocess extracted data
│   ├── extraction/          # Entity extraction modules
│   │   ├── medicine_extractor.py  # Extract medicine names using Bioformer-8L
│   │   └── symptom_extractor.py   # Extract symptoms using NER
│   ├── matching/            # Matching extracted entities with FAERS
│   │   └── faers_matcher.py # Match medicines and symptoms with FAERS
│   ├── model/               # ML model for severity prediction
│   │   ├── train.py         # Train the model
│   │   └── predict.py       # Make predictions using the trained model
│   └── utils/               # Utility functions
│       └── severity.py      # Functions for severity categorization
└── requirements.txt         # Project dependencies
```

## Approach

### Step 1: Extract Relevant FAERS Data

Extract key datasets from OpenFAERS:
- DRUG Dataset: primaryid, caseid, drugname
- REAC Dataset: primaryid, caseid, pt (reaction term)
- OUTC Dataset: primaryid, caseid, outc_cod (outcome code)

### Step 2: Data Processing & Preprocessing

- Extract Medicine & Symptoms from conversations
  - Use Whisper for transcription
  - Use Bioformer-8L API to extract medicine names
  - Use NER from transformers to extract symptoms
- Map extracted entities to FAERS data

### Step 3: Categorization of Severity

Categorize adverse events into 3 levels:
- Critical (⚠️🚨): DE (Death), LT (Life-Threatening), HO (Hospitalization)
- Near-Critical (⚠️): DS (Disability), CA (Congenital Anomaly), RI (Required Intervention)
- Needs Attention (⚠): OT (Other Serious Events)

### Step 4: Model Training

Train a supervised learning model to predict severity labels based on:
- Medicine Name
- Symptoms
- Past FAERS Data (Outcomes & Adverse Effects)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Download FAERS data and place it in the `data/raw/` directory
2. Run data extraction and preprocessing:
   ```
   python src/data_processing/extract_faers.py
   python src/data_processing/preprocess.py
   ```
3. Train the model:
   ```
   python src/model/train.py
   ```
4. Make predictions:
   ```
   python src/model/predict.py
   ```