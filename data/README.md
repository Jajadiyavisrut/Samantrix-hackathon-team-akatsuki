# Resume Classification Dataset Storage

This directory holds the raw and processed resume data assets.

## Structure
- `raw/`: Contains original `Resume.csv` and `Resume.xlsx`.
- `processed/`: Contains preprocessed and stratified train/val/test splits:
  - `cleaned_resumes.csv`: Deduplicated, valid records (2,481 records).
  - `train.csv`: 70% stratified training split (1,736 records).
  - `val.csv`: 15% stratified validation split (372 records).
  - `test.csv`: 15% untouched test split (373 records).
