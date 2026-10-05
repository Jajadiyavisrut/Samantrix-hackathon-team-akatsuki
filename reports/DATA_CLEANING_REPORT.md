# SAMATRIX RESUMEFORGE 2026 — Data Cleaning & Leakage Protection Report

**Generated:** 2026-10-05  
**Artifact Status:** Production Cleaned Dataset Verified  

---

## 1. Row Accounting & Audit Trail

| Step | Description | Rows Remaining | Delta |
| :--- | :--- | :---: | :---: |
| **Raw Ingestion** | Initial `Resume.csv` loaded | 2,484 | - |
| **Corrupted Filter** | Removal of empty/corrupted whitespace resumes | 2,483 | -1 |
| **Exact Deduplication** | Removal of duplicate `Resume_str` texts (retaining first occurrence) | 2,481 | -2 |
| **Final Clean Output** | Persisted to `reports\DATA_CLEANING_REPORT.md` | **2,481** | **-3 total** |

---

## 2. Detailed Records Removed

### A. Corrupted Records Removed (1 record)
- **ID `12632728`**: Category: `BUSINESS-DEVELOPMENT`, Words: 0 (Empty whitespace placeholder)


### B. Duplicate Records Removed (2 records)
- **ID `28398216`**: Category: `FINANCE`, Duplicate snippet: `         FINANCE OFFICER         Professional Summary    To ...`
- **ID `37473139`**: Category: `AVIATION`, Duplicate snippet: `         STOREKEEPER II       Professional Summary    The pu...`


---

## 3. Final Class Distribution (Clean Dataset)

Total Samples: **2,481** | Categories: **24**

| Category | Clean Sample Count | Percentage |
| :--- | :---: | :---: |
| `INFORMATION-TECHNOLOGY` | 120 | 4.84% |
| `BUSINESS-DEVELOPMENT` | 119 | 4.80% |
| `ADVOCATE` | 118 | 4.76% |
| `CHEF` | 118 | 4.76% |
| `ENGINEERING` | 118 | 4.76% |
| `ACCOUNTANT` | 118 | 4.76% |
| `FINANCE` | 117 | 4.72% |
| `FITNESS` | 117 | 4.72% |
| `SALES` | 116 | 4.68% |
| `AVIATION` | 116 | 4.68% |
| `BANKING` | 115 | 4.64% |
| `HEALTHCARE` | 115 | 4.64% |
| `CONSULTANT` | 115 | 4.64% |
| `CONSTRUCTION` | 112 | 4.51% |
| `PUBLIC-RELATIONS` | 111 | 4.47% |
| `HR` | 110 | 4.43% |
| `DESIGNER` | 107 | 4.31% |
| `ARTS` | 103 | 4.15% |
| `TEACHER` | 102 | 4.11% |
| `APPAREL` | 97 | 3.91% |
| `DIGITAL-MEDIA` | 96 | 3.87% |
| `AGRICULTURE` | 63 | 2.54% |
| `AUTOMOBILE` | 36 | 1.45% |
| `BPO` | 22 | 0.89% |
