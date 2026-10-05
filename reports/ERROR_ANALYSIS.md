# SAMATRIX RESUMEFORGE 2026 — Diagnostic Error Analysis

**Target Model:** `Calibrated LinearSVC (Word+Char TF-IDF)`  
**Validation Accuracy:** **0.7131**  
**Validation Macro-F1:** **0.6899**  

---

## 1. Per-Class Performance Breakdown

| Professional Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `AVIATION` | 0.8889 | 0.8889 | **0.8889** | 18 |
| `HR` | 0.8333 | 0.9375 | **0.8824** | 16 |
| `CONSTRUCTION` | 0.8000 | 0.9412 | **0.8649** | 17 |
| `DESIGNER` | 0.7500 | 0.9375 | **0.8333** | 16 |
| `BANKING` | 0.8235 | 0.8235 | **0.8235** | 17 |
| `FITNESS` | 0.7619 | 0.8889 | **0.8205** | 18 |
| `INFORMATION-TECHNOLOGY` | 0.7083 | 0.9444 | **0.8095** | 18 |
| `ENGINEERING` | 0.7778 | 0.7778 | **0.7778** | 18 |
| `CHEF` | 0.8125 | 0.7222 | **0.7647** | 18 |
| `PUBLIC-RELATIONS` | 0.6667 | 0.8750 | **0.7568** | 16 |
| `TEACHER` | 0.6500 | 0.8667 | **0.7429** | 15 |
| `ADVOCATE` | 0.8000 | 0.6667 | **0.7273** | 18 |
| `ACCOUNTANT` | 0.6250 | 0.8333 | **0.7143** | 18 |
| `FINANCE` | 0.7857 | 0.6111 | **0.6875** | 18 |
| `AGRICULTURE` | 0.8333 | 0.5556 | **0.6667** | 9 |
| `BUSINESS-DEVELOPMENT` | 0.6471 | 0.6111 | **0.6286** | 18 |
| `HEALTHCARE` | 0.6111 | 0.6111 | **0.6111** | 18 |
| `DIGITAL-MEDIA` | 0.7000 | 0.5000 | **0.5833** | 14 |
| `AUTOMOBILE` | 1.0000 | 0.4000 | **0.5714** | 5 |
| `APPAREL` | 0.5833 | 0.5000 | **0.5385** | 14 |
| `BPO` | 1.0000 | 0.3333 | **0.5000** | 3 |
| `SALES` | 0.4500 | 0.5000 | **0.4737** | 18 |
| `ARTS` | 0.5455 | 0.4000 | **0.4615** | 15 |
| `CONSULTANT` | 0.6000 | 0.3333 | **0.4286** | 18 |

---

## 2. Most Confused Category Pairs

| True Category | Predicted Category | Error Count | Class Error Rate (%) |
| :--- | :--- | :---: | :---: |
| `FINANCE` | `ACCOUNTANT` | **5** | 27.8% |
| `CONSULTANT` | `ACCOUNTANT` | **3** | 16.7% |
| `SALES` | `APPAREL` | **3** | 16.7% |
| `APPAREL` | `HR` | **2** | 14.3% |
| `ADVOCATE` | `HEALTHCARE` | **2** | 11.1% |
| `AVIATION` | `ENGINEERING` | **2** | 11.1% |
| `BUSINESS-DEVELOPMENT` | `CONSULTANT` | **2** | 11.1% |
| `CONSULTANT` | `INFORMATION-TECHNOLOGY` | **2** | 11.1% |
| `CHEF` | `TEACHER` | **2** | 11.1% |
| `FITNESS` | `SALES` | **2** | 11.1% |

---

## 3. Qualitative Inspection of Misclassified Samples

#### Case Study 1: True `APPAREL` vs Predicted `HR`
- **Resume Snippet:** *"CFO ASSISTANT/EXECUTIVE ADMINISTRATOR/HR MANAGER/CS           Professional Summary    To apply myself in a new and challenging position with a progressive organization for long-term employment. Organized, deadline-oriented, great attention t..."*
- **Word Length:** 837 words
- **Diagnosis:** Genuine cross-functional corporate terminology overlap.

#### Case Study 2: True `HEALTHCARE` vs Predicted `ARTS`
- **Resume Snippet:** *"OWNER       Summary     Results-oriented individual with diverse background in management and customer service. Dedicated to providing excellent customer service and Strong work ethic, professional demeanor and great initiative.         High..."*
- **Word Length:** 809 words
- **Diagnosis:** Genuine cross-functional corporate terminology overlap.

#### Case Study 3: True `ARTS` vs Predicted `PUBLIC-RELATIONS`
- **Resume Snippet:** *"MARKETING MANAGER       Summary    To use my skills, knowledge and enthusiasm to advance the public image and credibility of a business-driven company, in a manner consistent with its existing core values. Almost twenty years of experience i..."*
- **Word Length:** 692 words
- **Diagnosis:** Genuine cross-functional corporate terminology overlap.

#### Case Study 4: True `BUSINESS-DEVELOPMENT` vs Predicted `ENGINEERING`
- **Resume Snippet:** *"ASSOCIATE DIRECTOR BUSINESS DEVELOPMENT       Summary    Persuasive business development professional, successful at establishing and maintaining key partnerships with corporate decision makers. Offering more than 12 years of successful corp..."*
- **Word Length:** 1218 words
- **Diagnosis:** Genuine cross-functional corporate terminology overlap.

#### Case Study 5: True `SALES` vs Predicted `DESIGNER`
- **Resume Snippet:** *"SALES ASSOCIATE           Experience     04/2016   to   Current     Sales Associate    Company Name   －   City  ,   State      Help customers with their pet problems and assist them in choosing the right products for their pets.         06/2..."*
- **Word Length:** 169 words
- **Diagnosis:** Genuine cross-functional corporate terminology overlap.

#### Case Study 6: True `DIGITAL-MEDIA` vs Predicted `INFORMATION-TECHNOLOGY`
- **Resume Snippet:** *"SPRINT ISP MANAGEMENT TO THE VENDOR             Qualifications          Windows 95-XP-Windows 7/8.8,1/10  Windows NT/2000/2003/2008/2012  Red Hat (limited)  Ubuntu (limited) VIRTUALIZATION TECHNOLOGY:  ESX/ESXi 3.5-5.5  MS Hyperv 2005-2008 S..."*
- **Word Length:** 811 words
- **Diagnosis:** Genuine cross-functional corporate terminology overlap.

