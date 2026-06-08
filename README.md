# Case Study 2 — Property Matching System

## Folder Structure
```
case_study_2/
├── README.md                  ← You are here
├── property_match.py          ← Main Python solution
├── data.xlsx                  ← Input data (User & Property datasets)
├── case_study_brief.docx      ← Original case study brief
│
├── outputs/
│   ├── all_match_scores.csv   ← All 750 user-property match scores
│   ├── top_match_results.csv  ← Top-3 matches per user
│   ├── heatmap_full.png       ← Match score heatmap (25 users × 30 props)
│   ├── top_matches_bar.png    ← Bar chart: top-3 per user
│   ├── score_distribution.png ← Histogram of all scores
│   └── num_vs_text_scatter.png← Numerical vs text score scatter
```

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn openpyxl matplotlib seaborn
```

### 2. Run the solution
```bash
python property_match.py
```

## Methodology
| Component        | Technique                          | Weight |
|------------------|------------------------------------|--------|
| Numerical Score  | MinMax norm → Cosine Similarity    | 60%    |
| Text Score       | TF-IDF (bigrams) → Cosine Similarity | 40%  |
| **Match Score**  | Weighted blend, scaled to 0–100    | —      |

## Results Summary
- Total pairs evaluated : 750 (25 users × 30 properties)
- Score range           : 0.1 – 63.1
- Average score         : ~50.0
- Best match            : User 21 × Property 18 (63.1 / 100)
