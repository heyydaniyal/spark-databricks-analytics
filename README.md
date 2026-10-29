# Big Data Analysis Project — NOVA IMS 2026/27

Topic: _TBD (Mental Health / Asian Restaurant / Churn)_
Group: GX — rename `GX_` prefixes to your group letter.
Deadline: 11 Dec 2026 23:59 (Moodle, `GX_project.zip`)

## Structure
```
big-data-project/
├── data/
│   ├── raw/            # original CSVs (not committed)
│   └── processed/      # cleaned outputs (not committed)
├── notebooks/          # graded notebooks, run in order
│   ├── GX_00_setup.ipynb
│   ├── GX_01_ingestion.ipynb
│   ├── GX_02_data_engineering.ipynb
│   ├── GX_03_eda_sparksql.ipynb
│   ├── GX_04_ml.ipynb
│   ├── GX_05_graphframes.ipynb
│   └── GX_06_streaming_bonus.ipynb
├── src/                # shared helper code (.py allowed in submission)
│   ├── config.py       # topic switch, paths
│   ├── io.py           # loading
│   ├── cleaning.py
│   ├── features.py
│   ├── graph.py
│   ├── ml.py
│   └── utils.py
├── presentation/       # deck source + GX_presentation.pdf
├── docs/
│   └── decisions.md    # log of key decisions (for the Q&A)
└── submission/         # final GX_project.zip (not committed)
```

## Submission must contain only
- Executed `GX_*.ipynb` with visible outputs
- Optional `.py` files from `src/`
- `GX_presentation.pdf`
