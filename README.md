# Project SIA - Driving test pass rate analysis

## Project overview
This project investigates the factors influencing the **probability of passing a UK driving test**, with a focus on a fictional 22-year-old candidate named **SIA**.

We analyze whether **location**, **gender**, and **age** affect test outcomes across two driving test centers:

- **Bletchley** (small town)
- **Wood Green (London)** (urban center)

Using logistic regression modeling, we explore demographic and contextual differences in pass rates across years.

---

## Objectives
- Load, clean, and preprocess DVSA test data (Excel format)
- Perform Exploratory Data Analysis on pass rates
- Fit **Logit** and **Probit** regression models
- Interpret regression summaries and visual diagnostics
- Test model robustness and predict candidate outcomes
---

## Project Structure
```
Project_SIA_Driving_Test/
├── assets/                    # Image files (for notebook illustration)
│   └── Linear vs Logistic Model.png
├── data/                     # Contains local Excel data file (ignored in Git)
│   └── DataDSVA_Python.xlsm
├── scripts/                  # Custom Python modules
│   ├── __init__.py
│   ├── data_processing.py    # load_data(), preprocess_data()
│   ├── eda.py                # exploratory_data_analysis()
│   ├── evaluation.py         # evaluate_models(), predict_probabilities(), robustness_check()
│   └── modelling.py          # train_models()
├── tests/                    # Unit & integration tests
│   ├── test_modelling.py
│   └── test_processing.py
├── main_notebook.ipynb       # Main analysis notebook with narration
├── .gitignore                # Excludes data files and temp folders
├── README.md
└── requirements.txt          # Dependencies
```

---

## How to Run the Project
1. **Install Conda environment**:
```bash
conda create -n sia_driving python=3.10
conda activate sia_driving
pip install -r requirements.txt
```

2. **Add your local Excel data file** (not versioned):
   - Place `DataDSVA_Python.xlsm` inside the `/data` folder.

3. **Launch Jupyter Notebook**:
```bash
jupyter notebook main_notebook.ipynb
```

4. **Run tests**:
```bash
python -m unittest discover tests/
```

---

## Key methods
| Module | Function | Purpose |
|--------|----------|---------|
| `data_processing.py` | `load_data()` | Load raw Excel sheets and extract observations |
| | `preprocess_data()` | Clean and transform to long-format, encode variables |
| `eda.py` | `exploratory_data_analysis()` | Generate visualizations (histograms, boxplots) |
| `modelling.py` | `train_models()` | Train logistic and probit models |
| `evaluation.py` | `evaluate_models()` | Show regression summaries and AUC score |
| | `predict_probabilities()` | Predict success probabilities for sample profiles |
| | `robustness_check()` | Compare model on 80% subsample |

---

## Notes
- The dataset is large and excluded from GitHub. Please contact the me if problem to access.
- Some charts and regression results are visualized directly in the notebook.
- The analysis is exploratory and acknowledges limitations such as aggregation and omitted variables.

---

## Author
Camille Lemaignen  
Project completed as part of the **SIA Python & Statistics course (2025)**  
Deadline: March 27, 2025 @ 23:59

---
> *Thank you for reviewing the project — all feedback is welcome!*