# EconoCausal — Dynamic Pricing via Double Machine Learning



**Project:** EconoCausal  

**Focus:** Causal Machine Learning, Dynamic Pricing and Budget Optimization  

**Application:** Synthetic Retail Discount Optimization



## 1. Project Overview



EconoCausal is a causal machine learning application that estimates how different discount amounts influence customer conversion probabilities.



Traditional predictive models estimate which customers are likely to purchase. However, a high purchase probability does not necessarily mean that offering a discount will cause a customer to purchase.



EconoCausal uses **Double Machine Learning (DML)** to estimate the causal impact of discounts and identify differences in treatment effects across customers. It also uses Mixed-Integer Linear Programming (MILP) to recommend discount allocations under a specified budget.



The application provides a FastAPI backend and an interactive React dashboard for exploring causal estimates and budget optimization results.



## 2. Project Objectives



\- Estimate the causal effect of retail discounts on customer conversion.

\- Analyze heterogeneous treatment effects across customer characteristics.

\- Evaluate model accuracy and robustness.

\- Optimize discount allocation under a business budget.

\- Present model estimates and optimization results through an interactive dashboard.



## 3. Technology Stack



| Area | Technologies |

|---|---|

| Programming | Python |

| Data processing | Pandas, NumPy |

| Machine learning | Scikit-learn, LightGBM |

| Causal machine learning | EconML, Double Machine Learning |

| Causal analysis | DoWhy |

| Optimization | SciPy MILP |

| Backend | FastAPI, Uvicorn |

| Frontend | React, Vite |

| Visualization | Plotly |

| Testing | Pytest |



## 4. System Architecture



The project follows this workflow:



1. **Data generation:** Generate a synthetic retail customer dataset.

2. **Data validation:** Check dataset quality, eligibility and required columns.

3. **Preprocessing:** Prepare customer features and treatment/outcome variables.

4. **Causal modeling:** Train a continuous-treatment Double Machine Learning model.

5. **Effect estimation:** Estimate the effect of different discount levels.

6. **Evaluation:** Validate dose-level effects, effect ranking and model robustness.

7. **Budget optimization:** Use SciPy MILP to select discount allocations under a budget.

8. **API integration:** Expose model and optimization functionality through FastAPI.

9. **Dashboard:** Display estimates, customer inputs and optimization results through React and Plotly.



## 5. Dataset



The primary dataset is synthetically generated for development and evaluation.



\- Generated dataset: `data/processed/retail_synthetic.csv`

\- Original generated dataset size: 20,000 rows.

\- Eligible customer records used for modeling: 17,990.

\- Treatment variable: `discount_amount`.

\- Outcome variable: `conversion`.

\- Discount levels: 0, 5, 10 and 20.

\- Customer features include age, income, tenure, past purchases, average basket value, region and membership status.



The dataset includes synthetic ground-truth treatment effects for evaluation. These ground-truth fields are excluded from model input features to prevent target leakage.



**Important:** This is synthetic data, not real customer or retail campaign data.



## 6. Double Machine Learning Model



The main model uses EconML's `LinearDML` with continuous treatment.



| Configuration | Value |

|---|---|

| Estimator | LinearDML |

| Treatment | `discount_amount` |

| Outcome | `conversion` |

| Nuisance models | Gradient Boosting |

| Cross-validation | 3 folds |

| Random seed | 42 |

| Evaluation discount | 10 |



The model estimates heterogeneous causal effects while accounting for observed customer characteristics and treatment assignment.



## 7. Model Evaluation



The continuous-treatment DML model was evaluated against synthetic ground-truth effects.



### Dose-level evaluation



| Discount | Mean predicted effect | Mean true effect | MAE |

|---:|---:|---:|---:|

| 5 | 0.03936 | 0.03110 | 0.00853 |

| 10 | 0.07873 | 0.06220 | 0.01706 |

| 20 | 0.15746 | 0.12441 | 0.03413 |



At discount level 10:



\- Mean Absolute Error (MAE): **0.01706**

\- Root Mean Squared Error (RMSE): **0.02107**

\- Mean predicted effect: **0.07873**

\- Mean synthetic true effect: **0.06220**



### Effect ranking evaluation



\- Pearson correlation: **0.7888**

\- Spearman correlation: **0.7761**

\- Top 10% overlap: **0.5528**



### Model robustness



The Gradient Boosting nuisance-model configuration was compared with a linear nuisance-model configuration.



\- Correlation between estimated effects: **0.9817**

\- Top 10% customer overlap: **0.8468**



These results provide evidence of model consistency on the synthetic evaluation data. They do not establish real-world causal validity.



Detailed evaluation results are available in:



`reports/EconoCausal_Model_Evaluation_Report.md`



## 8. Budget-Constrained Discount Optimization



The project uses **SciPy Mixed-Integer Linear Programming (MILP)** to select discount recommendations under an expected discount-spend budget.



The optimizer aims to maximize estimated incremental contribution while respecting budget and customer allocation constraints.



The validation experiment used:



\- Eligible customers considered: 17,990

\- Budget: 1,000

\- Assumed contribution margin: 30%

\- Customers offered discounts: 1,232

\- Expected model discount spend: approximately 999.99

\- True synthetic discount spend: approximately 934.95

\- True synthetic incremental contribution: approximately 205.78



The synthetic oracle benchmark achieved approximately 235.21 in incremental contribution. It is an idealized benchmark using ground-truth information, not a deployable model.



All monetary values in these experiments are based on the synthetic dataset and its business assumptions.



## 9. Backend API



The FastAPI backend exposes the following endpoints:



| Method | Endpoint | Purpose |

|---|---|---|

| GET | `/api/v1/retail/model/status` | Check model status |

| POST | `/api/v1/retail/effect` | Estimate customer-level causal effects |

| POST | `/api/v1/retail/optimize` | Optimize discount allocation |



Interactive API documentation is available at:



`http://127.0.0.1:8000/docs`



## 10. Interactive Dashboard



The React dashboard provides:



\- Customer feature inputs.

\- Discount-level causal effect estimation.

\- Budget and contribution assumptions.

\- Discount allocation recommendations.

\- Interactive Plotly visualizations.

\- Budget allocation and optimization summaries.



The frontend communicates with the FastAPI backend through the `/api` proxy configured in Vite.



## 11. Project Setup



### Prerequisites



\- Python 3.12

\- Node.js and npm

\- Git (optional, for repository management)



### Step 1: Clone the repository



```powershell

git clone <YOUR_REPOSITORY_URL>

cd EconoCausal

```



If the project is already downloaded, open PowerShell in its root directory instead.



### Step 2: Create and activate the Python environment



```powershell

python -m venv Econoenv

.\\Econoenv\\Scripts\\Activate.ps1

```



### Step 3: Install Python dependencies



```powershell

python -m pip install --upgrade pip

pip install -r requirements.txt

```



### Step 4: Start the backend



Run from the project root:



```powershell

uvicorn backend.app.main:app --reload

```



Backend address:



`http://127.0.0.1:8000`



API documentation:



`http://127.0.0.1:8000/docs`



### Step 5: Start the frontend



Open a second PowerShell terminal:



```powershell

cd frontend

npm install

npm run dev

```



Open the dashboard:



`http://localhost:5173`



Keep both backend and frontend terminals running while using the application.



## 12. Testing and Verification



Run the backend test suite from the project root:



```powershell

python -m pytest -v

```



**Latest verified result: 52 passed, 0 failed.**



Build the frontend for production:



```powershell

cd frontend

npm run build

```



The frontend build completed successfully during project verification. It produced a large JavaScript bundle warning, which is a performance optimization consideration rather than a build failure.



## 13. Project Structure



```text

EconoCausal/

│

├── backend/

│   └── app/

│       ├── api/

│       ├── causal_engine/

│       ├── core/

│       ├── dml/

│       ├── eda/

│       ├── evaluation/

│       ├── graph/

│       ├── preprocessing/

│       ├── propensity/

│       └── validation/

│

├── data/

│   ├── raw/

│   ├── processed/

│   └── sample/

│

├── docs/

├── frontend/

│   └── src/

├── reports/

├── scripts/

├── tests/

├── archive/

│

├── .env.example

├── .gitignore

├── LICENSE

├── pytest.ini

├── README.md

└── requirements.txt

```



## 14. Assumptions and Limitations



\- The primary retail dataset is synthetic.

\- Causal estimates depend on observed confounders, treatment overlap and modeling assumptions.

\- Estimated treatment effects are not guaranteed individual customer outcomes.

\- Budget optimization depends on estimated effects and assumed business margins.

\- Synthetic ground-truth performance does not establish performance on real retail campaigns.

\- Real-world validation with actual retail campaign data is required before business deployment.



## 15. Conclusion



EconoCausal demonstrates an end-to-end causal machine learning workflow for retail discount analysis.



It combines continuous-treatment Double Machine Learning, heterogeneous treatment effect evaluation, budget-constrained optimization, API integration and an interactive dashboard.



The project illustrates how causal estimation can support discount allocation decisions beyond conventional purchase prediction, while making the assumptions and limitations of synthetic evaluation explicit.





