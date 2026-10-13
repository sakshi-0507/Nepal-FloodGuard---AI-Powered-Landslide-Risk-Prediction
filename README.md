# 🌧️ Nepal FloodGuard — Landslide Susceptibility Prediction

A machine learning project that predicts **landslide susceptibility in Nepal** from terrain, rainfall, land-use and distance features, with an interactive **Streamlit** dashboard.

> ⚠️ **Research prototype only.** Not an official warning or emergency-response system. Do not use it for evacuation, travel, or safety decisions.

---

## 📌 Overview (short)

- **Goal:** classify a location as landslide (`1`) or non-landslide (`0`).
- **Data:** landslide points + non-landslide points (Nepal), merged into one labelled dataset (~610 samples).
- **Features:** Slope, Aspect, Curvature, TWI, Elevation, Rain, River distance, Road distance, Land use (LULC), etc.
- **Model:** Random Forest inside a full scikit-learn pipeline (imputation + one-hot encoding), compared against Logistic Regression and XGBoost.
- **Result (Random Forest, 20% test set):** Accuracy **81.2%**, Landslide precision 0.76, recall 0.67, F1 0.71.
- **App:** Streamlit dashboard where you enter feature values and get a predicted class and class-1 probability.

---

## 🗂️ Project Structure

```
Nepal-FloodGuard/
├── app.py                                   # Streamlit dashboard
├── Nepal_FloodGuard___AI-Powered_Landslide_Risk_Prediction.ipynb   # Data prep, EDA, training
├── floodguard_models/
│   ├── landslide_model.pkl                  # Trained pipeline (preprocessing + Random Forest)
│   └── X_train_reference.pkl                # Training columns/values used to build app inputs
└── README.md
```

> The raw `.xls` datasets are not included in this repo. Place your own copies locally and update the file paths in the notebook.

---

## 🔄 How It Works (Steps)

1. **Load data** — read the landslide and non-landslide Excel files.
2. **Label & merge** — add `landslide = 1` / `0` and concatenate into one DataFrame.
3. **Clean data**
   - Merge `Elevation_` into `Elevation`.
   - Drop ID columns (`OBJECTID`, `OID`) and the duplicate `Landslide` column.
   - Fill missing `Slope` and `TWI` with the median.
   - Fill missing `Aspect` with a **circular mean** (since 359° ≈ 1°).
4. **Explore (EDA)** — class balance, slope boxplot, rainfall histogram, correlation heatmap, land-use counts, outlier check.
5. **Prepare features** — separate target, keep `X_cord`/`Y_cord` aside for mapping, split numeric vs categorical columns.
6. **Build preprocessing** — median imputer for numeric; most-frequent imputer + one-hot encoder for categorical.
7. **Train/test split** — 80/20, stratified, `random_state=42`.
8. **Train & compare models** — Logistic Regression, Random Forest, XGBoost; pick the best by landslide F1.
9. **Evaluate** — accuracy, classification report, confusion matrix, feature importance, prediction scatter map.
10. **Save artifacts** — export the full pipeline and training reference data with `joblib`.
11. **Serve** — `app.py` loads the artifacts and builds sidebar inputs automatically from the training columns.

---

## 🚀 Run Locally

```bash
# 1. Clone
git clone https://github.com/<your-username>/Nepal-FloodGuard.git
cd Nepal-FloodGuard

# 2. (Optional) virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate model files (run the notebook's final "save" cell)
#    -> creates floodguard_models/landslide_model.pkl and X_train_reference.pkl

# 5. Start the dashboard
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

### `requirements.txt`

```
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
plotly
xgboost
xlrd
openpyxl
```

> Use the same scikit-learn version for training and running the app, or the `.pkl` file may fail to load.

---

## 🖥️ Using the Dashboard

1. Enter environmental values in the **sidebar** (defaults are training-data medians).
2. Click **Predict Landslide Susceptibility**.
3. View the **predicted class** and the **class-1 probability**.

The probability is the model's estimate, not a validated real-world chance of a landslide.

---

## ⚠️ Limitations

- Small dataset (~610 samples) and a single train/test split; no spatial cross-validation.
- Landslide recall is 0.67, so about one third of landslides are missed.
- Predicts **susceptibility** only, not timing or triggers.
- No flood prediction yet despite the project name.

## 🔮 Future Work

- Flood-risk prediction with suitable flood data
- GIS-based susceptibility maps
- Road disruption analysis
- Live rainfall/weather integration
- Spatial cross-validation and hyperparameter tuning

---

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · Plotly · Streamlit · joblib

## 👤 Author

**Your Name** — [GitHub](https://github.com/<your-username>)
