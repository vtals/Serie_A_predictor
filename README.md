# Serie A Predictor ⚽

This repository provides a **machine learning and data visualization project** focused on predicting the **Serie A league table for the 2025-2026 season** using historical match data.  
It combines **data preprocessing, feature engineering, and predictive modeling** with **visual exploratory analysis**.

---

## 📂 Project Structure

- **`Serie_A_predictor.py`**  
  Main Python script that:
  - Loads the dataset `matches_serie_A.csv`
  - Cleans and preprocesses the data  
  - Generates additional season-based CSV datasets saved in the `Season_data/` folder  
  - Trains a **Random Forest model** to predict the final league standings for the 2025-2026 season  

- **`Serie_A_visualizations_notebook.ipynb`**  
  Jupyter Notebook used for **data visualization and analysis**.  
  It includes:
  - Exploratory plots created with **Matplotlib** and **Seaborn**  
  - Commentary and insights for each visualization  
  - Exported plots saved in the `Plot/` folder  

- **`matches_serie_A.csv`**  
  Input dataset containing historical Serie A match results used for training and analysis.  

- **`Season_data/`**  
  Auto-generated folder containing season-specific processed datasets.  

- **`Plot/`**  
  Folder containing exported data visualizations.  

---

## 🔧 Installation & Requirements

Make sure you have **Python 3.8+** installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

Typical requirements include:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter`

---

## 🚀 Usage

1. **Run the predictor script**  
   ```bash
   python Serie_A_predictor.py
   ```
   This will clean the dataset, generate season-based CSV files, and predict the Serie A table for 2025-2026.

2. **Explore data visualizations**  
   Open the Jupyter Notebook:
   ```bash
   Serie_A_visualizations_notebook.ipynb
   ```
   The generated plots are also saved inside the `Plot/` folder.

---

## 📊 Model

The predictive model is based on a **Random Forest Classifier** trained on past Serie A seasons.  
The pipeline includes:
- Data cleaning & preprocessing  
- Feature engineering (team statistics, match results, season summaries)  
- Model training & prediction of league standings  

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open a pull request to improve the model, dataset, or visualizations.

---

## 📜 License

This project is released under the **MIT License**.
