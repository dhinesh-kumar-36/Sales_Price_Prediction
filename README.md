# 📈 Sales Prediction Using Linear Regression

A machine learning project that predicts product **Sales** based on advertising budgets spent on **TV**, **Radio**, and **Newspaper** channels using a **Linear Regression** model.

The dataset (`advertising.csv`) is automatically loaded and cleaned. If the dataset is not found, the script generates a realistic synthetic dataset automatically.

---

## 🚀 Features

- Automatic dataset loading or synthetic dataset generation
- Data cleaning and preprocessing
- Correlation heatmap visualization
- Scatter plot analysis
- Linear Regression model training
- Model evaluation using R², MAE, MSE, and RMSE
- Actual vs Predicted visualization
- Sample prediction testing
- Model saving using Pickle

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pickle

---

## 📂 Project Structure

```

sales-prediction/
│
├── advertising.csv
├── sales_prediction.py
├── correlation_heatmap.png
├── scatter_plots.png
├── actual_vs_predicted.png
├── saved_model/
│   └── sales_model.pkl
└── README.md

```

---

## ▶️ Installation

Install required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## ▶️ Run the Project

```bash
python sales_prediction.py
```

---

## 📊 Model Formula

Sales Prediction Equation:

Sales = 0.046 × TV + 0.188 × Radio + 0.004 × Newspaper + 2.9

---

## 📈 Evaluation Metrics

The model evaluates performance using:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

## 💾 Saved Model

The trained model is saved in:

```

saved_model/sales_model.pkl

```

You can reload the model anytime using Pickle.

---

## 🧪 Sample Prediction

Input:

- TV Budget = 200
- Radio Budget = 30
- Newspaper Budget = 50

Predicted Sales:

```

Predicted Sales ≈ 17–18 units

```

---

## 📌 Future Improvements

- Add Streamlit or Flask web application
- Use advanced ML algorithms
- Deploy the model online
- Add real-time user input prediction

---

## 👨‍💻 Author

Developed as a Machine Learning project using Python and Scikit-learn.
