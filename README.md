# 📱 AI-Based Smartphone Decision Support System

A decision support system that helps users choose a smartphone in two ways:

1. **Rule-based recommender** — enter your budget and priorities such as gaming, camera, and battery, and the system recommends suitable smartphones from a real-world dataset.
2. **ML-based predictor** — enter a smartphone's specifications, and trained Machine Learning models predict its **price (₹)** and **market category** (Budget / Mid-Range / Premium).

Built with **Python, pandas, scikit-learn, and Streamlit**.

---

## 📁 Project Structure

```text
.
├── Proj_SmartPhone_DSS.ipynb   # EDA, cleaning, and model experimentation
├── train.py                    # Data processing, model training, and saving
├── app.py                      # Streamlit recommendation and prediction app
├── requirements.txt            # Required Python libraries
├── .gitignore                  # Files excluded from Git
└── models/                     # Saved models, encoders, scaler, and dataset
```

The notebook contains the exploratory work such as EDA, data cleaning, and model experimentation.

`train.py` contains the cleaned training pipeline. It downloads the dataset, processes the data, trains the models, and saves the required artifacts inside the `models/` folder.

`app.py` loads these saved artifacts and provides the user interface through Streamlit.

---

## 📊 Dataset

**Smartphones Cleaned Dataset**

Source: Kaggle — Smartphones Cleaned Dataset

The dataset contains smartphone information such as:

* Brand
* Model
* Price
* RAM
* Storage
* Processor
* Battery
* Camera
* Display
* Refresh rate
* Operating system
* Other smartphone features

The dataset is downloaded automatically in `train.py` using `kagglehub`.

---

## ⚙️ How It Works

### 🔍 1. Smartphone Recommendation

The user enters:

* Preferred budget
* Minimum RAM
* Minimum storage
* Gaming priority
* Camera priority
* Battery priority

The system converts the priority levels into minimum requirements and filters the smartphone dataset.

The matching smartphones are then sorted by **rating score**, and the top 5 results are displayed.

---

### 🤖 2. Price & Category Prediction

The user enters smartphone specifications such as:

* Brand
* Processor brand
* Operating system
* RAM
* Storage
* Battery
* Camera
* Display
* Refresh rate
* Other features

The input is converted into the same format used during model training.

The trained models then make two predictions:

**Price Prediction**

```text
Linear Regression
        ↓
Predicted Smartphone Price
```

**Category Prediction**

```text
Logistic Regression
        ↓
Budget / Mid-Range / Premium
```

---

## 🧠 Machine Learning Models

| Task                | Algorithm           | Target           |
| ------------------- | ------------------- | ---------------- |
| Price Prediction    | Linear Regression   | `price_inr`      |
| Category Prediction | Logistic Regression | `phone_category` |

### Data Preprocessing

The project uses:

* **One-hot encoding** for smartphone brand and processor brand
* **Label encoding** for operating system and phone category
* **StandardScaler** for feature scaling

The category is created using these price ranges:

| Category  | Price             |
| --------- | ----------------- |
| Budget    | Below ₹20,000     |
| Mid-Range | ₹20,000 – ₹40,000 |
| Premium   | Above ₹40,000     |

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Streamlit**
* **Matplotlib**
* **Seaborn**
* **KaggleHub**

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Install required libraries

```bash
pip install -r requirements.txt
```

### 3. Train the models

```bash
python train.py
```

This downloads the dataset, performs preprocessing, trains the Machine Learning models, evaluates them, and saves the required files inside the `models/` folder.

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📂 Saved Model Files

The `models/` folder contains the files required by the Streamlit application, including:

* `price_model.pkl`
* `category_model.pkl`
* `scaler.pkl`
* `os_encoder.pkl`
* `category_encoder.pkl`
* `feature_columns.pkl`
* `brands.pkl`
* `processor_brands.pkl`
* `os_names.pkl`
* `phones_original.csv`

These files allow `app.py` to make predictions without retraining the models every time.

---

##  Important Note

The project uses KaggleHub to download the dataset during training.

If Kaggle authentication is required, configure your Kaggle credentials locally.

**Never upload `kaggle.json` or any API credentials to GitHub.**

These files are excluded through `.gitignore`.

---

##  Future Improvements

Possible future improvements include:

* Improving the recommendation algorithm
* Adding more smartphone preferences
* Adding visualizations to the Streamlit app
* Comparing additional Machine Learning models
* Improving prediction accuracy
* Adding a more advanced recommendation scoring system

---

##  Project

**AI-Based Smartphone Decision Support System**

A Machine Learning project combining a rule-based recommendation system with Machine Learning-based smartphone price and category prediction.
