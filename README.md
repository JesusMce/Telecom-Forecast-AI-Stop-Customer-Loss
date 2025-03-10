# **📡 AI-Driven Telecom Churn Prediction: Maximizing Retention & Revenue 🚀**  

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-blue)  
![Data Science](https://img.shields.io/badge/Data%20Science-Python-green)  
![Customer Retention](https://img.shields.io/badge/Customer%20Retention-Business%20Intelligence-orange)  

---

## **📌 Project Overview**  
This project leverages **cutting-edge machine learning** to predict **customer churn in the telecom industry**. By analyzing **contract, personal, phone, and internet service data**, the model identifies high-risk customers **before they leave**, allowing the company to implement **targeted retention strategies and maximize revenue**.  

🔥 **Key Features:**  
✅ **AI-powered churn prediction** for proactive customer retention  
✅ **99.43% accuracy & 0.9916 AUC-ROC** for high-performance insights  
✅ **Feature Engineering & Class Balancing** for reliable predictions  
✅ **Data-driven retention strategies** to reduce customer churn  
✅ **Scalable for enterprise telecom operators**  

---

## **📂 Project Structure**  
```
├── datasets/
│   ├── final_provider/
│   │   ├── contract.csv  # Customer contract details
│   │   ├── personal.csv  # Personal customer data
│   │   ├── internet.csv  # Internet service details
│   │   ├── phone.csv  # Phone service information
├── notebooks/
│   ├── eda_analysis.ipynb  # Exploratory Data Analysis
│   ├── train_model.ipynb  # Model Training & Evaluation
├── models/
│   ├── telecom_churn_model.pkl  # Trained ML Model
├── scripts/
│   ├── run_churn_prediction.py  # Automated Churn Prediction
├── README.md
```

---

## **📊 Dataset**  
📌 **Source:** Telecom customer data from multiple sources  
📌 **Size:** 7,000+ customer records  
📌 **Data Files:**  

| **File**          | **Description**  |
|------------------|--------------------------------|
| `contract.csv`   | Customer contract details (tenure, billing, payment method) |
| `personal.csv`   | Customer demographics (age, gender, senior citizen) |
| `internet.csv`   | Internet service details (DSL, fiber-optic, security) |
| `phone.csv`      | Telephone service details (multiple lines, international calls) |

---

## **⚙️ Installation & Execution**  
### 1️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

### 2️⃣ **Load & Clean Data**  
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

contract = pd.read_csv('datasets/final_provider/contract.csv')
personal = pd.read_csv('datasets/final_provider/personal.csv')
internet = pd.read_csv('datasets/final_provider/internet.csv')
phone = pd.read_csv('datasets/final_provider/phone.csv')

# Feature Engineering
contract['TotalCharges'] = pd.to_numeric(contract['TotalCharges'], errors='coerce')
contract['ContractDuration'] = contract['EndDate'].apply(lambda x: 1 if x == 'No' else 0)

# Standardization
scaler = StandardScaler()
contract[['MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(contract[['MonthlyCharges', 'TotalCharges']])
```

### 3️⃣ **Train the Model**  
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = contract.drop(columns=['customerID', 'EndDate'])
y = contract['ContractDuration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 4️⃣ **Evaluate Model Performance**  
```python
from sklearn.metrics import accuracy_score, roc_auc_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
```

---

## **📈 Model Performance**  
📌 **Validation Accuracy:** `99.43%`  
📌 **AUC-ROC Score:** `0.9916`  
📌 **Selected Features:**  

| Feature | Description |
|---------|------------|
| `MonthlyCharges` | Monthly billing amount |
| `ChargesRatio` | TotalCharges / MonthlyCharges |
| `Type_Two year` | Contract duration type |
| `PaymentMethod_Credit card` | Payment method used |
| `Partner_Yes` | Whether customer has a partner |
| `InternetService_Fiber optic` | Fiber-optic internet usage |
| `OnlineSecurity_Yes` | Whether customer has online security |

**🛠 Insights:**  
✔ **Customers with higher MonthlyCharges are more likely to churn.**  
✔ **Fiber-optic users show increased churn risk.**  
✔ **Payment method affects retention – automatic payments have lower churn.**  
✔ **Customers with partners have higher retention rates.**  

---

## **🚀 Real-World Applications**  
💡 **Customer Retention Teams** – Identify high-risk customers & offer personalized retention incentives.  
💡 **Marketing & Promotions** – Target customers with special offers based on churn probability.  
💡 **Revenue Optimization** – Predict revenue loss & improve pricing strategies.  
💡 **AI-Driven Business Intelligence** – Leverage machine learning to enhance telecom decision-making.  

---

## **🤝 Contributing**  
Want to enhance this project? Follow these steps:  
1. **Fork** the repository  
2. Create a new branch: `git checkout -b feature-improvement`  
3. Make your changes and **commit**: `git commit -m "Enhanced feature engineering"`  
4. Push the changes: `git push origin feature-improvement`  
5. Open a **Pull Request** 🎉  

---

## **📜 License**  
This project is licensed under the **MIT License** – Free to use, modify, and contribute. 🎯  

---

## **💼 Work With Me**  
📧 **Email:** econejes@gmail.com
🌍 **LinkedIn:** https://www.linkedin.com/in/edaga/
🚀 **Portfolio:** https://github.com/JesusMce

🔥 **If you find this project valuable, give it a ⭐ on GitHub!** 🚀  

---

### **💰 Monetization Strategies:**  
- **Sell it as a SaaS API** – Provide predictive analytics for telecom operators  
- **Integrate with CRM tools** – Enhance customer engagement with AI insights  
- **License it to telecom firms** – Help companies **reduce churn & increase customer lifetime value**  
- **Use it for targeted ads** – Predict customer behaviors for precision marketing  

---
