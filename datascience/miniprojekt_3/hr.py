import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


data_path = r"C:\\Users\\macie\\Desktop\\ISA\\miniprojekt-2\\dane\\HR_Analytics.csv"

data = pd.read_csv(data_path)


print("Podgląd danych:")
print(data.head())
print("\nInformacje o danych:")
print(data.info())
print("\nStatystyki opisowe:")
print(data.describe())


missing_values = data.isnull().sum()
print("\nBrakujące wartości w kolumnach:")
print(missing_values)


sns.countplot(x='Attrition', data=data)
plt.title('Rozkład odejść z firmy')
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Macierz korelacji')
plt.show()


attrition_yes = data[data['Attrition'] == 'Yes']
print("\nStatystyki pracowników, którzy odeszli:")
print(attrition_yes.describe())


plt.figure(figsize=(8, 6))
sns.scatterplot(x='YearsInCurrentRole', y='YearsSinceLastPromotion', hue='Attrition', data=data)
plt.title('Lata w obecnej roli vs. lata od ostatniego awansu')
plt.show()


categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = pd.factorize(data[col])[0]


X = data.drop('Attrition', axis=1)
y = data['Attrition']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))

print("\nMacierz pomyłek:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"\nROC AUC: {roc_auc:.2f}")


feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Ważność cech w modelu Random Forest')
plt.show()

print("\nRekomendacje dla działu HR:")
print("1. Skup się na pracownikach z niską satysfakcją z pracy i długim czasem bez awansu.")
print("2. Rozważ wdrożenie programów rozwoju zawodowego dla kluczowych grup.")
print("3. Poprawa równowagi między życiem zawodowym a prywatnym może zmniejszyć rotację.")
