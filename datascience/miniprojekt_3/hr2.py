import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


education_groups = data.groupby('Education')['YearsAtCompany'].mean()
print("\nŚredni staż pracy w firmie w zależności od wykształcenia:")
print(education_groups)

sns.barplot(x=education_groups.index, y=education_groups.values)
plt.title('Średni staż pracy w zależności od wykształcenia')
plt.xlabel('Poziom wykształcenia')
plt.ylabel('Średni staż pracy')
plt.show()


no_promotion = data[data['YearsSinceLastPromotion'] == 0]
recent_hires = data[data['YearsAtCompany'] <= 1]

print("\nPracownicy bez awansu:")
print(no_promotion)
print("\nNowo zatrudnieni pracownicy:")
print(recent_hires)


promotions_by_level = data.groupby('JobLevel')['YearsSinceLastPromotion'].count()
print("\nLiczba awansów na różnych poziomach stanowisk:")
print(promotions_by_level)

sns.barplot(x=promotions_by_level.index, y=promotions_by_level.values)
plt.title('Liczba awansów na różnych poziomach stanowisk')
plt.xlabel('Poziom stanowiska')
plt.ylabel('Liczba awansów')
plt.show()


never_promoted = data[data['YearsSinceLastPromotion'] == data['YearsAtCompany']]
print("\nPracownicy bez żadnego awansu:")
print(never_promoted)


attrition_cost = attrition_yes['MonthlyIncome'].sum()
savings_from_attrition = attrition_yes['MonthlyIncome'].mean() * len(attrition_yes)

print(f"\nStrata wynikająca z odejść pracowników: {attrition_cost}")
print(f"Oszczędności wynikające z odejść: {savings_from_attrition}")


promoted = data[data['YearsSinceLastPromotion'] > 0]
not_promoted = data[data['YearsSinceLastPromotion'] == 0]

avg_job_satisfaction_promoted = promoted['JobSatisfaction'].mean()
avg_job_satisfaction_not_promoted = not_promoted['JobSatisfaction'].mean()

print("\nŚrednia satysfakcja z pracy:")
print(f"Pracownicy z awansami: {avg_job_satisfaction_promoted}")
print(f"Pracownicy bez awansów: {avg_job_satisfaction_not_promoted}")

satisfaction_data = pd.DataFrame({
    'Status': ['Z awansem', 'Bez awansu'],
    'Średnia satysfakcja': [avg_job_satisfaction_promoted, avg_job_satisfaction_not_promoted]
})

sns.barplot(x='Status', y='Średnia satysfakcja', data=satisfaction_data)
plt.title('Porównanie satysfakcji z pracy')
plt.ylabel('Średnia satysfakcja')
plt.show()
