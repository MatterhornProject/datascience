# Predykcja opóźnień pociągów — dokumentacja projektu
https://www.kaggle.com/datasets/bartek358/train-delays
## Cel projektu

Celem projektu jest **stworzenie narzędzia do prognozowania opóźnień pociągów** na podstawie historycznych danych o przejazdach. Model predykcyjny wspiera **zarządcę infrastruktury kolejowej** w priorytetyzacji przepustowości szlaków oraz w podejmowaniu decyzji operacyjnych.

---

## Przebieg i etapy projektu

Projekt składa się z trzech głównych etapów, realizowanych przez niezależne skrypty Python:

1. **Eksploracyjna analiza danych**  
2. **Feature engineering i przygotowanie danych**  
3. **Trenowanie i ocena modeli ML**

---

### 1. Analiza danych (skrypt 1)

- **Cel:** Poznanie struktury danych, wykrycie braków, identyfikacja cech i zmiennych czasowych oraz targetu (`delay`).
- **Główne działania:**  
  - Wczytanie próbki danych i analiza typów kolumn
  - Zliczanie unikalnych wartości, wykrycie potencjalnych cech czasowych i opóźnień
  - Generacja prostych wizualizacji (histogramy, boxploty)
  - Automatyczne raportowanie do pliku tekstowego
- **Efekt:** Raport o strukturze i jakości danych.

---

### 2. Feature engineering (skrypt 2)

- **Cel:** Stworzenie nowych cech istotnych dla predykcji, optymalizacja pamięci oraz przygotowanie finalnych danych do modelowania.
- **Kluczowe kroki:**
  - Tworzenie kolumny `delay_minutes` (target — liczba minut opóźnienia)
  - Inżynieria cech czasowych: godzina, dzień tygodnia, pora dnia, weekend, kwartał, itp.
  - Tworzenie cech agregowanych (średnie i odchylenia opóźnień dla przewoźnika, stacji, połączenia)
  - Kodowanie zmiennych kategorycznych (label/frequency encoding)
  - Optymalizacja typów danych dla oszczędności RAM (category, uint8 itp.)
  - Automatyczne zapisywanie:
    - Pełnego przetworzonego zbioru (`processed_data.pkl`)
    - Zoptymalizowanego pliku CSV tylko z istotnymi cechami (`delays_cleaned.csv`)
    - Próbki do szybkiego podglądu (`processed_sample.csv`)
- **Dlaczego tak:**  
  - **Inżynieria cech** zwiększa moc predykcyjną modeli ML.
  - **Agregaty i kody** pozwalają modelom lepiej wykrywać zależności.
  - **Optymalizacja typów** jest kluczowa przy pracy na milionowych zbiorach danych.

---

### 3. Trenowanie modeli ML (skrypt 3)

- **Cel:** Zbudowanie i ocena skuteczności różnych modeli regresyjnych do prognozowania opóźnień.
- **Główne działania:**
  - Automatyczny wybór i przygotowanie cech numerycznych
  - Podział na zbiór treningowy i testowy (80/20)
  - Skalowanie danych (StandardScaler)
  - Trening modeli:
    - **SGDRegressor**
    - **Ridge Regression**
    - **HistGradientBoostingRegressor** (skuteczny i szybki dla dużych zbiorów)
    - **RandomForestRegressor**
    - **XGBoost** (jeśli dostępny)
  - Automatyczna ewaluacja: MAE, RMSE, R², czas treningu
  - Zapis najlepszych modeli i metryk do plików
  - Generacja wykresów ważności cech i analizy predykcji

- **Dlaczego tak:**  
  - **Porównanie różnych modeli** pozwala wybrać najefektywniejszy algorytm dla danego problemu i danych.
  - **Boosting (HistGB, XGBoost)** zwykle daje najlepsze wyniki na tablicowych danych (structured data).
  - **Automatyczna ewaluacja i zapisywanie wyników** ułatwia iteracyjne ulepszanie projektu.

---

## Wykorzystane technologie i biblioteki

- **Python** — główny język programowania (wszechstronny i szybki dla Data Science)
- **Pandas** — analiza, przetwarzanie i czyszczenie danych
- **NumPy** — operacje numeryczne i optymalizacja
- **Matplotlib/Seaborn** — wizualizacje danych i wyników
- **Scikit-learn** — zestaw modeli ML, podział danych, skalowanie, metryki
- **Joblib** — szybkie serializowanie dużych obiektów (modele, dataframe’y)
- **XGBoost** — szybki i skuteczny algorytm boostingowy (jeśli zainstalowany)

### **Dlaczego te technologie?**

- **Python i pandas** to standard w branży do analizy danych i prototypowania modeli.
- **Scikit-learn** pozwala łatwo budować, trenować i porównywać modele ML.
- **XGBoost/HistGradientBoosting** są zoptymalizowane pod wydajność i bardzo skuteczne na dużych zbiorach.
- **Joblib** umożliwia sprawne zapisywanie i wczytywanie nawet dużych modeli.

---

## Pliki wygenerowane przez pipeline

- `processed_data.pkl` — pełny przetworzony zbiór do dalszej analizy
- `delays_cleaned.csv` — zoptymalizowany, lekki plik CSV z wybranymi istotnymi cechami
- `processed_sample.csv` — próbka danych do podglądu
- `best_model.pkl` — najlepszy wytrenowany model
- `model_results.txt` — szczegółowe wyniki modeli (MAE, RMSE, R², czas)
- `feature_importance.png` — wykres ważności cech
- `predictions_analysis.png` — analiza predykcji (scatter plot, rozkład błędów, porównanie modeli)

---

## Sposób użycia i workflow

1. **Uruchom skrypt 1** (analiza danych) – przejrzyj raport i wykresy
2. **Uruchom skrypt 2** (feature engineering) – powstaną zoptymalizowane dane
3. **Uruchom skrypt 3** (trening modeli) – uzyskasz najlepszy model i podsumowanie wyników  
**Każdy etap możesz powtarzać dowolną ilość razy** — pliki są nadpisywane.

---

## Możliwe dalsze kierunki rozwoju

- Dodanie predykcji w czasie rzeczywistym (API, dashboard)
- Dalsza optymalizacja cech, tuning hyperparametrów modeli
- Analiza sezonowości i zdarzeń specjalnych (święta, awarie, pogoda)
- Wizualizacja predykcji i raportowanie dla zarządcy kolei

---

## Kontakt i wsparcie

Projekt zbudowany i udokumentowany przez Maciej Znaleźniak.  


