"""
Skrypt 1: Analiza struktury danych
Uruchom ten skrypt jako pierwszy, aby poznać strukturę swojego datasetu
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ===============================
# KONFIGURACJA - ZMIEŃ TĘ ŚCIEŻKĘ
# ===============================
FILE_PATH = r"D:\Dev-Env\sources\repos\DataScience\datascience\datascience\miniprojekt_4\dane\train_delays.csv"

def analyze_dataset(file_path):
    """Kompleksowa analiza struktury datasetu"""
    
    print("="*60)
    print("ANALIZA DATASETU OPÓŹNIEŃ KOLEJOWYCH")
    print("="*60)
    print(f"Ścieżka: {file_path}")
    print(f"Rozmiar pliku: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")
    

    print("\n1. Wczytuję próbkę danych...")
    df_sample = pd.read_csv(file_path, nrows=10000)
    
  
    print("\n2. STRUKTURA DANYCH:")
    print(f"   Liczba kolumn: {len(df_sample.columns)}")
    print(f"   Liczba wierszy w próbce: {len(df_sample)}")
    
    print("\n3. NAZWY KOLUMN:")
    for i, col in enumerate(df_sample.columns):
        print(f"   [{i}] {col}")
    
    print("\n4. TYPY DANYCH:")
    print(df_sample.dtypes)
    
    print("\n5. PIERWSZE 5 WIERSZY:")
    print(df_sample.head())
    
    print("\n6. BRAKUJĄCE WARTOŚCI:")
    missing = df_sample.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("   Brak brakujących wartości w próbce")
    
    print("\n7. STATYSTYKI OPISOWE:")
    print(df_sample.describe())
    
  
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df_sample.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n8. KOLUMNY NUMERYCZNE ({len(numeric_cols)}):")
    print(f"   {numeric_cols}")
    
    print(f"\n9. KOLUMNY TEKSTOWE ({len(text_cols)}):")
    print(f"   {text_cols}")
    
    
    print("\n10. POTENCJALNE KOLUMNY Z OPÓŹNIENIAMI:")
    delay_cols = [col for col in df_sample.columns if 'delay' in col.lower() or 'opóźn' in col.lower()]
    if delay_cols:
        for col in delay_cols:
            print(f"   - {col}")
            if col in numeric_cols:
                print(f"     Min: {df_sample[col].min()}, Max: {df_sample[col].max()}, Średnia: {df_sample[col].mean():.2f}")
    else:
        print("   Nie znaleziono kolumn z 'delay' w nazwie")
    

    print("\n11. POTENCJALNE KOLUMNY CZASOWE:")
    time_cols = [col for col in df_sample.columns if any(word in col.lower() for word in ['date', 'time', 'data', 'czas'])]
    if time_cols:
        for col in time_cols:
            print(f"   - {col}")
            print(f"     Przykład: {df_sample[col].iloc[0]}")
    
   
    report_path = "data_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAPORT ANALIZY DANYCH\n")
        f.write("="*50 + "\n")
        f.write(f"Data: {datetime.now()}\n")
        f.write(f"Plik: {file_path}\n")
        f.write(f"Rozmiar: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB\n\n")
        f.write("KOLUMNY:\n")
        for i, col in enumerate(df_sample.columns):
            f.write(f"{i}: {col} ({df_sample[col].dtype})\n")
    
    print(f"\n✓ Raport zapisany jako: {report_path}")
    
    return df_sample

def create_visualizations(df_sample):
    """Tworzenie podstawowych wizualizacji"""
    
    print("\n12. TWORZENIE WIZUALIZACJI...")
    
 
    delay_col = None
    for col in df_sample.columns:
        if 'delay' in col.lower() and df_sample[col].dtype in ['int64', 'float64']:
            delay_col = col
            break
    
    if delay_col:
        plt.figure(figsize=(15, 10))
        
       
        plt.subplot(2, 2, 1)
        plt.hist(df_sample[delay_col].dropna(), bins=50, edgecolor='black')
        plt.title(f'Rozkład {delay_col}')
        plt.xlabel('Opóźnienie')
        plt.ylabel('Częstość')
        
        
        plt.subplot(2, 2, 2)
        plt.boxplot(df_sample[delay_col].dropna())
        plt.title(f'Box plot {delay_col}')
        plt.ylabel('Opóźnienie')
        
      
        plt.subplot(2, 2, 3)
        top_delays = df_sample[delay_col].value_counts().head(10)
        plt.bar(range(len(top_delays)), top_delays.values)
        plt.title('Top 10 najczęstszych wartości opóźnień')
        plt.xlabel('Wartość')
        plt.ylabel('Częstość')
        
      
        plt.subplot(2, 2, 4)
        stats_text = f"""
        Średnia: {df_sample[delay_col].mean():.2f}
        Mediana: {df_sample[delay_col].median():.2f}
        Std: {df_sample[delay_col].std():.2f}
        Min: {df_sample[delay_col].min():.2f}
        Max: {df_sample[delay_col].max():.2f}
        Liczba unikalnych: {df_sample[delay_col].nunique()}
        """
        plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='center')
        plt.axis('off')
        plt.title('Statystyki opóźnień')
        
        plt.tight_layout()
        plt.savefig('data_analysis_plots.png', dpi=300, bbox_inches='tight')
        print("   ✓ Wykresy zapisane jako: data_analysis_plots.png")
        plt.close()

if __name__ == "__main__":
    try:
       
        if not os.path.exists(FILE_PATH):
            print(f"❌ BŁĄD: Nie znaleziono pliku: {FILE_PATH}")
            print("   Zmień zmienną FILE_PATH na właściwą ścieżkę do pliku!")
        else:
          
            df_sample = analyze_dataset(FILE_PATH)
            
        
            create_visualizations(df_sample)
            
            print("\n✅ ANALIZA ZAKOŃCZONA!")
            print("   Sprawdź pliki:")
            print("   - data_analysis_report.txt")
            print("   - data_analysis_plots.png")
            
    except Exception as e:
        print(f"\n❌ BŁĄD: {e}")
        print("   Sprawdź czy ścieżka do pliku jest poprawna!")