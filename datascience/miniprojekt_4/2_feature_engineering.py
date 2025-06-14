import pandas as pd
import numpy as np
import gc
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

FILE_PATH = r"D:\Dev-Env\sources\repos\DataScience\datascience\datascience\miniprojekt_4\dane\train_delays.csv"
OUTPUT_PATH = "processed_data.pkl"

def convert_delay_to_minutes(delay_str):
    if pd.isna(delay_str) or delay_str == '0 min':
        return 0
    try:
        return int(delay_str.replace(' min', '').strip())
    except:
        return 0

def optimize_dtypes(df):
    print("Optymalizuję typy danych...")
  
    if 'delay_minutes' not in df.columns and 'delay' in df.columns:
        print("  Tworzę kolumnę 'delay_minutes'...")
        df['delay_minutes'] = df['delay'].apply(convert_delay_to_minutes)
        df['delay_minutes'] = df['delay_minutes'].astype('int16')
    start_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    for col in df.select_dtypes(include=['int']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min >= 0:
            if col_max <= 255:
                df[col] = df[col].astype(np.uint8)
            elif col_max <= 65535:
                df[col] = df[col].astype(np.uint16)
            elif col_max <= 4294967295:
                df[col] = df[col].astype(np.uint32)
        else:
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype(np.int8)
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype(np.int16)
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype(np.int32)
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.5:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Pamięć przed: {start_mem:.2f} MB")
    print(f"Pamięć po: {end_mem:.2f} MB")
    print(f"Redukcja: {(1 - end_mem/start_mem)*100:.1f}%")
    return df

def create_time_features(df):
    print("\nTworzę cechy czasowe...")
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour.astype('uint8')
        df['day'] = df['datetime'].dt.day.astype('uint8')
        df['month'] = df['datetime'].dt.month.astype('uint8')
        df['year'] = df['datetime'].dt.year.astype('uint16')
        df['dayofweek'] = df['datetime'].dt.dayofweek.astype('uint8')
        df['quarter'] = df['datetime'].dt.quarter.astype('uint8')
        df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype('uint8')
        df['is_month_start'] = df['datetime'].dt.is_month_start.astype('uint8')
        df['is_month_end'] = df['datetime'].dt.is_month_end.astype('uint8')
        df['is_morning_rush'] = df['hour'].isin([7, 8, 9]).astype('uint8')
        df['is_evening_rush'] = df['hour'].isin([16, 17, 18, 19]).astype('uint8')
        df['part_of_day'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        )
        print("  ✓ Utworzono cechy czasowe z datetime")
    if 'arrival' in df.columns:
        try:
            df['arrival_hour'] = df['arrival'].str.split(':').str[0].astype('uint8')
            df['arrival_minute'] = df['arrival'].str.split(':').str[1].astype('uint8')
            df['arrival_total_minutes'] = df['arrival_hour'] * 60 + df['arrival_minute']
            print("  ✓ Utworzono cechy z arrival time")
        except:
            print("  ⚠ Nie udało się przetworzyć arrival time")
    return df

def create_aggregated_features(df):
    print("\nTworzę cechy agregowane...")
    delay_col = 'delay_minutes'
    if delay_col not in df.columns:
        print("  ⚠ Nie znaleziono kolumny delay_minutes")
        return df
    if 'carrier' in df.columns:
        print("  Obliczam statystyki dla przewoźników...")
        df['avg_delay_by_carrier'] = df.groupby('carrier')[delay_col].transform('mean').astype('float32')
        df['std_delay_by_carrier'] = df.groupby('carrier')[delay_col].transform('std').fillna(0).astype('float32')
        df['count_by_carrier'] = df.groupby('carrier')[delay_col].transform('count').astype('int32')
    if 'name' in df.columns:
        print("  Obliczam statystyki dla stacji...")
        df['avg_delay_by_station'] = df.groupby('name')[delay_col].transform('mean').astype('float32')
        df['std_delay_by_station'] = df.groupby('name')[delay_col].transform('std').fillna(0).astype('float32')
        df['count_by_station'] = df.groupby('name')[delay_col].transform('count').astype('int32')
    if 'connection' in df.columns:
        print("  Obliczam statystyki dla połączeń...")
        connection_counts = df['connection'].value_counts()
        popular_connections = connection_counts[connection_counts > 100].index
        mask = df['connection'].isin(popular_connections)
        df.loc[mask, 'avg_delay_by_connection'] = df[mask].groupby('connection')[delay_col].transform('mean').astype('float32')
        df['avg_delay_by_connection'] = df['avg_delay_by_connection'].fillna(df[delay_col].mean()).astype('float32')
    if 'hour' in df.columns:
        print("  Obliczam statystyki dla godzin...")
        df['avg_delay_by_hour'] = df.groupby('hour')[delay_col].transform('mean').astype('float32')
        df['avg_delay_by_dayofweek'] = df.groupby('dayofweek')[delay_col].transform('mean').astype('float32')
    if 'carrier' in df.columns and 'hour' in df.columns:
        print("  Tworzę cechy interakcyjne...")
   
        df['carrier_hour'] = df['carrier'].astype(str) + '_' + df['hour'].astype(str)
        carrier_hour_delays = df.groupby('carrier_hour')[delay_col].mean().to_dict()
        df['avg_delay_carrier_hour'] = df['carrier_hour'].map(carrier_hour_delays).fillna(df[delay_col].mean()).astype('float32')
        df.drop('carrier_hour', axis=1, inplace=True)
    return df


def encode_categorical_features(df):
    print("\nKoduję zmienne kategoryczne...")
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_columns = [col for col in cat_columns if not any(word in col.lower() for word in ['date', 'time'])]
    label_encoders = {}
    for col in cat_columns:
        unique_values = df[col].nunique()
        if unique_values < 100:
            print(f"  Label encoding: {col} ({unique_values} kategorii)")
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        elif unique_values < 1000:
            print(f"  Frequency encoding: {col} ({unique_values} kategorii)")
            freq_encoding = df[col].value_counts().to_dict()
            df[f'{col}_freq'] = df[col].map(freq_encoding).astype('int32')
        else:
            print(f"  ⚠ Pomijam {col} (zbyt wiele kategorii: {unique_values})")
    return df, label_encoders

def process_full_dataset(file_path):
    print("="*60)
    print("PRZETWARZANIE PEŁNEGO DATASETU")
    print("="*60)
    print("\n1. Wczytuję dane...")
    sample = pd.read_csv(file_path, nrows=1000)
    dtypes = {}
    for col in sample.columns:
        if sample[col].dtype == 'object':
            if sample[col].nunique() / len(sample) < 0.5:
                dtypes[col] = 'category'
        elif sample[col].dtype == 'int64':
            if sample[col].min() >= 0 and sample[col].max() <= 65535:
                dtypes[col] = 'uint16'
            else:
                dtypes[col] = 'int32'
        elif sample[col].dtype == 'float64':
            dtypes[col] = 'float32'
    df = pd.read_csv(file_path, dtype=dtypes)
    print(f"  ✓ Wczytano {len(df):,} wierszy")

    if 'delay_minutes' not in df.columns and 'delay' in df.columns:
        print("Tworzę kolumnę delay_minutes (po raz drugi, na wszelki wypadek)...")
        df['delay_minutes'] = df['delay'].apply(convert_delay_to_minutes)
        df['delay_minutes'] = df['delay_minutes'].astype('int16')
    df = optimize_dtypes(df)
    gc.collect()
    df = create_time_features(df)
    gc.collect()
    df = create_aggregated_features(df)
    gc.collect()
    df, label_encoders = encode_categorical_features(df)
    gc.collect()

  
    istotne_kolumny = [
        'delay_minutes', 'carrier', 'connection', 'arrival_total_minutes', 'name', 'datetime',
        'hour', 'dayofweek', 'is_weekend',
       
        'avg_delay_by_carrier', 'avg_delay_by_station', 'avg_delay_by_hour', 'avg_delay_by_connection'
    ]
    istotne_kolumny = [col for col in istotne_kolumny if col in df.columns]
    df_clean = df[istotne_kolumny].copy()
    df_clean.to_csv('delays_cleaned.csv', index=False)
    print(f"\n  ✓ Lekki plik CSV z istotnymi kolumnami zapisany jako: delays_cleaned.csv")


    print("\n2. Zapisuję przetworzone dane (pickle)...")
    processed_data = {
        'data': df,
        'label_encoders': label_encoders,
        'columns': df.columns.tolist(),
        'shape': df.shape
    }
    joblib.dump(processed_data, OUTPUT_PATH)
    print(f"  ✓ Dane zapisane jako: {OUTPUT_PATH}")
    df.head(1000).to_csv('processed_sample.csv', index=False)
    print(f"  ✓ Próbka zapisana jako: processed_sample.csv")
    print("\n3. PODSUMOWANIE:")
    print(f"  - Liczba wierszy: {len(df):,}")
    print(f"  - Liczba kolumn: {len(df.columns)}")
    print(f"  - Pamięć: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print(f"  - Nowe cechy: {len(df.columns) - len(sample.columns)}")
    return df

if __name__ == "__main__":
    try:
        import os
        if not os.path.exists(FILE_PATH):
            print(f"❌ BŁĄD: Nie znaleziono pliku: {FILE_PATH}")
        else:
            df = process_full_dataset(FILE_PATH)
            print("\n✅ PRZETWARZANIE ZAKOŃCZONE!")
    except Exception as e:
        print(f"\n❌ BŁĄD: {e}")
        import traceback
        traceback.print_exc()
