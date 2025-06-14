"""
Skrypt 3: Trenowanie modeli
Trenowanie różnych modeli ML z optymalizacją dla dużych danych
"""

import pandas as pd
import numpy as np
import joblib
import time
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import SGDRegressor, Ridge
import warnings
warnings.filterwarnings('ignore')

# ===============================
# KONFIGURACJA
# ===============================
PROCESSED_DATA_PATH = "processed_data.pkl"
TARGET_COLUMN = "delay_minutes"  
TEST_SIZE = 0.2
RANDOM_STATE = 42

def find_target_column(df):
    """Automatyczne wykrywanie kolumny docelowej"""
    
    print("Szukam kolumny z opóźnieniami...")
    
  
    delay_cols = [col for col in df.columns if 'delay' in col.lower()]
    
   
    numeric_delay_cols = []
    for col in delay_cols:
        if df[col].dtype in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
          
            if not any(prefix in col for prefix in ['avg_', 'std_', 'count_', 'p75_', 'rolling_']):
                numeric_delay_cols.append(col)
    
    if numeric_delay_cols:
        variances = {col: df[col].var() for col in numeric_delay_cols}
        target = max(variances, key=variances.get)
        print(f"  ✓ Znaleziono kolumnę docelową: {target}")
        return target
    else:
        print("  ⚠ Nie znaleziono kolumny z opóźnieniami!")
        print("    Dostępne kolumny numeryczne:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for i, col in enumerate(numeric_cols[:20]):
            print(f"    [{i}] {col}")
        return None

def prepare_features(df, target_column):
    """Przygotowanie cech do modelowania"""
    
    print("\nPrzygotowuję cechy...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    valid_cols = []
    for col in numeric_cols:
        missing_ratio = df[col].isnull().sum() / len(df)
        if missing_ratio < 0.3:
            valid_cols.append(col)
    
    print(f"  Liczba cech: {len(valid_cols)}")
    
    if len(valid_cols) == 0:
        print("  ⚠ Brak odpowiednich cech numerycznych!")
        return None, None
    
    X = df[valid_cols].fillna(0)
    y = df[target_column]
    
    return X, y, valid_cols

def train_fast_models(X_train, X_test, y_train, y_test):
    """Trenowanie szybkich modeli dla dużych danych"""
    
    results = {}
    
    models = {
        'SGD Regressor': SGDRegressor(
            max_iter=1000,
            tol=1e-3,
            eta0=0.01,
            penalty='l2',
            random_state=RANDOM_STATE
        ),
        
        'Ridge Regression': Ridge(
            alpha=1.0,
            solver='saga',
            random_state=RANDOM_STATE
        ),
        
        'HistGradient Boosting': HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=8,
            learning_rate=0.1,
            l2_regularization=0.1,
            random_state=RANDOM_STATE
        ),
        
        'Random Forest (Fast)': RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    }
    
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    except ImportError:
        print("  ℹ XGBoost nie jest zainstalowany")
    
    print("\nTrenowanie modeli:")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n{name}:")
        start_time = time.time()
        
        try:
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            start_time = time.time()
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            pred_time = time.time() - start_time
            
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            mse_test = mean_squared_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mse_test)
            r2_test = r2_score(y_test, y_pred_test)
            
            results[name] = {
                'model': model,
                'mae_train': mae_train,
                'mae_test': mae_test,
                'rmse_test': rmse_test,
                'r2_test': r2_test,
                'train_time': train_time,
                'pred_time': pred_time
            }
            
            print(f"  ✓ MAE (train): {mae_train:.2f}")
            print(f"  ✓ MAE (test): {mae_test:.2f}")
            print(f"  ✓ RMSE (test): {rmse_test:.2f}")
            print(f"  ✓ R² (test): {r2_test:.3f}")
            print(f"  ✓ Czas trenowania: {train_time:.1f}s")
            
        except Exception as e:
            print(f"  ✗ Błąd: {e}")
    
    return results

def save_results(results, X_test, y_test, feature_columns):
    """Zapisywanie wyników i tworzenie wizualizacji"""
    
    print("\nZapisywanie wyników...")
    
    best_model_name = min(results, key=lambda x: results[x]['mae_test'])
    best_model = results[best_model_name]['model']
    
    model_data = {
        'model': best_model,
        'model_name': best_model_name,
        'features': feature_columns,
        'metrics': results[best_model_name]
    }
    
    joblib.dump(model_data, 'best_model.pkl')
    print(f"  ✓ Najlepszy model ({best_model_name}) zapisany jako: best_model.pkl")
    
    with open('model_results.txt', 'w', encoding='utf-8') as f:
        f.write("WYNIKI TRENOWANIA MODELI\n")
        f.write("="*60 + "\n\n")
        
        for name, metrics in sorted(results.items(), key=lambda x: x[1]['mae_test']):
            f.write(f"{name}:\n")
            f.write(f"  MAE (test): {metrics['mae_test']:.2f}\n")
            f.write(f"  RMSE (test): {metrics['rmse_test']:.2f}\n")
            f.write(f"  R² (test): {metrics['r2_test']:.3f}\n")
            f.write(f"  Czas trenowania: {metrics['train_time']:.1f}s\n\n")
    
    print("  ✓ Wyniki zapisane jako: model_results.txt")
    
    try:
        import matplotlib.pyplot as plt
        
        if hasattr(best_model, 'feature_importances_'):
            plt.figure(figsize=(10, 8))
            importances = pd.DataFrame({
                'feature': feature_columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            plt.barh(range(len(importances)), importances['importance'])
            plt.yticks(range(len(importances)), importances['feature'])
            plt.xlabel('Ważność')
            plt.title(f'Top 20 najważniejszych cech - {best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Wykres ważności cech: feature_importance.png")
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        y_pred = best_model.predict(X_test)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Rzeczywiste opóźnienia')
        plt.ylabel('Przewidywane opóźnienia')
        plt.title('Rzeczywiste vs Przewidywane')
        
        plt.subplot(2, 2, 2)
        errors = y_pred - y_test
        plt.hist(errors, bins=50, edgecolor='black')
        plt.xlabel('Błąd predykcji')
        plt.ylabel('Częstość')
        plt.title('Rozkład błędów')
        
        plt.subplot(2, 2, 3)
        model_names = list(results.keys())
        mae_scores = [results[m]['mae_test'] for m in model_names]
        plt.bar(range(len(model_names)), mae_scores)
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        plt.ylabel('MAE')
        plt.title('Porównanie modeli')
        
        plt.tight_layout()
        plt.savefig('predictions_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Analiza predykcji: predictions_analysis.png")
        
    except Exception as e:
        print(f"  ⚠ Nie udało się utworzyć wizualizacji: {e}")

def main():
    """Główna funkcja"""
    
    print("="*60)
    print("TRENOWANIE MODELI PREDYKCJI OPÓŹNIEŃ")
    print("="*60)
    
    print("\n1. Wczytywanie danych...")
    try:
        data = joblib.load(PROCESSED_DATA_PATH)
        df = data['data']
        print(f"  ✓ Wczytano dane: {df.shape}")
    except FileNotFoundError:
        print(f"  ✗ Nie znaleziono pliku: {PROCESSED_DATA_PATH}")
        print("    Najpierw uruchom: python 2_feature_engineering.py")
        return
    

    if TARGET_COLUMN:
        target = TARGET_COLUMN
    else:
        target = find_target_column(df)
        if not target:
            print("\n⚠ Ustaw TARGET_COLUMN ręcznie w pliku!")
            return
    
    X, y, feature_columns = prepare_features(df, target)
    if X is None:
        return
    

    print("\n2. Podział na zbiory treningowy i testowy...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  ✓ Zbiór treningowy: {X_train.shape}")
    print(f"  ✓ Zbiór testowy: {X_test.shape}")
    
    print("\n3. Skalowanie danych...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = train_fast_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    save_results(results, X_test_scaled, y_test, feature_columns)
    
    joblib.dump(scaler, 'scaler.pkl')
    
    print("\n✅ TRENOWANIE ZAKOŃCZONE!")
    print("   Sprawdź pliki:")
    print("   - best_model.pkl")
    print("   - model_results.txt")
    print("   - feature_importance.png")
    print("   - predictions_analysis.png")

if __name__ == "__main__":
    main()