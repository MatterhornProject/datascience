import os
import glob

# 1. Pokazuje gdzie jesteś
print("="*60)
print("AKTUALNY FOLDER:")
print(os.getcwd())
print("="*60)

# 2. Pokazuje co jest w aktualnym folderze
print("\nPLIKI W TYM FOLDERZE:")
files = os.listdir()
csv_files = [f for f in files if f.endswith('.csv')]
if csv_files:
    for f in csv_files:
        print(f"  - {f}")
else:
    print("  Brak plików CSV")

# 3. Szuka pliku w standardowych lokalizacjach
print("\n" + "="*60)
print("SZUKAM PLIKU train_delays.csv...")
print("="*60)

# Ścieżki do sprawdzenia (dla Windows)
paths_to_check = [
    r"C:\Users\macie\Downloads",
    r"C:\Users\macie\Desktop", 
    r"C:\Users\macie\Documents",
    r"D:\Dev-Env\sources\repos\DataScience\datascience\datascience\miniprojekt_4\dane", 
    os.path.expanduser("~\\Downloads"),
    os.path.expanduser("~\\Desktop"),
]

found = False
for path in paths_to_check:
    if os.path.exists(path):
        print(f"\nSprawdzam: {path}")
        try:
            files = os.listdir(path)
            for file in files:
                if 'train' in file.lower() and 'delay' in file.lower():
                    full_path = os.path.join(path, file)
                    size_mb = os.path.getsize(full_path) / 1024 / 1024
                    print(f"  ✓ ZNALAZŁEM: {file}")
                    print(f"    Pełna ścieżka: {full_path}")
                    print(f"    Rozmiar: {size_mb:.2f} MB")
                    found = True
        except Exception as e:
            print(f"  Nie mogę przeszukać tego folderu: {e}")

if not found:
    print("\n❌ NIE ZNALAZŁEM PLIKU!")
    print("\nSPRAWDŹ:")
    print("1. Gdzie pobrałeś plik z Kaggle?")
    print("2. Czy plik nazywa się dokładnie 'train_delays.csv'?")
    print("3. Może jest spakowany (.zip)?")
    print("\nMożesz też ręcznie poszukać pliku i podać mi pełną ścieżkę.")