import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import os
import pandas as pd

# Wymuszenie odbudowy pliku SHX
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Wczytanie danych z przesłanego pliku
file_path = r'C:/Users/macie/Desktop/ISA/miniprojekt_2/dane/Border_Crossing_Entry_Data.csv'
data = pd.read_csv(file_path)

# Podgląd danych
print(data.info())
print(data.head())

# Konwersja kolumny 'Date' na format datetime
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Sprawdzenie i przetwarzanie lokalizacji
if 'Location' in data.columns:
    data['geometry'] = data['Location'].apply(lambda loc: Point([float(coord) for coord in loc.strip('POINT ()').split()]))
    border_crossings = gpd.GeoDataFrame(data, geometry='geometry')
else:
    raise ValueError("Kolumna 'Location' nie istnieje w danych.")

# Wczytanie mapy krajów
countries = gpd.read_file(r'C:/Users/macie/Desktop/ISA/miniprojekt_2/vectors/ne_10m_admin_0_countries.shp')

# Wyświetlenie dostępnych kolumn, aby znaleźć odpowiednią
print(countries.columns)

# Debugowanie nazwy kolumny
if 'NAME' in countries.columns:
    usa = countries[countries['NAME'] == 'United States of America']
elif 'ADMIN' in countries.columns:
    usa = countries[countries['ADMIN'] == 'United States of America']
else:
    raise KeyError("Nie znaleziono odpowiedniej kolumny dla nazwy kraju (np. 'NAME' lub 'ADMIN'). Dostępne kolumny to: {}".format(countries.columns))

# Wykres przejść granicznych
fig, ax = plt.subplots(figsize=(12, 8))
usa.plot(ax=ax, color='lightgrey', edgecolor='black')
border_crossings.plot(ax=ax, color='red', markersize=10, label='Przejścia graniczne')

plt.title('Przejścia graniczne w USA')
plt.legend()
plt.show()

# Dodanie informacji o miesiącu i roku
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Agregacja sezonowa
seasonal_trends = data.groupby(['Month', 'Border'])['Value'].mean().reset_index()

# Wykres trendów sezonowych
plt.figure(figsize=(14, 8))
for border in seasonal_trends['Border'].unique():
    border_data = seasonal_trends[seasonal_trends['Border'] == border]
    plt.plot(border_data['Month'], border_data['Value'], label=border, marker='o')

plt.legend()
plt.title('Trendy sezonowe: Średni miesięczny ruch dla US-Canada i US-Mexico')
plt.xlabel('Miesiąc')
plt.ylabel('Średni ruch')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid()
plt.tight_layout()
plt.show()

# Agregacja danych w czasie dla obu granic
ruch_czas_granice = data.groupby(['Date', 'Border'])['Value'].sum().reset_index()

# Wykres trendów czasowych dla obu granic
plt.figure(figsize=(14, 8))
for border in ruch_czas_granice['Border'].unique():
    border_data = ruch_czas_granice[ruch_czas_granice['Border'] == border]
    plt.plot(border_data['Date'], border_data['Value'], label=border)

plt.legend()
plt.title('Trendy ruchu w czasie: US-Canada vs US-Mexico')
plt.xlabel('Data')
plt.ylabel('Łączny ruch')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Agregacja danych: ruch dla każdego przejścia
ruch_przejscia = data.groupby('Port Name')['Value'].sum().reset_index()
ruch_przejscia = ruch_przejscia.sort_values(by='Value', ascending=False).head(10)

# Filtracja danych dla Top 5 przejść
top_ports = ruch_przejscia['Port Name'].head(5).tolist()
data_top_ports = data[data['Port Name'].isin(top_ports)]

# Agregacja danych w czasie (miesiącami) dla Top 5 przejść
data_top_ports['Month'] = data_top_ports['Date'].dt.to_period('M')
ruch_czas_top = data_top_ports.groupby(['Month', 'Port Name'])['Value'].sum().reset_index()

# Wykres trendów czasowych dla Top 5 przejść
plt.figure(figsize=(14, 8))
for port in top_ports:
    port_data = ruch_czas_top[ruch_czas_top['Port Name'] == port]
    plt.plot(port_data['Month'].astype(str), port_data['Value'], label=port)

plt.legend()
plt.title('Trendy ruchu w czasie dla Top 5 przejść granicznych')
plt.xlabel('Miesiąc')
plt.ylabel('Łączny ruch')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Agregacja danych: ruch dla każdego typu miary
ruch_miary = data.groupby('Measure')['Value'].sum().reset_index()
ruch_miary = ruch_miary.sort_values(by='Value', ascending=False)

# Wykres: Top 10 przejść
plt.figure(figsize=(12, 6))
plt.bar(ruch_przejscia['Port Name'], ruch_przejscia['Value'])
plt.xticks(rotation=45)
plt.title('Top 10 najbardziej obciążonych przejść granicznych')
plt.ylabel('Łączny ruch')
plt.show()

# Wykres: Miary ruchu
plt.figure(figsize=(12, 6))
plt.bar(ruch_miary['Measure'], ruch_miary['Value'])
plt.xticks(rotation=45)
plt.title('Łączny ruch dla każdego typu miary')
plt.ylabel('Łączny ruch')
plt.show()
