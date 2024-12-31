create TABLE new_def_area AS
SELECT 
    "Ano/Estados" AS year,  
    'Acre' AS state, 
    "AC" AS deforestation
FROM def_area
UNION ALL
SELECT 
    "Ano/Estados" AS year,
    'Amazonas' AS state,  
    "AM" AS deforestation
FROM def_area
UNION ALL
SELECT 
    "Ano/Estados" AS year,
    'Amapa' AS state, 
    "AP" AS deforestation
FROM def_area
UNION ALL
SELECT 
    "Ano/Estados" AS year,
    'Maranhao' AS state, 
    "MA" AS deforestation
FROM def_area
UNION ALL
SELECT 
    "Ano/Estados" AS year,
    'Mato Grosso' AS state, 
    "MT" AS deforestation
FROM def_area
UNION ALL
SELECT 
    "Ano/Estados" AS year,
    'Para' AS state, 
    "PA" AS deforestation
FROM def_area
UNION ALL
SELECT 
    "Ano/Estados" AS year,
    'Rondonia' AS state, 
    "RO" AS deforestation
FROM def_area
UNION ALL
SELECT 
    "Ano/Estados" AS year,
    'Roraima' AS state, 
    "RR" AS deforestation
FROM def_area
UNION ALL
SELECT 
    "Ano/Estados" AS year,
    'Tocantins' AS state, 
    "TO" AS deforestation
from def_area;

-- Analiza trendów deforestacji (2004–2019) na poziomie stanów:

SELECT year, state, SUM(deforestation) AS total_deforestation
FROM new_def_area
GROUP BY year, state
ORDER BY year, state;

--Suma deforestacji dla wszystkich stanów na przestrzeni lat:

SELECT year, SUM(deforestation) AS total_deforestation
FROM new_def_area
GROUP BY year
ORDER BY year;

--Korelacja między deforestacją a pożarami (2004–2019):

SELECT d.year, SUM(d.deforestation) AS total_deforestation, SUM(ibaf.firespots) AS total_fires 
FROM new_def_area d
JOIN inpe_brazilian_amazon_fires ibaf 
    ON d.year = ibaf.year 
    AND UPPER(d.state) = UPPER(ibaf.state)  -- Ignorowanie wielkości liter w nazwach stanów
GROUP BY d.year
ORDER BY d.year;

--Wykrywanie stanów o największej deforestacji:

SELECT state, SUM(deforestation) AS total_deforestation
FROM new_def_area
GROUP BY state
ORDER BY total_deforestation DESC;

--Wpływ zjawisk El Niño/La Niña na deforestację i pożary:

SELECT e.phenomenon, e.severity, d.year, SUM(d.deforestation) AS total_deforestation, SUM(f.firespots) AS total_fires
FROM new_def_area d
LEFT JOIN el_nino_la_nina e ON e."start year" <= d.year AND e."end year" >= d.year
JOIN inpe_brazilian_amazon_fires f ON d.year = f.year AND UPPER(d.state) = UPPER(f.state)
GROUP BY e.phenomenon, e.severity, d.year
ORDER BY d.year;











