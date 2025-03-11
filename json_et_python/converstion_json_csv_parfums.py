import json
import pandas as pd

with open('parfums.json', 'r') as parfums:
    data_json = json.load(parfums)

liste = []
for url, parfum in data_json.items():
    liste.append({
        'URL': url,
        'Nom': parfum.get('nom', ''),
        'Genre': parfum.get('genre', ''),
        'Tete': ', '.join(parfum.get('Tête', [])),
        'Coeur': ', '.join(parfum.get('Coeur', [])),
        'Fond': ', '.join(parfum.get('Fond', []))
    })

data = pd.DataFrame(liste)
data.to_csv('parfums.csv')

print(data.columns)