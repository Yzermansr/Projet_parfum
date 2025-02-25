import json
import pandas as pd


with open('ingredients.json', 'r') as ingredient_fichier:
    data = json.load(ingredient_fichier)

liste = []

for ingredient, url  in data.items():
    liste.append({
        'Ingredient': ingredient,
        'URL': url
        
    })

data = pd.DataFrame(liste)
data.to_csv('ingredients.csv')


