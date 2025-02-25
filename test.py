import csv

# Fichiers d'entrée et de sortie
input_csv = "parfums.csv"
output_csv = "parfums_numerotes.csv"
mapping_file = "mapping_ingredients.csv"

ingredient_to_id = {}
id_counter = 1

def get_ingredient_id(ingredient):
    global id_counter
    if ingredient not in ingredient_to_id:
        ingredient_to_id[ingredient] = id_counter
        id_counter += 1
    return ingredient_to_id[ingredient]

# Fonction pour convertir la rubrique Genre
def convert_genre(genre):
    genre = genre.lower()
    if genre == "woman":
        return "0"
    elif genre == "man":
        return "1"
    else:
        return "2"

# Lire le CSV, remplacer les ingrédients par des IDs et convertir le Genre
data = []
with open(input_csv, "r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames

    for row in reader:
        # Conversion du genre
        if "Genre" in row:
            row["Genre"] = convert_genre(row["Genre"])

        # Conversion des ingrédients
        for note in ["Tete", "Coeur", "Fond"]:
            ingredients = row[note].split(", ")
            ids = [str(get_ingredient_id(ing.strip())) for ing in ingredients]
            row[note] = ", ".join(ids)
        data.append(row)

# Sauvegarder le CSV modifié
with open(output_csv, "w", encoding="utf-8", newline="") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

# Sauvegarder le mapping ingrédient <-> ID
with open(mapping_file, "w", encoding="utf-8", newline="") as mapfile:
    writer = csv.writer(mapfile)
    writer.writerow(["ID", "Ingredient"])
    for ingredient, id_ in sorted(ingredient_to_id.items(), key=lambda x: x[1]):
        writer.writerow([id_, ingredient])

print("Traitement terminé :")
print(f"- Fichier transformé : {output_csv}")
print(f"- Mapping ingrédients : {mapping_file}")