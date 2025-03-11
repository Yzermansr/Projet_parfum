import csv


parfums = "parfums.csv"
parfum_numerote = "parfums_numerotes.csv"
ingredient_id_nom = "ingredients_id_nom.csv"

ingredient_to_id = {}
id_counter = 1

def get_ingredient_id(ingredient):
    global id_counter
    if ingredient not in ingredient_to_id:
        ingredient_to_id[ingredient] = id_counter
        id_counter += 1
    return ingredient_to_id[ingredient]


def convert_genre(genre):
    genre = genre.lower()
    if genre == "woman":
        return "0"
    elif genre == "man":
        return "1"
    else:
        return "2"


data = []
with open(parfums, "r") as parfum:
    reader = csv.DictReader(parfum)
    fieldnames = reader.fieldnames

    for row in reader:
        if "Genre" in row:
            row["Genre"] = convert_genre(row["Genre"])

        
        for note in ["Tete", "Coeur", "Fond"]:
            ingredients = row[note].split(", ")
            ids = [str(get_ingredient_id(ing.strip())) for ing in ingredients]
            row[note] = ", ".join(ids)
        data.append(row)


with open(parfum_numerote, "w") as parfum_numerote:
    writer = csv.DictWriter(parfum_numerote, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)


with open(ingredient_id_nom, "w") as mapfile:
    writer = csv.writer(mapfile)
    writer.writerow(["ID", "Ingredient"])
    for ingredient, id_ in sorted(ingredient_to_id.items(), key=lambda x: x[1]):
        writer.writerow([id_, ingredient])

