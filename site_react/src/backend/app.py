from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import random
import numpy as np
from auto import generate_W ,generate_P
from min_regret import min_max_regret
from best_question import get_best_question
from mon_min_max import min_regret

app = Flask(__name__)
CORS(app)

DB = "parfums_numerotes.db"
DB2 = "database"

@app.route("/api/2parfum/search/with/newton") 
def get_random_duel_with_regret():
    try:
        # Connexion à la base
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT Nom, URL FROM parfums_numerotes")
        rows = c.fetchall()
        conn.close()

        if len(rows) < 2:
            return jsonify({"error": "Pas assez de parfums dans la base"}), 500

        # Sélection de 2 parfums aléatoires différents
        parfum1, parfum2 = random.sample(rows, 2)

        # Calcul du regret actuel
        W, b = generate_W()
        P = generate_P()
        result = 69
        print(parfum1)
        print(parfum2)
        return jsonify({
            "parfumA": {
                "nom": parfum1[0],
                "description": f"Voir plus : {parfum1[1]}"
            },
            "parfumB": {
                "nom": parfum2[0],
                "description": f"Voir plus : {parfum2[1]}"
            },
            "regret": float(result)
        })

    except Exception as e:
        print("Erreur dans /api/parfum/search :", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/parfum/search")
def get_regret():
    try:
        print(">>> get_regret appelé")
        W, b = generate_W()
        print(">>> generate_W terminé")
        P = generate_P()
        print(">>> generate_P terminé")

        print(">>> W =", type(W))
        print(">>> b =", type(b))
        print(">>> P =", type(P))

        x, y, _ = get_best_question("newton", W, b, P)
        print(">>> get_best_question terminé")

        _, _, result = min_regret(P, W)
        print(">>> min_regret terminé")

        result = np.linalg.norm(result)
        nom1 = x.nom
        nom2 = y.nom
        print(">>> noms récupérés :", nom1, nom2)

        return jsonify({
            "parfumA": {
                "nom": nom1,
                "description": f"Voir plus : {nom1}"
            },
            "parfumB": {
                "nom": nom2,
                "description": f"Voir plus : {nom2}"
            },
            "regret": float(result)
        })

    except Exception as e:
        print("Erreur dans /api/parfum/search :", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



@app.route("/api/response", methods=["POST"])
def save_response():
    conn = sqlite3.connect(DB)
    conn2 = sqlite3.connect(DB2)
    c = conn.cursor()
    c2 = conn2.cursor()

    data = request.json
    parfum1 = data.get("parfum")
    parfum2 = data.get("parfum2")

    print("Réponse enregistrée:", data)

    c.execute("SELECT id FROM parfums_numerotes WHERE Nom = ?", (parfum1,))
    parfum1 = c.fetchone()
    c.execute("SELECT id FROM parfums_numerotes WHERE Nom = ?", (parfum2,))
    parfum2 = c.fetchone()

    if parfum1 and parfum2:
        c2.execute("INSERT INTO comparaison (parfum1, parfum2) VALUES (?, ?)", (parfum1[0], parfum2[0]))
        conn2.commit()
        print(f"Parfum 1 ID: {parfum1[0]} et Parfum 2 ID: {parfum2[0]}")

    conn.close()
    conn2.close()
    return jsonify({"status": "ok"})

@app.route("/api/reset", methods=["DELETE"])
def reset_comparaison():
    try:
        conn = sqlite3.connect(DB2)
        c = conn.cursor()
        c.execute("DELETE FROM comparaison")
        conn.commit()
        conn.close()
        print("Table comparaison réinitialisée.")
        return jsonify({"status": "réinitialisé"}), 200
    except Exception as e:
        print("Erreur lors de la réinitialisation :", e)
        return jsonify({"error": "Erreur lors de la réinitialisation"}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)