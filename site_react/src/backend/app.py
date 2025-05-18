from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import random
from auto import generate_W
from site_react.src.backend.min_regret import min_max_regret

app = Flask(__name__)
CORS(app)

DB = "parfums_numerotes.db"
DB2 = "database"

@app.route("/api/parfum/search")
@app.route("/api/parfum/search")
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
        W, b, P = generate_W()
        X_min, Y_max, w_opt, min_max_value = min_max_regret(P, W)

        print("Minimax Regret Solution:")
        print(f"Alternative with minimal maximum regret: {X_min}")
        print(f"Worst-case alternative: {Y_max}")
        print(f"Weight vector producing maximum regret: {w_opt}")
        print(f"Minimum maximum regret value: {min_max_value}")

        return jsonify({
            "parfumA": {
                "nom": parfum1[0],
                "description": f"Voir plus : {parfum1[1]}"
            },
            "parfumB": {
                "nom": parfum2[0],
                "description": f"Voir plus : {parfum2[1]}"
            },
            "regret": float(min_max_value)
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