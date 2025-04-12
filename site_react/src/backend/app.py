from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import random

app = Flask(__name__)
CORS(app)
DB = "parfums_numerotes.db"
DB2 = "comparaison.db"

@app.route("/api/parfum/search")
def get_random_parfum():
    genre = request.args.get("genre", default=None, type=int)
    if genre is None:
        return jsonify({"error": "Genre manquant"}), 400

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT Nom, URL FROM parfums_numerotes WHERE Genre = (?) or Genre = '2'", (genre,))
    rows = c.fetchall()
    conn.close()

    if not rows:
        return jsonify({})

    parfum = random.choice(rows)
    return jsonify({
        "nom": parfum[0],
        "description": f"Voir plus : {parfum[1]}"
    })

@app.route("/api/response", methods=["POST"])
def save_response():
    conn = sqlite3.connect(DB)
    conn2 = sqlite3.connect(DB2)
    c2 = conn2.cursor()
    c = conn.cursor()
    data = request.json
    parfum1 = data.get("parfum")
    parfum2 = data.get("parfum2")
    print("Réponse enregistrée:", data)
    c.execute("SELECT id FROM parfums_numerotes WHERE Nom = ?", (parfum1,))
    parfum1 = c.fetchone()
    c.execute("SELECT id FROM parfums_numerotes WHERE Nom = ?", (parfum2,))
    parfum2 = c.fetchone()
    c2.execute("INSERT INTO comparaison (parfum1, parfum2) VALUES (?, ?)",(parfum1[0], parfum2[0]))
    print("Parfum 1 ID:{} et Prafum2 ID {}:".format(parfum1[0],parfum2[0]))
    conn2.commit()
    conn.close()
    conn2.close()
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(port=5001, debug=True)
