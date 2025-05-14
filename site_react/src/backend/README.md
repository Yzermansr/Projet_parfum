# Site Parfum – Utilisation du Makefile

Ce projet contient un site web pour la sélection de parfums, avec un frontend (React), un backend Python, et un serveur Node.js (Express).

## 🛠️ Prérequis

- Python 3
- Node.js et npm
- `make` installé (disponible par défaut sur macOS/Linux, sinon utiliser WSL sur Windows)

## 📂 Structure typique attendue

```
projet/
├── app.py             # Backend Python
├── server.js          # Serveur Node.js
├── package.json       # Fichier npm
├── Makefile
└── README.md
```

## ⚙️ Commandes Make disponibles

### Installation

```bash
make install-all
```

Installe les dépendances Node.js (via `npm install`).

### Lancement du site

```bash
make run-all
```

Lance le backend Python (`app.py`) et le frontend React (`npm run dev`) en parallèle.

### Commandes spécifiques

```bash
make run-backend     # Lance uniquement le backend Python
make run-frontend    # Lance uniquement le frontend React
make run-node        # Lance uniquement le serveur Node.js (server.js)
```

## 💡 Astuces

- Tu peux modifier `app.py` et `server.js` selon ta structure.
- Si tu veux lancer chaque commande dans un terminal séparé automatiquement, utilise un gestionnaire de terminal comme `tmux`, ou ajoute les commandes dans un script `bash`.

---

Made with ❤️
