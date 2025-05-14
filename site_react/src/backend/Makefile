# Makefile pour le site parfum

# Lancer le backend Python (Flask ou autre)
run-backend:
	python3 app.py

# Lancer le frontend React
run-frontend:
	npm run dev

# Installer les dépendances Node (npm)
install-node:
	npm install

# Lancer le serveur Node.js (Express)
run-node:
	node server.js

# Tout installer
install-all: install-node

# Tout lancer (frontend + backend)
run-all:
	make -j2 run-backend run-frontend
