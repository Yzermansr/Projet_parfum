import express from "express";
import { newuser } from "../controllers/UtilisateurController.js";
import { rechercheuser } from "../controllers/UtilisateurController.js";
import { rechercheuserbis } from "../controllers/UtilisateurController.js";

export default (dbUtilisateur) => {
  if (!dbUtilisateur) {
    console.error("Erreur : Base de données utilisateur non chargée !");
  }

  const router = express.Router();
  router.post("/newuser", (req, res) => newuser(req, res,dbUtilisateur));
  router.get("/rechercheruser", (req, res) => rechercheuser(req, res, dbUtilisateur));
  router.post("/rechercheruserbis", (req, res) => rechercheuserbis(req, res, dbUtilisateur));
  return router;
};

