import { newuser as addNewUser } from "../models/UtilisateurModel.js"; 
import { rechercheuser as Finduser } from "../models/UtilisateurModel.js"; 
import { rechercheuserbis as Finduserbis } from "../models/UtilisateurModel.js"; 

export const newuser = async (req, res, dbUtilisateur) => {
  if (!dbUtilisateur) {
    console.error("Erreur : La connexion à la base de données est introuvable !");
    return res.status(500).json({ error: "Problème avec la base de données" });
  }

  console.log("Requête reçue:", req.body);
  const { Pseudo, Mot_de_passe } = req.body;

  if (!Pseudo || !Mot_de_passe) {
    return res.status(400).json({ error: "Pseudo et Mot de passe requis" });
  }

  try {
    await addNewUser(dbUtilisateur, Pseudo, Mot_de_passe);
    console.log("Utilisateur ajouté :", Pseudo);
    res.json({ message: "Utilisateur ajouté avec succès !" });
  } catch (err) {
    console.error("Erreur serveur :", err);
    res.status(500).json({ error: err.message });
  }
};

export const rechercheuser = async (req, res, dbUtilisateur) => {
  if (!dbUtilisateur) {
    console.error(" Erreur : La connexion à la base de données est introuvable !");
    return res.status(500).json({ error: "Problème avec la base de données" });
  }


  const { Pseudo } = req.query;

  if (!Pseudo) {
    console.error(" Erreur : Pseudo manquant dans la requête !");
    return res.status(400).json({ error: "Pseudo requis" });
  }

  try {
    const count = await Finduser(dbUtilisateur, Pseudo);
    console.log("Nombre d'utilisateurs trouvés :", count);
    res.json({ count });
  } catch (err) {
    console.error("Erreur serveur :", err);
    res.status(500).json({ error: err.message });
  }
};

export const rechercheuserbis = async (req, res, dbUtilisateur) => {
  if (!dbUtilisateur) {
    console.error(" Erreur : La connexion à la base de données est introuvable !");
    return res.status(500).json({ error: "Problème avec la base de données" });
  }
  const { Pseudo, Mot_de_passe } = req.body;

  try {
    const count = await Finduserbis(dbUtilisateur, Pseudo,Mot_de_passe);
    console.log("Nombre d'utilisateurs trouvés :", count);
    res.json({ count });
  } catch (err) {
    console.error("Erreur serveur :", err);
    res.status(500).json({ error: err.message });
  }
};




  

  



  