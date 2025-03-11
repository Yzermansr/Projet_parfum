import { addNote, getAverageRating, getTop5, getTop5bis } from "../models/notesModel.js";

export const rateParfum = async (req, res, db) => {
  const { name, note, pseudo} = req.body;
  if (!name || !note || !pseudo) return res.status(400).json({ error: "Nom du parfum et note et pseudo requis" });

  try {
    await addNote(db, name, note,pseudo);
    res.json({ message: "Note enregistrée avec succès !" });
  } catch (err) {
    res.status(500).json({ error: err.message });
    console.log("erreur :", pseudo, note, name);
  }
};

export const getParfumRating = async (req, res, db) => {
  const { name } = req.query;
  if (!name) return res.status(400).json({ error: "Nom du parfum requis" });

  try {
    const moyenne = await getAverageRating(db, name);
    res.json({ moyenne });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

export const getTopParfums = async (req, res, db) => {
  try {
    const top5 = await getTop5(db);
    res.json(top5);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

export const getTopParfumsbis = async (req, res, db) => {
  try {
    const {pseudo} = req.query;
    const top5 = await getTop5bis(db,pseudo);
    res.json(top5);
    console.log("Nombre d'utilisateurs trouvés :", pseudo);
  } catch (err) {
    res.status(500).json({ error: err.message });

  }
};