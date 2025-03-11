import { searchParfum, autocompleteParfum } from "../models/parfumsModel.js";

export const autocomplete = async (req, res, db) => {
  const { query } = req.query;
  if (!query) return res.json([]);

  try {
    const results = await autocompleteParfum(db, query);
    res.json(results.map(row => row.Nom));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

export const search = async (req, res, db) => {
  const { name } = req.query;
  if (!name) return res.status(400).json({ error: "Nom du parfum requis" });

  try {
    const parfum = await searchParfum(db, name);
    res.json(parfum || { message: "Aucun parfum trouvé" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
