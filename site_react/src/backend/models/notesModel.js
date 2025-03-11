export const addNote = async (db, name, note, pseudo) => {
  await db.run("INSERT INTO notes (parfum, note, pseudo) VALUES (?, ?, ?)", [name, note, pseudo]);
};

export const getAverageRating = async (db, name) => {
  const result = await db.get("SELECT AVG(note) AS moyenne FROM notes WHERE parfum = ?", [name]);
  return result?.moyenne ? result.moyenne.toFixed(1) : "Pas encore noté";
};

export const getTop5 = async (db) => {
  return await db.all(`SELECT parfum, AVG(note) AS moyenne FROM notes GROUP BY parfum ORDER BY moyenne DESC LIMIT 5`);
};

export const getTop5bis = async (db,pseudo) => {
  return await db.all(`SELECT parfum, AVG(note) AS moyenne FROM notes WHERE pseudo = ? GROUP BY parfum ORDER BY moyenne DESC LIMIT 5`,[pseudo]);
};
