export const newuser = async (db, Pseudo, Mot_de_passe) => {
  await db.run("INSERT INTO Utilisateur (Pseudo, Mot_de_passe) VALUES (?, ?)", [Pseudo, Mot_de_passe]);
};
export const rechercheuser = async (db, Pseudo) => {
  const result = await db.get("SELECT COUNT(*) as count FROM Utilisateur WHERE Pseudo = ?", [Pseudo]);
  return result ? result.count : 0;
};
export const rechercheuserbis = async (db, Pseudo, Mot_de_passe) => {
  const result = await db.get("SELECT COUNT(*) as count FROM Utilisateur WHERE Pseudo = ? AND Mot_de_passe = ?", [Pseudo,Mot_de_passe]);
  return result ? result.count : 0;
};



