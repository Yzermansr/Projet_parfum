export const searchParfum = async (db, name) => {
  return await db.get("SELECT * FROM parfums WHERE LOWER(Nom) = LOWER(?)", [name.toLowerCase()]);
};

export const autocompleteParfum = async (db, query) => {
  return await db.all("SELECT Nom FROM parfums WHERE LOWER(Nom) LIKE LOWER(?) LIMIT 5", [`${query}%`]);
};
