import sqlite3 from "sqlite3";
import { open } from "sqlite";

export const initDB = async () => {
  const dbParfums = await open({ filename: "./parfums.db", driver: sqlite3.Database });
  const dbNotes = await open({ filename: "./notes.db", driver: sqlite3.Database });
  const dbUtilisateur = await open({ filename: "./Utilisateur.db", driver: sqlite3.Database });
  await dbNotes.exec(`
    CREATE TABLE IF NOT EXISTS notes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      parfum TEXT NOT NULL,
      note INTEGER NOT NULL,
      pseudo TEXT NOT NULL
    )
  `);

  await dbUtilisateur.exec(`
    CREATE TABLE IF NOT EXISTS Utilisateur (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      Pseudo TEXT UNIQUE NOT NULL,
      Mot_de_passe TEXT NOT NULL
    )
  `);

  return { dbParfums, dbNotes, dbUtilisateur };
};
