import express from "express";
import cors from "cors";
import { initDB } from "./config/database.js";
import parfumsRoutes from "./routes/parfumsRoutes.js";
import notesRoutes from "./routes/notesRoutes.js";
import UtilisateurRoutes from "./routes/UtilisateurRoutes.js";

const app = express();
app.use(express.json());
app.use(cors());

(async () => {
  const { dbParfums, dbNotes, dbUtilisateur } = await initDB();
  app.use("/api", parfumsRoutes(dbParfums));
  app.use("/api", notesRoutes(dbNotes));
  app.use("/api", UtilisateurRoutes(dbUtilisateur));
  app.listen(5000, () => console.log("Serveur démarré sur le port 5000"));
})();