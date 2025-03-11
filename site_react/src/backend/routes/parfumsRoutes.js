import express from "express";
import { autocomplete, search } from "../controllers/parfumsController.js";

export default (db) => {
  const router = express.Router();
  router.get("/autocomplete", (req, res) => autocomplete(req, res, db));
  router.get("/search", (req, res) => search(req, res, db));
  return router;
};