import express from "express";
import { rateParfum, getParfumRating,getTopParfums , getTopParfumsbis } from "../controllers/notesController.js";

export default (db) => {
  const router = express.Router();
  router.post("/rate", (req, res) => rateParfum(req, res, db));
  router.get("/ratings", (req, res) => getParfumRating(req, res, db));
  router.get("/top5", (req, res) => getTopParfums(req, res, db));
  router.get("/top5bis", (req, res) => getTopParfumsbis(req, res, db));
  return router;
};