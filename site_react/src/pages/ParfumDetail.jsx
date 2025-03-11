import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";  

function ParfumDetail() {
  const { name } = useParams();
  const navigate = useNavigate(); 
  const [parfum, setParfum] = useState(null);
  const [rating, setRating] = useState("Pas encore noté");
  const [selectedNote, setSelectedNote] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pseudo, setPseudo] = useState("");


  useEffect(() => {
    const storedPseudo = localStorage.getItem("pseudo");
    if (storedPseudo) {
      setPseudo(storedPseudo);
    }
  }, []);


  useEffect(() => {
    async function fetchParfum() {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`http://localhost:5000/api/search?name=${name}`);
        if (!response.ok) {
          throw new Error("Erreur lors de la récupération du parfum");
        }
        const data = await response.json();
        setParfum(data);

        const ratingResponse = await fetch(`http://localhost:5000/api/ratings?name=${name}`);
        if (!ratingResponse.ok) {
          throw new Error("Erreur lors de la récupération de la note");
        }
        const ratingData = await ratingResponse.json();
        setRating(ratingData.moyenne);
      } catch (err) {
        console.error("Erreur :", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    fetchParfum();
  }, [name]);

  const handleRate = async () => {
    if (!parfum) return;

    await fetch("http://localhost:5000/api/rate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: parfum.Nom, note: selectedNote, pseudo: pseudo }),
    });

    
    const ratingResponse = await fetch(`http://localhost:5000/api/ratings?name=${parfum.Nom}`);
    const ratingData = await ratingResponse.json();
    setRating(ratingData.moyenne);
  };

  const handleBack = () => {
    navigate("/Home");  
  };

  if (loading) return <p>Chargement en cours...</p>;
  if (error) return <p style={{ color: "red" }}>Erreur : {error}</p>;
  if (!parfum) return <p>Parfum introuvable.</p>;

  return (
    <center>
      <div style={{ alignSelf: "flex-start", padding: "10px" }}>
          {pseudo && <p>Connecté en tant que : <strong>{pseudo}</strong></p>}
          <button onClick={() => navigate("/")}>Se Deconnecter</button>
        </div>
    <div style={{ marginTop: "20px", padding: "15px", border: "1px solid #ccc", borderRadius: "10px", maxWidth: "400px", background: "#f9f9f9" }}>
      <h2>{parfum.Nom}</h2>
      <a href={parfum.URL} target="_blank" rel="noopener noreferrer">Voir le parfum</a>
      <p><strong>Genre :</strong> {parfum.Genre}</p>
      <p><strong>Notes de tête :</strong> {parfum.Tete}</p>
      <p><strong>Notes de cœur :</strong> {parfum.Coeur}</p>
      <p><strong>Notes de fond :</strong> {parfum.Fond}</p>

      <p><strong>Note moyenne :</strong> {rating}</p>

      <p><strong>Noter ce parfum :</strong></p>
      <select value={selectedNote} onChange={(e) => setSelectedNote(Number(e.target.value))}>
        {[1, 2, 3, 4, 5].map((n) => (
          <option key={n} value={n}>{n} ⭐</option>
        ))}
      </select>
      <button onClick={handleRate} style={{ marginLeft: "10px", padding: "5px", cursor: "pointer" }}>
        Envoyer
      </button>

      <button onClick={handleBack} style={{ marginTop: "10px", padding: "8px 12px", backgroundColor: "#007bff", color: "white", borderRadius: "5px", cursor: "pointer" }}>
        Retour
      </button>
    </div>
    </center>
  );
}

export default ParfumDetail;
