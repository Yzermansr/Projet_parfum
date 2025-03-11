import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

function Home() {
  const [search, setSearch] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [parfum, setParfum] = useState(null);
  const [top5, setTop5] = useState([]);
  const [top5bis, setTop5bis] = useState([]);
  const [pseudo, setPseudo] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    const storedPseudo = localStorage.getItem("pseudo");
    if (storedPseudo) {
      setPseudo(storedPseudo);
    }
  }, []);

  useEffect(() => {
    if (search.length > 0) {
      fetch(`http://localhost:5000/api/autocomplete?query=${search}`)
        .then(res => res.json())
        .then(data => setSuggestions(data))
        .catch(err => console.error("Erreur API :", err));
    } else {
      setSuggestions([]);
    }
  }, [search]);

  useEffect(() => {
    fetch(`http://localhost:5000/api/top5`)
      .then(res => res.json())
      .then(data => setTop5(data))
      .catch(err => console.error("Erreur API Top 5:", err));
  }, []);

  const handleSearch = (name) => {
    navigate(`/parfum/${name}`);
  };

  useEffect(() => {
    const Pseudo = localStorage.getItem("pseudo");
    fetch(`http://localhost:5000/api/top5bis?pseudo=${Pseudo}`)
      .then(res => res.json())
      .then(data => setTop5bis(data))
      .catch(err => console.error("Erreur API Top 5:", err));
  }, []);

  return (
    
    <div style={{ maxWidth: "1000px", margin: "auto", textAlign: "center" }}>
      <div style={{ alignSelf: "flex-start", padding: "10px" }}>
          {pseudo && <p>Connecté en tant que : <strong>{pseudo}</strong></p>}
          <button onClick={() => navigate("/")}>Se Deconnecter</button>
        </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: "20px" }}>
        <div style={{ padding: "15px", border: "2px solid #ccc", borderRadius: "10px", background: "#f9f9f9", width: "30%", marginRight: "50px" }}>
          <h3>🏆 Top 5 des meilleurs parfums</h3>
          <ul style={{ listStyle: "none", padding: "0" }}>
            {top5.length > 0 ? (
              top5.map((p, index) => (
                <li key={index} style={{ padding: "5px", borderBottom: "1px solid #ddd" }}>
                  <strong>{p.parfum}</strong> - ⭐ {p.moyenne.toFixed(1)}
                </li>
              ))
            ) : (
              <p>Aucun parfum noté pour le moment.</p>
            )}
          </ul>
        </div>

        {/* Section Recherche */}
        <div style={{ textAlign: "center", padding: "20px", border: "2px solid black", borderRadius: "10px", width: "30%", marginRight: "50px" }}>
          <h1 style={{ fontWeight: "bold" }}>Site de parfum</h1>
          <p>Quel parfum recherchez-vous ?</p>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Tapez un nom de parfum..."
            style={{ width: "100%", padding: "5px", marginBottom: "10px" }}
          />

          {!parfum && suggestions.length > 0 && (
            <ul style={{ listStyle: "none", padding: "5px", border: "1px solid #ccc", width: "100%", background: "white", margin: "10px auto 0", textAlign: "left" }}>
              {suggestions.map((s, index) => (
                <li
                  key={index}
                  style={{ cursor: "pointer", padding: "8px", borderBottom: "1px solid #eee" }}
                  onClick={() => handleSearch(s)}
                >
                  {s}
                </li>
              ))}
            </ul>
          )}
        </div>

        <div style={{ padding: "15px", border: "2px solid #ccc", borderRadius: "10px", background: "#f9f9f9", width: "30%" }}>
          <h3>🏆 Votre top 5 des meilleurs parfums</h3>
          <ul style={{ listStyle: "none", padding: "0" }}>
            {top5bis.length > 0 ? (
              top5bis.map((p, index) => (
                <li key={index} style={{ padding: "5px", borderBottom: "1px solid #ddd" }}>
                  <strong>{p.parfum}</strong> - ⭐ {p.moyenne.toFixed(1)}
                </li>
              ))
            ) : (
              <p>Aucun parfum noté pour le moment.</p>
            )}
          </ul>
        </div>
      </div>
        
      
    </div>
  );
}

export default Home;
