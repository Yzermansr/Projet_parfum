import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const App = () => {
  const [round, setRound] = useState(1);
  const [parfum, setParfum] = useState(null);
  const [parfum2, setParfum2] = useState(null);
  const [regret, setRegret] = useState(null);
  const navigate = useNavigate();

  // Fonction pour récupérer deux parfums aléatoires + regret
  const fetchParfum = async () => {
    try {
      const res = await fetch("http://localhost:5001/api/parfum/search");
      const data = await res.json();

      setParfum(data.parfumA);
      setParfum2(data.parfumB);
      setRegret(data.regret);
    } catch (err) {
      console.error("Erreur lors du chargement des parfums :", err);
    }
  };

  const fetchParfum2 = async () => {
    try {
      const res = await fetch("http://localhost:5001/api/parfum/search");
      const data = await res.json();

      setParfum(data.parfumA);
      setParfum2(data.parfumB);
      setRegret(data.regret);
    } catch (err) {
      console.error("Erreur lors du chargement des parfums :", err);
    }
  };

  // Chargement initial
  useEffect(() => {
    fetchParfum();
  }, []);

  // Lors d’un vote (A ou B)
  const handleVote = (choixA) => {
    const payload = choixA
      ? { parfum: parfum.nom, parfum2: parfum2.nom }
      : { parfum: parfum2.nom, parfum2: parfum.nom };

    fetch("http://localhost:5001/api/response", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if(round < 5) {
      setRound(round + 1);
      fetchParfum2();
    } 
   else if (5 < round < 25) {
      setRound(round + 1);
      fetchParfum();
    } else {
      setRound(26); // Fin
    }
  };

  // Réinitialisation : reset + nouvelle question
  const resetQuiz = async () => {
    try {
      await fetch("http://localhost:5001/api/reset", {
        method: "DELETE",
      });
      setRound(1);
      setParfum(null);
      setParfum2(null);
      setRegret(null);
      fetchParfum();
    } catch (err) {
      console.error("Erreur de réinitialisation :", err);
    }
  };

  // ---------------- Fin du quiz ----------------
  if (round > 25) {
    return (
      <div style={styles.container}>
        <h2 style={styles.title}>Merci pour ta participation !</h2>
        <button style={styles.btn} onClick={resetQuiz}>
          Recommencer
        </button>
      </div>
    );
  }

  // ---------------- Quiz en cours ----------------
  return (
    <div style={styles.container}>
      <button onClick={() => navigate("/Home")} style={styles.linkBtn}>
        ← Accueil
      </button>
      <h1 style={styles.subtitle}>Duel {round} / 25 : Quel parfum préfères-tu ?</h1>

      {parfum && parfum2 ? (
        <>
          <div style={styles.btnRow}>
            <button style={styles.btn} onClick={() => handleVote(true)}>
              {parfum.nom}
            </button>
            <button style={styles.btn} onClick={() => handleVote(false)}>
              {parfum2.nom}
            </button>
          </div>
          <div style={styles.descBox}>
            <p><strong>{parfum.nom}</strong> : {parfum.description}</p>
            <p><strong>{parfum2.nom}</strong> : {parfum2.description}</p>
          </div>
          {regret && (
            <p style={{ color: "#888", fontStyle: "italic", marginTop: 10 }}>
              Regret actuel : {regret.toFixed(4)}
            </p>
          )}
          <div style={{ marginTop: 30 }}>
            <button style={styles.smallBtn} onClick={resetQuiz}>
              Recommencer
            </button>
          </div>
        </>
      ) : (
        <p>Chargement des parfums...</p>
      )}
    </div>
  );
};

// ---------------- Styles ----------------
const styles = {
  container: {
    padding: 30,
    textAlign: "center",
    background: "#111",
    color: "#fff",
    minHeight: "100vh",
  },
  title: {
    fontSize: 26,
    marginBottom: 20,
  },
  subtitle: {
    fontSize: 22,
    marginBottom: 15,
  },
  btnRow: {
    display: "flex",
    justifyContent: "center",
    gap: "20px",
    flexWrap: "wrap",
    marginBottom: 20,
  },
  btn: {
    background: "#222",
    color: "#fff",
    border: "1px solid #444",
    padding: "40px 60px",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "16px",
    transition: "0.3s",
  },
  smallBtn: {
    background: "#444",
    color: "#fff",
    border: "1px solid #666",
    padding: "12px 20px",
    borderRadius: "6px",
    cursor: "pointer",
    fontSize: "14px",
  },
  linkBtn: {
    background: "none",
    border: "none",
    color: "#0af",
    fontSize: "16px",
    marginBottom: 20,
    cursor: "pointer",
    textDecoration: "underline",
  },
  descBox: {
    marginTop: 20,
    fontSize: 14,
    color: "#ccc",
    maxWidth: 600,
    marginLeft: "auto",
    marginRight: "auto",
    textAlign: "left",
  },
};

export default App;