import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";


const App = () => {
  const [step, setStep] = useState(0);
  const [gender, setGender] = useState(null);
  const [parfum, setParfum] = useState(null);
  const [parfum2, setParfum2] = useState(null);
  const navigate = useNavigate();


  const fetchParfum = async (genre) => {
    const res = await fetch(`http://localhost:5001/api/parfum/search?genre=${genre}`);
    const data = await res.json();
    setParfum(data);
    const res2 = await fetch(`http://localhost:5001/api/parfum/search?genre=${genre}`);
    const data2 = await res2.json();
    setParfum2(data2);
    console.log("Parfum 1:", data);
    console.log("Parfum 2:", data2);

  };

  const handleGender = (g) => {
    setGender(g);
    setStep(1);
    const genre = g === "homme" ? 0 : 1;
    fetchParfum(genre);
  };

  const handleVoteA = () => {
    fetch('http://localhost:5001/api/response', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({parfum: parfum.nom,parfum2 : parfum2.nom }),
    });
    setStep(2);
  };

  const handleVoteB = () => {
    fetch('http://localhost:5001/api/response', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({parfum: parfum2.nom,parfum2 : parfum.nom }),
    });
    setStep(2);
  };

  const resetQuiz = () => {
    setStep(0);
    setGender(null);
    setParfum(null);
  };

  if (step === 0) {
    return (
      <div style={styles.container}>
        <button onClick={() => navigate("/Home")}>Home page</button>
        <h1 style={styles.title}>Es-tu un homme ou une femme ?</h1>
        <div style={styles.btnRow}>
          <button style={styles.btn} onClick={() => handleGender("homme")}>Homme</button>
          <button style={styles.btn} onClick={() => handleGender("femme")}>Femme</button>
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div style={styles.container}>
        <h2 style={styles.title}>Merci pour ta réponse !</h2>
        <button style={styles.btn} onClick={resetQuiz}>Recommencer</button>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h1 style={styles.subtitle}>Ce parfum te plaît-il ?</h1>
      {parfum && parfum2 &&(
        <>
          <div style={styles.btnRow}>
            <button style={styles.btn} onClick={() => handleVoteA()}>{parfum.nom}</button>
            <button style={styles.btn} onClick={() => handleVoteB()}>{parfum2.nom}</button>
          </div>
          <p style={styles.desc}>{parfum.description}</p>
          <p style={styles.desc}>{parfum2.description}</p>
        </>
      )}
    </div>
  );
};

const styles = {
  container: {
    padding: 30,
    textAlign: 'center',
    background: '#111',
    color: '#fff',
    minHeight: '100vh',
  },
  title: {
    fontSize: 26,
    marginBottom: 20,
  },
  subtitle: {
    fontSize: 20,
    marginBottom: 10,
  },
  name: {
    fontSize: 24,
    marginBottom: 8,
    fontWeight: 'bold',
  },
  desc: {
    fontSize: 14,
    marginBottom: 20,
  },
  btnRow: {
    display: 'flex',
    justifyContent: 'center',
    gap: '10px',
    flexWrap: 'wrap',
  },
  btn: {
    background: '#222',
    color: '#fff',
    border: '1px solid #444',
    padding: '50px 100px',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '16px',
  }
};

export default App;