import { useState } from "react";
import { useNavigate } from "react-router-dom";  

function Identification() {
  const navigate = useNavigate(); 
  const [utilisateur, setUtilisateur] = useState({ pseudo: "", password: ""});
  const [message, setMessage] = useState("");
  
  const handleChange = (e) => {
    setUtilisateur({ ...utilisateur, [e.target.name]: e.target.value });
  };

  const handleConnectAccount = async () => {
    if (!utilisateur.pseudo || !utilisateur.password) {
      setMessage("Tous les champs sont requis !");
      return;
    }
  
    console.log("📤 Envoi des identifiants JSON :", JSON.stringify({
      Pseudo: utilisateur.pseudo,
      Mot_de_passe: utilisateur.password,
    })); // ✅ Vérifie si les données envoyées sont correctes
  
    try {
      const response = await fetch("http://localhost:5000/api/rechercheruserbis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          Pseudo: utilisateur.pseudo,
          Mot_de_passe: utilisateur.password,
        }),
      });
  
      console.log("🔍 Réponse brute de l'API :", response);
  
      if (!response.ok) {
        throw new Error("Erreur API");
      }
  
      const data = await response.json();
      console.log("🔍 Réponse de l'API :", data);
  
      if (data.count === 0) {
        setMessage("Pseudo ou mot de passe incorrect");
        return;
      }
  
      // Stocker le pseudo dans le stockage local
      localStorage.setItem("pseudo", utilisateur.pseudo);
  
      // Rediriger vers la page d'accueil
      navigate("/Home");
  
    } catch (error) {
      console.error("Erreur lors de la connexion :", error);
      setMessage("Erreur serveur, réessayez plus tard.");
    }
  };
  
  const handlecreation = () => {
    navigate("/Identification");  
  };
  
  return (
    <div style={{ maxWidth: "400px", margin: "auto", padding: "20px", border: "1px solid #ccc", borderRadius: "10px" }}>
      <h1>Connexion</h1>
      <input type="text" name="pseudo" value={utilisateur.pseudo} onChange={handleChange} placeholder="Pseudo" style={{ display: "block", width: "100%", marginBottom: "10px" }} />
      <input type="password" name="password" value={utilisateur.password} onChange={handleChange} placeholder="Mot de passe" style={{ display: "block", width: "100%", marginBottom: "10px" }} />
      <button onClick={handleConnectAccount} style={{ width: "100%", padding: "10px", background: "#007bff", color: "white", borderRadius: "5px", cursor: "pointer" }}>
        Connexion
      </button>
      <button onClick={handlecreation} style={{ width: "100%", padding: "10px", background: "#007bff", color: "white", borderRadius: "5px", cursor: "pointer" }}>
        Creation Compte
      </button>
      {message && <p style={{ marginTop: "10px", color: "red" }}>{message}</p>}
    </div>
  );
}

export default Identification;