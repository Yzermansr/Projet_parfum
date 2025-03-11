import { useState } from "react";
import { useNavigate } from "react-router-dom";  


function Identification() {

    const navigate = useNavigate();
  const [utilisateur, setUtilisateur] = useState({ pseudo: "", password: "", confirmPassword: "" });
  const [message, setMessage] = useState("");
  

  const handleChange = (e) => {
    setUtilisateur({ ...utilisateur, [e.target.name]: e.target.value });
  };

  const handleCreateAccount = async () => {
    if (!utilisateur.pseudo || !utilisateur.password || !utilisateur.confirmPassword) {
      setMessage("Tous les champs sont requis !");
      return;
    }

    if (utilisateur.password !== utilisateur.confirmPassword) {
      setMessage("Les mots de passe ne correspondent pas !");
      return;
    }

    const response = await fetch(`http://localhost:5000/api/rechercheruser?Pseudo=${encodeURIComponent(utilisateur.pseudo)}`);
    const data = await response.json();
    if (data.count > 0) {
      setMessage("Pseudo déjà utilisé");
      return;
    }
    try {
      const response = await fetch("http://localhost:5000/api/newuser", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ Pseudo: utilisateur.pseudo, Mot_de_passe: utilisateur.password }),
      });

      const data = await response.json();
      if (response.ok) {
        setMessage("Compte créé avec succès !");
      } else {
        setMessage(data.message || "Erreur lors de la création du compte");
      }
    } catch (error) {
      console.error("Erreur lors de l'inscription :", error);
      setMessage("Erreur serveur, réessayez plus tard.");
    }
  };

  const handlecreation = () => {
    navigate("/");  
  };

  return (
    <div style={{ maxWidth: "400px", margin: "auto", padding: "20px", border: "1px solid #ccc", borderRadius: "10px" }}>
      <h1>Créer un compte </h1>
      <input type="text" name="pseudo" value={utilisateur.pseudo} onChange={handleChange} placeholder="Pseudo" style={{ display: "block", width: "100%", marginBottom: "10px" }} />
      <input type="password" name="password" value={utilisateur.password} onChange={handleChange} placeholder="Mot de passe" style={{ display: "block", width: "100%", marginBottom: "10px" }} />
      <input type="password" name="confirmPassword" value={utilisateur.confirmPassword} onChange={handleChange} placeholder="Répétez le mot de passe" style={{ display: "block", width: "100%", marginBottom: "10px" }} />
      <button onClick={handleCreateAccount} style={{ width: "100%", padding: "10px", background: "#007bff", color: "white", borderRadius: "5px", cursor: "pointer" }}>
       Creation compte
      </button>
      <button onClick={handlecreation} style={{ width: "100%", padding: "10px", background: "#007bff", color: "white", borderRadius: "5px", cursor: "pointer" }}>
        Retour page connexion
      </button>
      {message && <p style={{ marginTop: "10px", color: "red" }}>{message}</p>}
    </div>
  );
}

export default Identification;