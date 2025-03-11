import { useEffect,useState } from "react";
import { useNavigate } from "react-router-dom";

function Test() {


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


    return (
    <div>
        <h1>Test</h1>
        <button onClick={() => navigate("/home")}>Voir le parfum 1</button>
    </div>
);






}
