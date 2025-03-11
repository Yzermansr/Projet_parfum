import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import ParfumDetail from "./pages/ParfumDetail";
import Utilisateur from "./pages/Utilisateur";
import Identification from "./pages/Identifiaction";
import Test from "./pages/Test";


function App() {
  return (
    <Router>
      <Routes>
        <Route path="/Test" element={<Test />} />
        <Route path="/Home" element={<Home />} />
        <Route path="/parfum/:name" element={<ParfumDetail />} />
        <Route path="/" element={<Utilisateur />} />
        <Route path="/Identification" element={<Identification />} />
      </Routes>
    </Router>
  );
}

export default App;