import { memo } from "react";
import "../styles/reseau-aneo.css";
// Onglet d'administration « Réseau ANEO » (phase 8). Point de montage seulement : le contenu est rendu par
// engine/reseau-aneo.js (habillage provisoire, à reprendre de la maquette reseau_aneo.html quand elle sera fournie).
export default memo(function ViewReseauAneo(){ return (
<section className="view" id="v-reseau"><div id="aneo-reseau"></div></section>
); });
