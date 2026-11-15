import { memo } from "react";
import "../styles/reseau-aneo-plan.css";
// Onglet d'administration « Réseau ANEO » : planification nationale et installation des capteurs par zone, d'après
// la maquette atom_wind_reseau_aneo.html. Point de montage seulement : le contenu est rendu par
// engine/reseau-aneo-plan.js (ATOM.reseauAneo.afficher(), appelé à l'ouverture de la vue).
export default memo(function ViewReseauAneo(){ return (
<section className="view" id="v-reseau"><div id="aneo-reseau"></div></section>
); });
