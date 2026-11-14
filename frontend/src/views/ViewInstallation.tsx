import { memo } from "react";
import "../styles/installation-aneo.css";
// Onglet d'administration « Installation ANEO » : l'administrateur clique une zone sur la carte, l'application
// propose où installer les capteurs ANEO dans cette zone. Contenu rendu par engine/installation-aneo.js.
export default memo(function ViewInstallation(){ return (
<section className="view" id="v-installation"><div id="aneo-installation"></div></section>
); });
