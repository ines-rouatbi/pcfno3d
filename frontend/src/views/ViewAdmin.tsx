import { memo, type CSSProperties } from "react";
// Administration : tout vient du serveur (comptes, clés d'API, abonnés, supervision) ; les cartes sont
// ajoutées par atom-data.js si la session a la portée administrateur.
export default memo(function ViewAdmin(){ return (
<section className="view" id="v-admin">
<div className="card"><p className="muted" style={{"margin": "0"} as CSSProperties}>{"Validez les comptes, gérez les clés d'accès à l'API, suivez les abonnés aux alertes et l'état du service. Réservé aux administrateurs."}</p></div>
<div className="card"><h3>{"Rôles et accès"}</h3><div className="tblwrap"><table className="t" id="adroles"></table></div></div>
</section>
); });
