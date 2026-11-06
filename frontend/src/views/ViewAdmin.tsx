import { memo, type CSSProperties } from "react";
export default memo(function ViewAdmin(){ return (
<section className="view" id="v-admin">
<div className="card"><p className="muted" style={{"margin": "0"} as CSSProperties}>{"Validez les comptes, attribuez les rôles et gérez les clés d'accès à l'API. Réservé aux administrateurs."}</p></div>
<div className="kpis" id="adkpis"></div>
<div className="card"><h3>{"Utilisateurs"}</h3><div className="tblwrap"><table className="t" id="adusers"></table></div></div>
<div className="grid2">
<div className="card"><h3>{"Rôles et accès"}</h3><div className="tblwrap"><table className="t" id="adroles"></table></div></div>
<div className="card"><h3>{"Clés d'API"}</h3><p className="muted" style={{"marginTop": "0", "fontSize": "14px"} as CSSProperties}>{"Une clé permet à un logiciel partenaire de lire le vent en temps réel. Elle n'est affichée qu'une seule fois."}</p>
<div className="form"><label htmlFor="k-name">{"Nom de la clé"}<input className="inp" id="k-name" placeholder="Ex. : Parc éolien de Bizerte" /></label><label htmlFor="k-scope">{"Accès"}<select className="inp" id="k-scope"><option value="lecture">{"Lecture du vent"}</option><option value="simulation">{"Lecture et simulations (R&D)"}</option><option value="admin">{"Administration"}</option></select></label><button className="btn primary" id="k-add">{"Créer la clé"}</button></div>
<div id="k-new"></div><div className="mine" id="k-list"></div></div>
</div>
</section>
); });
