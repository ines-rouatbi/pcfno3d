import { memo, type CSSProperties } from "react";
export default memo(function ViewFiabilite(){ return (
<section className="view" id="v-fiabilite">
<div className="grid2">
<div className="card">
<h2>{"Où le vent est-il bien connu ?"}</h2>
<div className="row" style={{"marginBottom": "10px"} as CSSProperties}><div className="seg" role="group" aria-label="Réseau de capteurs"><button id="m-reel" aria-pressed="true">{"Réseau actuel"}</button><button id="m-aneo" aria-pressed="false">{"Avec le réseau ANEO prévu"}</button></div></div>
<div className="mapbox" id="map2"><canvas id="c2"></canvas></div>
<div className="legend"><span className="pill good">{"Fiabilité bonne"}</span><span className="pill mid">{"Moyenne"}</span><span className="pill bad">{"Faible"}</span><span>{"Les zones pâles sont moins sûres."}</span></div>
</div>
<div className="steps">
<div className="card" id="relstats"></div>
<div className="card">
<h3>{"Vérifié sur de vraies mesures"}</h3>
<p className="muted" style={{"marginTop": "0"} as CSSProperties}>{"Le modèle servi, comparé aux anémomètres et aux ballons-sondes sur des jours jamais vus à l'entraînement (plus c'est petit, mieux c'est)."}</p>
<div id="aw-val-mesures"><p className="muted" style={{"margin": "0"} as CSSProperties}>{"Chargement de l'évaluation du modèle servi…"}</p></div>
</div>
</div>
</div>
<div className="card" id="aw-valeur" hidden={true}></div>
</section>
); });
