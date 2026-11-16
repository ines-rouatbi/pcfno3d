import { memo, type CSSProperties } from "react";
export default memo(function ViewEolien(){ return (
<section className="view" id="v-eolien">
<div className="card"><h2>{"Parc de référence du djebel Ressas"}</h2><p className="muted" style={{"margin": "0 0 10px"} as CSSProperties}>{"Trois éoliennes de 2 MW sur la crête du djebel Ressas, moyeux à 80 m : un parc de référence pour lire le vent en production. Les chiffres décrivent la journée sélectionnée, pas un parc en exploitation."}</p>
<div><b style={{"fontSize": "14px"} as CSSProperties}>{"Jour"}</b><div className="chips" id="eo-day" style={{"marginTop": "4px"} as CSSProperties}></div></div>
<p className="src" style={{"margin": "8px 0 0"} as CSSProperties} id="aw-eol-src">{"Aujourd'hui : vent du modèle ; demain et après-demain : prévision GFS affinée par ATOM Wind, quand le temps réel est servi."}</p></div>
<div className="kpis" id="kpis"></div>
<div className="card"><h3>{"Production du parc sur la journée"}</h3><div id="prodchart"></div></div>
<div className="turb" id="turbs"></div>
<div className="grid2">
<div className="card"><h3>{"Placer mon éolienne"}</h3>
<div className="form"><label htmlFor="tm">{"Modèle"}<select defaultValue="l" className="inp" id="tm"><option value="s">{"Petite : moyeu 30 m, 100 kW"}</option><option value="m">{"Moyenne : moyeu 50 m, 800 kW"}</option><option value="l">{"Grande : moyeu 80 m, 2 MW"}</option></select></label></div>
<p className="muted" style={{"fontSize": "14px", "margin": "8px 0"} as CSSProperties}>{"Touchez la carte pour placer l'éolienne."}</p>
<div className="mapbox" id="map4"><canvas id="c4"></canvas></div>
<div id="myt" style={{"marginTop": "10px"} as CSSProperties}></div>
</div>
<div className="steps">
<div className="card"><h3>{"Le vent sur toute la hauteur du rotor"}</h3><div id="rotor"><div className="rotor-layout"><div className="rotor-fig" id="rotor-fig"></div><div className="rotor-txt" id="rotor-txt"></div></div></div></div>
<div className="card"><h3 id="rose-t">{"D'où vient le vent au moyeu ?"}</h3><div id="rose"></div></div>
</div>
</div>
<div className="grid2">
<div className="card"><h3>{"Carte du gisement : où installer une éolienne ?"}</h3><p className="muted" style={{"fontSize": "14px", "margin": "0 0 8px"} as CSSProperties}>{"Taux de charge estimé sur la journée, à hauteur de moyeu du modèle choisi. Touchez la carte pour y placer votre éolienne."}</p>
<div className="gisbox"><canvas id="gis"></canvas></div>
<div className="legend"><span>{"Faible"}</span><span className="ramp gisramp"></span><span>{"Fort"}</span><span>{"· calculé tous les 1 km pour l'aperçu"}</span></div>
<div id="gisbest" style={{"marginTop": "8px"} as CSSProperties}></div></div>
<div className="card"><h3>{"Peut-on se fier au vent en altitude ?"}</h3>
<p style={{"margin": "0 0 8px"} as CSSProperties}>{"Seule mesure disponible en altitude : les ballons-sondes de Tunis-Carthage, deux fois par jour. Erreur vectorielle moyenne du modèle servi, par tranche de hauteur."}</p>
<div id="aw-val-sondes"><p className="muted" style={{"margin": "0"} as CSSProperties}>{"Chargement de l'évaluation du modèle servi…"}</p></div>
<p className="muted" style={{"fontSize": "14px", "margin": "8px 0 0"} as CSSProperties}>{"Les mesures en altitude sont rares : c'est la limite principale de l'espace éolien. Les mâts ANEO de 30 m et un mât de mesure sur site la renforceront."}</p></div>
</div>
<div className="card" id="aw-atlas" hidden={true}></div>
</section>
); });
