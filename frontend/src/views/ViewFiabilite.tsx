import { memo, type CSSProperties } from "react";
export default memo(function ViewFiabilite(){ return (
<section className="view" id="v-fiabilite">
<div className="grid2">
<div className="card">
<h2>{"Où le vent est-il bien connu ?"}</h2>
<div className="row" style={{"marginBottom": "10px"} as CSSProperties}><div className="seg" role="group" aria-label="Réseau de capteurs"><button id="m-reel" aria-pressed="true">{"Aujourd'hui : 3 stations"}</button><button id="m-aneo" aria-pressed="false">{"Avec le réseau ANEO prévu"}</button></div></div>
<div className="mapbox" id="map2"><canvas id="c2"></canvas></div>
<div className="legend"><span className="pill good">{"Fiabilité bonne"}</span><span className="pill mid">{"Moyenne"}</span><span className="pill bad">{"Faible"}</span><span>{"Les zones pâles sont moins sûres."}</span></div>
</div>
<div className="steps">
<div className="card" id="relstats"></div>
<div className="card">
<h3>{"Vérifié sur de vraies mesures"}</h3>
<p className="muted" style={{"marginTop": "0"} as CSSProperties}>{"Test sur des jours de 2022 jamais vus par le modèle. Chaque station a été cachée au modèle, qui a dû la retrouver. Écart moyen entre le vent calculé et le vent mesuré (plus c'est petit, mieux c'est)."}</p>
<div className="tblwrap"><table className="t">
<thead><tr><th>{"Mesure réelle"}</th><th style={{"textAlign": "right"} as CSSProperties}>{"ATOM Wind"}</th><th style={{"textAlign": "right"} as CSSProperties}>{"ERA5"}</th></tr></thead>
<tbody>
<tr><td>{"Anémomètre de Nabeul"}</td><td className="n"><b>{"8,7 km/h"}</b></td><td className="n">{"9,3 km/h"}</td></tr>
<tr><td>{"Anémomètre de Tunis-Carthage"}</td><td className="n"><b>{"9,7 km/h"}</b></td><td className="n">{"10,1 km/h"}</td></tr>
<tr><td>{"Anémomètre de Zaghouan"}</td><td className="n">{"8,2 km/h"}</td><td className="n"><b>{"7,8 km/h"}</b></td></tr>
<tr><td>{"Ballons-sondes de Tunis, 0 à 250 m"}</td><td className="n"><b>{"8,0 km/h"}</b></td><td className="n">{"8,3 km/h"}</td></tr>
</tbody></table></div>
<p className="muted" style={{"marginBottom": "0", "fontSize": "14px"} as CSSProperties}>{"ATOM Wind fait mieux que la réanalyse ERA5 à Nabeul, à Tunis-Carthage et sur les ballons-sondes, et un peu moins bien à Zaghouan, en bord de zone. Une mesure ponctuelle et une valeur calculée sur 100 m diffèrent toujours de quelques km/h."}</p>
</div>
</div>
</div>
<div className="card" id="aw-valeur" hidden={true}></div>
</section>
); });
