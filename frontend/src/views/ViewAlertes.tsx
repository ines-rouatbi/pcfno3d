import { memo, type CSSProperties } from "react";
export default memo(function ViewAlertes(){ return (
<section className="view" id="v-alertes">
<div className="card"><h2>{"Alertes de la journée"}</h2><p className="muted" style={{"margin": "0"} as CSSProperties}>{"Chaque alerte dit quoi, où, quand et que faire. Les plus urgentes sont en haut."}</p></div>
<div id="svc-alertes" className="sv-pile"></div>
<div className="grid2">
<div className="alerts" id="alertlist"></div>
<div className="steps">
<div className="card"><h3>{"Les alertes sur la carte"}</h3><p className="muted" style={{"marginTop": "0", "fontSize": "14px"} as CSSProperties}>{"Touchez une alerte pour la repérer sur la carte."}</p><div className="mapbox" id="map3"><canvas id="c3"></canvas></div></div>
<div className="card"><h3>{"Créer mon alerte"}</h3>
<div className="form">
<label htmlFor="al-place">{"Où ?"}<select className="inp" id="al-place"></select></label>
<label htmlFor="al-act">{"Pour quelle activité ?"}<select className="inp" id="al-act"></select></label>
<label htmlFor="al-chan">{"Me prévenir par"}<select className="inp" id="al-chan"><option>{"l'application"}</option><option>{"SMS"}</option><option>{"WhatsApp"}</option></select></label>
<button className="btn primary" id="al-add">{"Créer l'alerte"}</button>
</div>
<div className="mine" id="al-mine"></div>
<p className="muted" style={{"margin": "8px 0 0", "fontSize": "13px"} as CSSProperties}>{"Prototype : l'envoi par SMS ou WhatsApp est simulé. Vos alertes restent sur cet appareil."}</p>
</div>
</div>
</div>
</section>
); });
