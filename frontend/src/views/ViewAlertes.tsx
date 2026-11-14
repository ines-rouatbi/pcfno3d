import { memo, type CSSProperties } from "react";
export default memo(function ViewAlertes(){ return (
<section className="view" id="v-alertes">
<div className="card"><h2>{"Alertes de la journée"}</h2><p className="muted" style={{"margin": "0"} as CSSProperties}>{"Chaque alerte dit quoi, où, quand et que faire. Les plus urgentes sont en haut."}</p></div>
<div className="grid2">
<div className="alerts" id="alertlist"></div>
<div className="steps">
<div className="card"><h3>{"Les alertes sur la carte"}</h3><p className="muted" style={{"marginTop": "0", "fontSize": "14px"} as CSSProperties}>{"Touchez une alerte pour la repérer sur la carte."}</p><div className="mapbox" id="map3"><canvas id="c3"></canvas></div></div>
<div className="card"><h3>{"Mes alertes SMS"}</h3>
<div id="al-login"><p className="muted" style={{"margin": "0 0 10px"} as CSSProperties}>{"Connectez-vous pour recevoir un SMS quand le vent devient gênant pour votre activité, là où vous êtes."}</p><button className="btn primary" id="al-login-btn" type="button">{"Se connecter"}</button></div>
<div className="form" id="al-form" hidden={true}>
<label htmlFor="al-place">{"Où ?"}<select className="inp" id="al-place"></select></label>
<label htmlFor="al-act">{"Pour quelle activité ?"}<select className="inp" id="al-act"></select></label>
<label htmlFor="al-tel">{"Numéro de téléphone"}<input className="inp" id="al-tel" type="tel" inputMode="tel" autoComplete="tel" placeholder="+216 12 345 678" /></label>
<label htmlFor="al-lang">{"Langue du SMS"}<select className="inp" id="al-lang"><option value="fr">{"Français"}</option><option value="ar">{"العربية"}</option></select></label>
<div className="err" id="al-err" role="alert"></div>
<button className="btn primary" id="al-add" type="button">{"Créer l'alerte"}</button>
</div>
<div className="mine" id="al-mine"></div>
<p className="muted" style={{"margin": "8px 0 0", "fontSize": "13px"} as CSSProperties}>{"Vos alertes sont gardées sur le serveur ATOM Wind ; vous pouvez les supprimer à tout moment."}</p>
</div>
</div>
</div>
</section>
); });
