import { memo, type CSSProperties } from "react";
export default memo(function ViewAgri(){ return (
<section className="view" id="v-agri">
<div className="card">
<h2>{"Votre parcelle"}</h2>
<p className="muted" style={{"marginTop": "0"} as CSSProperties}>{"Les conseils s’adaptent à votre culture, à votre surface et à votre mode d’arrosage."}</p>
<div className="agform">
<label htmlFor="ag-place">{"Lieu"}<select className="inp" id="ag-place"></select></label>
<label htmlFor="ag-crop">{"Culture"}<select className="inp" id="ag-crop"></select></label>
<label htmlFor="ag-area">{"Surface (hectares)"}<input className="inp" id="ag-area" type="number" min="0.1" max="500" step="0.1" defaultValue="2" /></label>
<label htmlFor="ag-irr">{"Arrosage"}<select className="inp" id="ag-irr"><option value="0.9">{"Goutte-à-goutte"}</option><option value="0.75">{"Aspersion"}</option><option value="0.6">{"Gravitaire (rigoles)"}</option></select></label>
</div>
<div style={{"marginTop": "12px", "display": "flex", "gap": "12px 24px", "flexWrap": "wrap", "alignItems": "flex-end"} as CSSProperties}>
<div><b style={{"fontSize": "14px"} as CSSProperties}>{"Jour"}</b><div className="chips" id="ag-day" style={{"marginTop": "4px"} as CSSProperties}></div></div>
<div><b style={{"fontSize": "14px"} as CSSProperties}>{"Milieu"}</b><div className="chips" id="ag-milieu" style={{"marginTop": "4px"} as CSSProperties}></div></div>
</div>
</div>
<div className="alerts" id="ag-alerts"></div>
<div className="agcards" id="ag-top"></div>
<div className="agcards" id="ag-cards"></div>
<div className="card">
<h3>{"D'où viennent les chiffres"}</h3>
<ul className="agsrc">
<li><span className="srctag atom">{"Vent : ATOM Wind"}</span>{" Vent à votre emplacement, à 2 m et à 10 m, calculé par le modèle PC-FNO3D : en temps réel aujourd'hui, à partir de la prévision GFS affinée par le modèle pour demain et après-demain."}</li>
<li><span className="srctag">{"Température, humidité : GFS"}</span>{" Ces grandeurs ne sont pas calculées par ATOM Wind : elles viennent de la prévision GFS, servie avec le temps réel."}</li>
<li><span className="srctag">{"Méthode FAO-56"}</span>{" Besoin en eau calculé par la méthode de Penman-Monteith de la FAO."}</li>
</ul>
<p className="src" style={{"margin": "8px 0 0"} as CSSProperties}>{"Perspectives : qualité de l'air, poussière et conditions de stockage des récoltes, quand les capteurs ANEO mesureront ces grandeurs."}</p>
</div>
</section>
); });
