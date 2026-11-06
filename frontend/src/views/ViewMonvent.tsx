import { memo, type CSSProperties } from "react";
export default memo(function ViewMonvent(){ return (
<section className="view" id="v-monvent">
<div className="grid2">
<div className="card">
<div className="step"><div className="lbl"><span className="num">{"1"}</span>{"Où êtes-vous ?"}</div>
<div className="row">
<select className="inp" id="town" aria-label="Choisir une ville" style={{"flex": "1", "minWidth": "180px"} as CSSProperties}></select>
</div>
<div className="mapbar"><div className="seg" role="group" aria-label="Fond de carte"><button id="bm-relief" aria-pressed="true">{"Relief"}</button><button id="bm-osm" aria-pressed="false">{"Vraie carte"}</button></div><button className="btn" id="locate" hidden={true}>{"Me localiser"}</button></div>
<div className="mapbox" id="map1"><canvas id="c1"></canvas><div className="maphint" id="hint1">{"Touchez la carte pour placer votre position"}</div></div>
<div className="lmapbox" id="lmapbox" hidden={true}><div id="lmap"></div><div className="glass lnotice" id="lnotice" hidden={true}></div></div>
<div className="legend"><span>{"Calme"}</span><span className="ramp"></span><span>{"Très fort"}</span><span className="tip" tabIndex={0} data-tip="Les traits qui bougent montrent le vent : leur couleur indique sa force, leur mouvement sa direction.">{"?"}</span></div>
</div>
</div>
<div className="steps">
<div className="card step"><div className="lbl"><span className="num">{"2"}</span>{"Que voulez-vous faire ?"}</div>
<div className="acts" id="acts"></div>
</div>
<div id="answer"></div>
<div className="card step"><div className="lbl"><span className="num">{"3"}</span>{"Votre journée, heure par heure"}</div>
<div className="day" id="day1"></div><div className="dayaxis" id="axis1"></div>
<div className="legend"><span className="pill good">{"Possible"}</span><span className="pill mid">{"Attention"}</span><span className="pill bad">{"Déconseillé"}</span><span>{"Touchez une heure pour la voir."}</span></div>
<div className="windows" id="win1"></div>
</div>
</div>
</div>
</section>
); });
