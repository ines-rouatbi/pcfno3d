import { memo, type CSSProperties } from "react";
export default memo(function ScienceViews(){return <>
<header>
<div className="brand">
<strong>{"PC-FNO3D"}</strong>
<span>{"Reconstruction du champ de vent 3D — réseau ANEO, Tunisie"}</span>
</div>
<div className="status" id="status">{"connexion…"}</div>
</header>

<section id="gate" className="gate" hidden={true}>
<div className="card">
<h2>{"Clé d'API"}</h2>
<p className="muted">{"Ce service est protégé. Collez une clé "}<code>{"pcf_…"}</code>{" pour continuer.\n      Elle reste dans ce navigateur et n'est envoyée qu'à ce serveur."}</p>
<input type="password" id="keyInput" placeholder="pcf_…" autoComplete="off" spellCheck="false" />
<div className="row">
<button id="keySubmit" className="primary">{"Se connecter"}</button>
<label className="check"><input type="checkbox" id="keyRemember" defaultChecked={true} />{" Se souvenir"}</label>
</div>
<p className="err" id="keyError" hidden={true}></p>
<p className="muted small">{"Pas de clé ? Sur le serveur :\n      "}<code>{"pcfno3d apikey create app-web"}</code></p>
</div>
</section>
<main id="app" hidden={true}>
<nav className="tabs" role="tablist">
<button role="tab" className="active" data-tab="carte">{"Carte"}</button>
<button role="tab" data-tab="point">{"Point & profil"}</button>
<button role="tab" data-tab="qualite">{"Qualité (DSO)"}</button>
<button role="tab" data-tab="integration">{"Intégration"}</button>
</nav>

<section className="tab active" id="tab-carte">
<div className="grid">
<div className="card">
<div className="canvas-wrap">
<canvas id="map" width="500" height="250"></canvas>
<div className="hint" id="mapHint">{"Cliquez sur la carte pour lire le vent en un point"}</div>
</div>
<div className="legend">
<span id="vmin">{"—"}</span><div className="ramp"></div><span id="vmax">{"—"}</span>
<span className="unit" id="unit">{"m/s"}</span>
</div>
</div>
<aside className="card controls">
<label htmlFor="instant">{"Instant"}</label>
<select id="instant"></select>
<label htmlFor="alt">{"Altitude ASL — "}<b id="altVal">{"—"}</b>{" m"}</label>
<input type="range" id="alt" min="25" max="1000" step="5" defaultValue="120" />
<div className="ticks" id="levelTicks"></div>
<label htmlFor="variable">{"Variable"}</label>
<select id="variable">
<option value="speed">{"Vitesse horizontale"}</option>
<option value="u">{"u — composante est"}</option>
<option value="v">{"v — composante nord"}</option>
<option value="w">{"w — composante verticale"}</option>
<option value="direction">{"Direction (d'où vient le vent)"}</option>
</select>
<label htmlFor="stride">{"Résolution — 1 point sur "}<b id="strideVal">{"4"}</b></label>
<input type="range" id="stride" min="1" max="16" step="1" defaultValue="4" />
<label className="check"><input type="checkbox" id="arrows" defaultChecked={true} />{" Flèches de direction"}</label>
<dl id="instantInfo"></dl>
</aside>
</div>
</section>

<section className="tab" id="tab-point">
<div className="grid">
<div className="card">
<h2>{"Profil vertical"}</h2>
<canvas id="profile" width="560" height="320"></canvas>
<p className="muted small" id="profileNote">{"Choisissez un point sur la carte ou saisissez\n          des coordonnées."}</p>
</div>
<aside className="card controls">
<label htmlFor="px">{"x (m, vers l'est)"}</label>
<input type="number" id="px" defaultValue="0" step="100" />
<label htmlFor="py">{"y (m, vers le nord)"}</label>
<input type="number" id="py" defaultValue="0" step="100" />
<label htmlFor="pz">{"Altitude ASL (m)"}</label>
<input type="number" id="pz" defaultValue="120" step="10" />
<button id="pointQuery" className="primary">{"Interroger"}</button>
<dl id="pointInfo"></dl>
</aside>
</div>
</section>

<section className="tab" id="tab-qualite">
<div className="card">
<h2>{"Objectifs de l'entreprise (DSO)"}</h2>
<p className="muted small">{"Dernière évaluation archivée par\n        "}<code>{"pcfno3d evaluate"}</code>{". Rien n'est recalculé ici : si aucune évaluation\n        n'existe, c'est indiqué plutôt qu'estimé."}</p>
<div id="dso"></div>
</div>
</section>

<section className="tab" id="tab-integration">
<div className="card">
<h2>{"Consommer ces données depuis une autre application"}</h2>
<h3>{"Serveur MCP"}</h3>
<p className="muted small">{"Transport streamable HTTP. La clé d'API passe dans l'en-tête\n        "}<code>{"X-API-Key"}</code>{". Outils : "}<code>{"grid_meta"}</code>{", "}<code>{"list_instants"}</code>{",\n        "}<code>{"wind_at_point"}</code>{", "}<code>{"wind_profile"}</code>{",\n        "}<code>{"wind_field_at_altitude"}</code>{", "}<code>{"instant_summary"}</code>{",\n        "}<code>{"dso_report"}</code>{", "}<code>{"export_instant"}</code>{" (admin)."}</p>
<pre id="mcpUrl"></pre>
<p className="muted small">{"Client local en stdio : "}<code>{"pcfno3d mcp"}</code></p>
<h3>{"API JSON"}</h3>
<pre id="apiExample"></pre>
<p className="muted small">{"Documentation interactive : "}<a id="docsLink" href="/api/docs">{"/api/docs"}</a></p>
<h3>{"Export binaire"}</h3>
<p className="muted small">{"Pour un volume complet, "}<code>{"POST /api/reconstruct"}</code>{" avec\n        "}<code>{"export: true"}</code>{" (portée admin) écrit "}<code>{"manifest.json"}</code>{" et des\n        binaires Float32 (ordre z, y, x ; "}<code>{"-9999"}</code>{" sous le relief), lisibles\n        ensuite via "}<code>{"/api/export/{stem}/…"}</code>{"."}</p>
<div className="row">
<button id="logout">{"Oublier la clé de ce navigateur"}</button>
</div>
</div>
</section>
</main>
</>;});
