// ATOM Wind : vue 3D complète (cahier v12, section 17.2, V3D1 à V3D10).
// La maquette d'Ines garde sa scène (relief, particules, stations, coupe verticale, brise, graine) ;
// build_atom_wind.py y pose des accroches et lui fait appeler ATOM.vue3d(ctx) à la fin d'init3d.
// Ce fichier ajoute, sur les objets de la scène que ctx expose :
//   V3D1 exagération verticale × 1 à × 10, socle, trait de côte, flèche du nord et échelle
//   V3D2 lignes de courant 3D à 10, 120 et 500 m, colorées par la vitesse, flèche de direction
//   V3D3 hauteur au-dessus du sol (120 m par défaut) ou altitude, repère « moyeu éolien »
//   V3D4 coupe horizontale à la hauteur choisie (la coupe verticale est celle de la maquette)
//   V3D5 fiche d'un point : statut, incertitude, fiabilité, station la plus proche, profil à 80 %
//   V3D6 couches : stations, réseau ANEO prévu, éoliennes, parcelle, coupes, particules, lignes
//   V3D7 couleur par vitesse, incertitude ou fiabilité, avec légende chiffrée
//   V3D8 frise : choisir l'heure, rejouer la journée
//   V3D10 fluidité : nombre de particules et résolution ajustés pour tenir 30 images par seconde
(() => {
"use strict";
const A = window.ATOM = window.ATOM || {};

const Z80 = 1.2816;                         // demi-largeur de l'intervalle à 80 % en écarts-types
const H_SOL = [10, 30, 50, 80, 120, 150, 300, 500, 1000];
const H_ALT = [50, 100, 200, 300, 500, 750, 1000, 1500];
const MARQUES_SOL = { 10: "cultures", 30: "mât ANEO", 120: "moyeu" };
const H_LIGNES = { sol: [10, 120, 500], alt: [200, 500, 1000] };
const MOYEU_M = 120;
const PROFIL_Z = [10, 25, 50, 100, 150, 200, 300, 400, 500, 650, 800, 1000];
// incertitude (écart-type de la vitesse, m/s) : vert -> orange -> rouge, bornes de la légende
const COUL_SIGMA = [[0, [31, 157, 85]], [0.9, [217, 138, 0]], [1.5, [210, 58, 58]], [2.5, [120, 20, 40]]];
const COUL_FIAB = { good: [31, 157, 85], mid: [217, 138, 0], bad: [210, 58, 58] };

function rampe(T, s){
  if(s <= T[0][0]) return T[0][1];
  for(let i = 1; i < T.length; i++) if(s <= T[i][0]){
    const [a, ca] = T[i-1], [b, cb] = T[i], t = (s - a) / (b - a);
    return ca.map((c, j) => Math.round(c + (cb[j] - c) * t));
  }
  return T[T.length-1][1];
}

A.vue3d = ctx => {
  const { THREE, scene, cam, ctl, renderer, host, geo, pos, pts, pg, pp, pc, P, N, labels, objets,
          wind, sigma, relClass, idx, elev, wat, W, H, STATIONS, TURB, ANEO, state, speedColor, kmh,
          fmtSpeed, dirName, fromDeg, nearestName, openSheet, setHour, hs } = ctx;
  const EX0 = ctx.ex();
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const cfg = { mode: "sol", couleur: "vitesse", fact: 5,
                couches: { particules: true, lignes: true, stations: true, aneo: true, eoliennes: true,
                           parcelle: true, coupes: true, horizontale: false, colonnes: false } };
  const sol = (x, y) => { const k = idx(x, y); return wat[k] ? 0 : elev[k]; };
  // hauteur du point dans la scène : la maquette étire l'air × 1,4 par rapport au relief
  const ySc = (x, y, zSol) => sol(x, y) * ctx.ex() + Math.max(.15, zSol * ctx.ex() * 1.4);
  const zSol = (x, y) => cfg.mode === "sol" ? ctx.z3() : ctx.z3() - sol(x, y);

  // ---------------------------------------------------------------- incertitude et couleurs
  const sigmaModele = () => !!(A.aSigma && A.aSigma());
  function incert(x, y, z){
    if(sigmaModele()){ const s = A.sigma(x, y, z, state.hour); if(s !== null && isFinite(s)) return s; }
    return sigma(x, y, "reel").s;
  }
  function couleur(x, y, z, sp){
    if(cfg.couleur === "vitesse") return speedColor(kmh(sp));
    const s = incert(x, y, z);
    return cfg.couleur === "incertitude" ? rampe(COUL_SIGMA, s) : COUL_FIAB[relClass(s)[0]];
  }

  // ---------------------------------------------------------------- panneau de commandes
  const hauteurs = () => cfg.mode === "sol" ? H_SOL : H_ALT;
  const marques = document.querySelector("#v-carte3d .hslider .marks");
  const titreH = document.querySelector('#v-carte3d label[for="h3"] b');
  const legende = document.querySelector("#v-carte3d .hslider .legend");
  function poserHauteurs(zVoulu){
    const L = hauteurs();
    ctx.H3.splice(0, ctx.H3.length, ...L);
    hs.max = L.length - 1;
    let i = L.indexOf(zVoulu); if(i < 0) i = L.reduce((b, z, k) => Math.abs(z - zVoulu) < Math.abs(L[b] - zVoulu) ? k : b, 0);
    hs.value = i;
    if(marques) marques.innerHTML = L.map(z => `<span>${z.toLocaleString("fr-FR")} m${cfg.mode === "sol" && MARQUES_SOL[z] ? `<br>${MARQUES_SOL[z]}` : ""}</span>`).join("");
    hs.oninput();
  }
  const ancienInput = hs.oninput;
  hs.oninput = () => {
    ancienInput();
    if(titreH) titreH.innerHTML = `${cfg.mode === "sol" ? "Hauteur au-dessus du sol" : "Altitude (au-dessus de la mer)"} : <span id="h3txt">${ctx.z3().toLocaleString("fr-FR")} m</span>${cfg.mode === "sol" && ctx.z3() === MOYEU_M ? " · moyeu éolien" : ""}`;
    sale = true;
  };
  function poserLegende(){
    if(!legende) return;
    if(cfg.couleur === "vitesse"){
      legende.innerHTML = `<span>0</span><span class="ramp"></span><span>62 km/h et plus</span><span>· vitesse du vent</span>`;
    } else if(cfg.couleur === "incertitude"){
      const g = COUL_SIGMA.map(([s, c]) => `rgb(${c}) ${s / 2.5 * 100}%`).join(",");
      legende.innerHTML = `<span>0</span><span class="ramp" style="background:linear-gradient(90deg,${g})"></span><span>2,5 m/s</span>`
        + `<span>· écart-type ${sigmaModele() ? "calibré du modèle" : "estimé selon la distance aux stations"}</span>`;
    } else {
      legende.innerHTML = ["good", "mid", "bad"].map((k, i) => `<span class="pill ${k}">${["bonne < 0,9 m/s", "moyenne 0,9 à 1,5", "faible > 1,5 m/s"][i]}</span>`).join("")
        + `<span>· fiabilité (écart-type de la vitesse)</span>`;
    }
  }

  const pan = document.createElement("details");
  pan.className = "ov glass"; pan.id = "aw3d";
  pan.style.cssText = "right:10px;top:58px;display:block;width:min(300px,calc(100% - 20px));max-height:calc(100% - 200px);overflow:auto;font-size:14px;z-index:3";
  pan.open = innerWidth > 760 && host.clientHeight > 560;
  const coche = (k, t) => `<label style="display:flex;gap:6px;align-items:center"><input type="checkbox" data-c="${k}" ${cfg.couches[k] ? "checked" : ""}>${t}</label>`;
  pan.innerHTML = `<summary style="cursor:pointer;font-weight:700">Affichage 3D</summary>
    <div style="display:grid;gap:8px;margin-top:8px">
      <label>Relief exagéré : <b id="aw3d-ext">× ${cfg.fact}</b><input type="range" id="aw3d-ex" min="1" max="10" step="1" value="${cfg.fact}" style="width:100%"></label>
      <div role="radiogroup" aria-label="Référence de hauteur" style="display:flex;gap:6px;flex-wrap:wrap">
        <label><input type="radio" name="aw3d-mode" value="sol" checked> au-dessus du sol</label>
        <label><input type="radio" name="aw3d-mode" value="alt"> altitude</label></div>
      <label>Couleur <select class="inp" id="aw3d-coul"><option value="vitesse">vitesse</option><option value="incertitude">incertitude</option><option value="fiabilite">fiabilité</option></select></label>
      <fieldset style="border:0;padding:0;margin:0;display:grid;gap:2px"><legend class="muted" style="font-size:13px">Couches</legend>
        ${coche("particules", "particules")}${coche("lignes", "lignes de courant (3 hauteurs)")}${coche("horizontale", "coupe horizontale à la hauteur choisie")}${coche("colonnes", "colonnes de vent par nœud")}
        ${coche("coupes", "coupe verticale")}${coche("stations", "stations de mesure")}${coche("aneo", "réseau ANEO prévu")}
        ${coche("eoliennes", "éoliennes et moyeu")}${coche("parcelle", "ma parcelle")}</fieldset>
      <div><label for="aw3d-h">Heure : <b id="aw3d-ht">${state.hour} h</b></label>
        <div style="display:flex;gap:6px;align-items:center"><input type="range" id="aw3d-h" min="0" max="23" step="1" value="${state.hour}" style="flex:1">
        <button class="btn" id="aw3d-rejouer" type="button">Rejouer la journée</button></div></div>
      <span class="muted" id="aw3d-ips" style="font-size:12px"></span>
    </div>`;
  host.appendChild(pan);

  // ---------------------------------------------------------------- V3D1 : socle, côte, nord, échelle
  let socle = null, cote = null;
  function construireSocle(){
    if(socle){ scene.remove(socle); socle.geometry.dispose(); }
    const ex = ctx.ex(), bas0 = -1.2 - 400 * ex, n = 160, v = [], ind = [];
    const bord = [];
    for(let i = 0; i <= n; i++) bord.push([W * i / n, 0]);
    for(let i = 1; i <= n; i++) bord.push([W, H * i / n]);
    for(let i = 1; i <= n; i++) bord.push([W - W * i / n, H]);
    for(let i = 1; i < n; i++) bord.push([0, H - H * i / n]);
    bord.push([0, 0]);
    bord.forEach(([x, y], i) => {
      const xx = Math.min(W - 1e-6, x), yy = Math.min(H - 1e-6, y), k = idx(xx, yy);
      const haut = wat[k] ? -.02 : elev[k] * ex;
      v.push(x - W/2, haut, -(y - H/2), x - W/2, bas0, -(y - H/2));
      if(i > 0){ const a = 2*(i-1), b = 2*i; ind.push(a, a+1, b, b, a+1, b+1); }
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(v, 3)); g.setIndex(ind); g.computeVertexNormals();
    socle = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: 0xb8925a, side: THREE.DoubleSide }));
    scene.add(socle);
  }
  function construireCote(){
    // limite terre / mer sur la grille de la page, au niveau de la mer
    const nx = ctx.NX, ny = ctx.NY, G = ctx.G, s = [];
    const mer = (i, j) => wat[j*nx + i] === 1;
    for(let j = 0; j < ny; j++) for(let i = 0; i < nx; i++){
      if(i < nx-1 && mer(i, j) !== mer(i+1, j)){ const x = (i+1)/G - W/2; s.push(x, .04, -(j/G - H/2), x, .04, -((j+1)/G - H/2)); }
      if(j < ny-1 && mer(i, j) !== mer(i, j+1)){ const z = -((j+1)/G - H/2); s.push(i/G - W/2, .04, z, (i+1)/G - W/2, .04, z); }
    }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(s, 3));
    cote = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x12263a, transparent: true, opacity: .75 }));
    scene.add(cote);
  }
  construireCote();
  // échelle de 10 km le long du bord sud, avec ses graduations
  const echelle = new THREE.Group(); scene.add(echelle);
  {
    const x0 = 2 - W/2, z = H/2 + 1.6, m = new THREE.LineBasicMaterial({ color: 0x12263a });
    const p = [x0, .1, z, x0 + 10, .1, z];
    for(let k = 0; k <= 10; k += 5) p.push(x0 + k, .1, z - .5, x0 + k, .1, z + .5);
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    echelle.add(new THREE.LineSegments(g, m));
    const d = document.createElement("div"); d.className = "lbl3d"; d.textContent = "10 km"; host.appendChild(d);
    labels.push({ d, v: new THREE.Vector3(x0 + 5, .4, z), fixe: true });
  }
  // flèche du nord : tourne avec la caméra (le nord de la scène est -z)
  const nord = document.createElement("div");
  nord.className = "ov glass"; nord.setAttribute("role", "img"); nord.setAttribute("aria-label", "Direction du nord");
  nord.style.cssText = "right:10px;bottom:10px;width:54px;height:54px;padding:4px;display:grid;place-items:center";
  nord.innerHTML = `<svg viewBox="-20 -20 40 40" width="44" height="44"><g id="aw3d-n"><path d="M0,-15 L6,4 L0,0 L-6,4 Z" fill="#d23a3a"/><path d="M0,15 L6,4 L0,0 L-6,4 Z" fill="currentColor" opacity=".35"/></g><text id="aw3d-nt" x="0" y="0" text-anchor="middle" dominant-baseline="central" font-size="9" font-weight="700">N</text></svg>`;
  host.appendChild(nord);
  const fN = nord.querySelector("#aw3d-n"), tN = nord.querySelector("#aw3d-nt");
  const vA = new THREE.Vector3(), vB = new THREE.Vector3();
  function tournerNord(){
    vA.copy(ctl.target).project(cam); vB.copy(ctl.target).add(new THREE.Vector3(0, 0, -10)).project(cam);
    const a = Math.atan2(vB.x - vA.x, vB.y - vA.y) * 180 / Math.PI;
    fN.setAttribute("transform", `rotate(${a.toFixed(1)})`);
    const r = a * Math.PI / 180; tN.setAttribute("x", (Math.sin(r) * 13).toFixed(1)); tN.setAttribute("y", (-Math.cos(r) * 13).toFixed(1));
  }

  // ---------------------------------------------------------------- V3D1 : exagération
  const posSt = (o, dy) => o.m.position.set(o.p[0] - W/2, sol(...o.p) * ctx.ex() + dy, -(o.p[1] - H/2));
  function appliquerExageration(f){
    const ex0 = ctx.ex(), ex = f / 1000, r = ex / ex0;
    ctx.setEx(ex);
    for(let i = 0; i < pos.count; i++){
      const x = pos.getX(i) + W/2, y = -pos.getZ(i) + H/2, k = idx(x, y);
      pos.setY(i, wat[k] ? -.02 : elev[k] * ex);
    }
    pos.needsUpdate = true; geo.computeVertexNormals(); geo.computeBoundingSphere();
    for(const o of objets.stations) posSt(o, .6);
    for(const o of objets.turb){ o.m.scale.y = (MOYEU_M * 1.4 * ex) / (80 * EX0 * 4); posSt(o, MOYEU_M * 1.4 * ex / 2); }
    for(const l of labels) if(!l.fixe && l.p) l.v.y = sol(...l.p) * ex + 1.6;
    for(const o of ctx.cutObjs()){                         // coupe verticale déjà tracée : même relief
      if(o.isLine){ const a = o.geometry.attributes.position; for(let i = 0; i < a.count; i++) a.setY(i, (a.getY(i) - .35) * r + .35); a.needsUpdate = true; }
      else o.position.y = (o.position.y - .5) * r + .5;
    }
    ctx.pin.visible = false; ctx.hring.visible = false;
    construireSocle(); placerMarqueurs(); sale = true;
  }

  // ---------------------------------------------------------------- V3D3 / V3D6 : moyeu, ANEO, parcelle
  const moyeux = new THREE.Group(), aneo = new THREE.Group(), parcelle = new THREE.Group();
  scene.add(moyeux, aneo, parcelle);
  const matMoyeu = new THREE.MeshBasicMaterial({ color: 0x1e6fb8, transparent: true, opacity: .85, side: THREE.DoubleSide });
  for(const o of objets.turb){ const r = new THREE.Mesh(new THREE.RingGeometry(.35, .6, 24), matMoyeu); r.rotation.x = -Math.PI/2; r.userData.p = o.p; moyeux.add(r); }
  const lblMoyeu = document.createElement("div"); lblMoyeu.className = "lbl3d"; lblMoyeu.textContent = `moyeu ${MOYEU_M} m`; host.appendChild(lblMoyeu);
  const lMoyeu = { d: lblMoyeu, v: new THREE.Vector3(), fixe: true }; labels.push(lMoyeu);
  const geoAneo = new THREE.CylinderGeometry(.12, .12, .8, 8), matAneo = new THREE.MeshLambertMaterial({ color: 0xe08a1e });
  for(const p of (ANEO || [])){ const m = new THREE.Mesh(geoAneo, matAneo); m.userData.p = p; aneo.add(m); }
  const carre = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), new THREE.MeshBasicMaterial({ color: 0x2e8b3a, transparent: true, opacity: .55, side: THREE.DoubleSide, depthWrite: false }));
  carre.rotation.x = -Math.PI/2; parcelle.add(carre);
  const lblParc = document.createElement("div"); lblParc.className = "lbl3d"; host.appendChild(lblParc);
  const lParc = { d: lblParc, v: new THREE.Vector3(), fixe: true }; labels.push(lParc);
  function placerMarqueurs(){
    const ex = ctx.ex();
    moyeux.children.forEach(r => r.position.set(r.userData.p[0] - W/2, sol(...r.userData.p) * ex + MOYEU_M * 1.4 * ex + .02, -(r.userData.p[1] - H/2)));
    if(moyeux.children[0]) lMoyeu.v.copy(moyeux.children[0].position).add(new THREE.Vector3(0, 1.2, 0));
    aneo.children.forEach(m => m.position.set(m.userData.p[0] - W/2, sol(...m.userData.p) * ex + .4, -(m.userData.p[1] - H/2)));
    const p = state.pin; carre.position.set(p[0] - W/2, sol(...p) * ex + .06, -(p[1] - H/2));
    lParc.v.set(p[0] - W/2, sol(...p) * ex + 1.4, -(p[1] - H/2)); lblParc.textContent = (state.pinName || "Ma parcelle").split(" (")[0];
  }

  // ---------------------------------------------------------------- V3D2 : lignes de courant
  const NPAS = 36, PAS_KM = .45, ECART_KM = 4.5;
  const nGr = Math.ceil(W / ECART_KM) * Math.ceil(H / ECART_KM) * 3;
  const lignes = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: .9 }));
  const lp = new Float32Array(nGr * NPAS * 6), lc = new Float32Array(nGr * NPAS * 6);
  lignes.geometry.setAttribute("position", new THREE.BufferAttribute(lp, 3));
  lignes.geometry.setAttribute("color", new THREE.BufferAttribute(lc, 3));
  lignes.frustumCulled = false; scene.add(lignes);
  const fleches = new THREE.InstancedMesh(new THREE.ConeGeometry(.22, .7, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }), nGr);
  fleches.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(nGr * 3), 3);
  fleches.frustumCulled = false; scene.add(fleches);
  const M4 = new THREE.Matrix4(), Q = new THREE.Quaternion(), HAUT = new THREE.Vector3(0, 1, 0), DIR = new THREE.Vector3(), POS = new THREE.Vector3(), UN = new THREE.Vector3(1, 1, 1), C = new THREE.Color();
  function construireLignes(){
    let n = 0, nf = 0;
    const L = H_LIGNES[cfg.mode];
    for(const zc of L) for(let gx = ECART_KM / 2; gx < W; gx += ECART_KM) for(let gy = ECART_KM / 2; gy < H; gy += ECART_KM){
      let x = gx, y = gy, k = 0, dernier = null;
      for(; k < NPAS; k++){
        const z = cfg.mode === "sol" ? zc : zc - sol(x, y);
        if(z < 2) break;
        const [u, v] = wind(x, y, z, state.hour), sp = Math.hypot(u, v);
        if(sp < .3) break;
        // pas de Runge-Kutta 2 sur la direction : la ligne suit le vent, la couleur dit sa force
        const xm = x + u / sp * PAS_KM / 2, ym = y + v / sp * PAS_KM / 2;
        const zm = cfg.mode === "sol" ? zc : zc - sol(xm, ym);
        const [um, vm] = wind(xm, ym, Math.max(zm, 2), state.hour), sm = Math.hypot(um, vm) || sp;
        const x2 = x + um / sm * PAS_KM, y2 = y + vm / sm * PAS_KM;
        if(x2 < 0 || x2 > W || y2 < 0 || y2 > H) break;
        const z2 = cfg.mode === "sol" ? zc : zc - sol(x2, y2);
        if(z2 < 2) break;
        const c = speedColor(kmh(sp)), o = n * 6;
        lp[o] = x - W/2; lp[o+1] = ySc(x, y, z); lp[o+2] = -(y - H/2);
        lp[o+3] = x2 - W/2; lp[o+4] = ySc(x2, y2, z2); lp[o+5] = -(y2 - H/2);
        for(let q = 0; q < 6; q += 3){ lc[o+q] = c[0]/255; lc[o+q+1] = c[1]/255; lc[o+q+2] = c[2]/255; }
        n++; dernier = [x, y, x2, y2, z, z2, c]; x = x2; y = y2;
      }
      if(dernier && k >= 3){
        const [xa, ya, xb, yb, za, zb, c] = dernier;
        POS.set(xb - W/2, ySc(xb, yb, zb), -(yb - H/2));
        DIR.set(xb - xa, ySc(xb, yb, zb) - ySc(xa, ya, za), -(yb - ya)).normalize();
        Q.setFromUnitVectors(HAUT, DIR); M4.compose(POS, Q, UN);
        fleches.setMatrixAt(nf, M4); fleches.setColorAt(nf, C.setRGB(c[0]/255, c[1]/255, c[2]/255)); nf++;
      }
    }
    lignes.geometry.setDrawRange(0, n * 2);
    lignes.geometry.attributes.position.needsUpdate = true; lignes.geometry.attributes.color.needsUpdate = true;
    fleches.count = nf; fleches.instanceMatrix.needsUpdate = true;
    if(fleches.instanceColor) fleches.instanceColor.needsUpdate = true;
  }

  // ---------------------------------------------------------------- BF32 : colonnes de vent par noeud
  // Tous les 5 km, une pile de fleches de 10 m a 1 000 m au-dessus du sol : direction et force du vent a
  // chaque hauteur (couleur selon le reglage de couleur), pour lire le cisaillement et la rotation.
  const H_COL = [10, 50, 100, 200, 500, 1000], PAS_COL_KM = 5;
  const nCol = Math.ceil(W / PAS_COL_KM) * Math.ceil(H / PAS_COL_KM) * H_COL.length;
  const geoFl = new THREE.ConeGeometry(.28, 1.1, 6); geoFl.rotateX(Math.PI / 2);     // pointe vers +z
  const colonnes = new THREE.InstancedMesh(geoFl, new THREE.MeshBasicMaterial({ color: 0xffffff }), nCol);
  colonnes.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(nCol * 3), 3);
  colonnes.frustumCulled = false; colonnes.visible = false; scene.add(colonnes);
  const mats = new THREE.Group(); scene.add(mats);
  function construireColonnes(){
    let n = 0;
    const tiges = [];
    for(let gx = PAS_COL_KM / 2; gx < W; gx += PAS_COL_KM) for(let gy = PAS_COL_KM / 2; gy < H; gy += PAS_COL_KM){
      if(wat[idx(gx, gy)]) continue;
      const base = sol(gx, gy) * ctx.ex();
      tiges.push(gx - W/2, base, -(gy - H/2), gx - W/2, ySc(gx, gy, H_COL[H_COL.length - 1]), -(gy - H/2));
      for(const z of H_COL){
        const [u, v] = wind(gx, gy, z, state.hour), sp = Math.hypot(u, v);
        if(!(sp > .2)) continue;
        POS.set(gx - W/2, ySc(gx, gy, z), -(gy - H/2));
        DIR.set(u / sp, 0, -v / sp);                       // x vers l'est, -z vers le nord
        Q.setFromUnitVectors(new THREE.Vector3(0, 0, 1), DIR);
        const e = .6 + Math.min(sp, 25) / 25 * 1.4;        // longueur selon la force
        M4.compose(POS, Q, new THREE.Vector3(e, e, e));
        colonnes.setMatrixAt(n, M4);
        const c = couleur(gx, gy, z, sp); colonnes.setColorAt(n, C.setRGB(c[0]/255, c[1]/255, c[2]/255));
        n++;
      }
    }
    colonnes.count = n; colonnes.instanceMatrix.needsUpdate = true; colonnes.instanceColor.needsUpdate = true;
    mats.clear();
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(tiges, 3));
    mats.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x12263a, transparent: true, opacity: .35 })));
  }

  // ---------------------------------------------------------------- V3D4 : coupe horizontale
  const SX = 96, SY = 80;
  const hg = new THREE.PlaneGeometry(W, H, SX, SY); hg.rotateX(-Math.PI/2);
  const hcol = new Float32Array(hg.attributes.position.count * 3);
  hg.setAttribute("color", new THREE.BufferAttribute(hcol, 3));
  const horiz = new THREE.Mesh(hg, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: .72, side: THREE.DoubleSide, depthWrite: false }));
  horiz.renderOrder = 2; horiz.visible = false; scene.add(horiz);
  function construireHorizontale(){
    const a = hg.attributes.position;
    for(let i = 0; i < a.count; i++){
      const x = Math.min(W - 1e-6, Math.max(0, a.getX(i) + W/2)), y = Math.min(H - 1e-6, Math.max(0, -a.getZ(i) + H/2));
      const z = zSol(x, y);
      if(z < 2){ a.setY(i, sol(x, y) * ctx.ex() - .05); hcol[i*3] = hcol[i*3+1] = hcol[i*3+2] = .5; continue; }   // sous le relief : caché
      const [u, v] = wind(x, y, z, state.hour), c = couleur(x, y, z, Math.hypot(u, v));
      a.setY(i, ySc(x, y, z)); hcol[i*3] = c[0]/255; hcol[i*3+1] = c[1]/255; hcol[i*3+2] = c[2]/255;
    }
    a.needsUpdate = true; hg.attributes.color.needsUpdate = true; hg.computeBoundingSphere();
  }

  // ---------------------------------------------------------------- particules (remplace la boucle de la maquette)
  let actives = N;
  pts.frustumCulled = false;
  function particules(){
    if(!pts.visible) return true;
    const z0 = ctx.z3(), ex = ctx.ex();
    for(let i = 0; i < actives; i++){
      const p = P[i];
      let z = cfg.mode === "sol" ? z0 : z0 - sol(p.x, p.y);
      let [u, v] = wind(p.x, p.y, Math.max(z, 2), state.hour);
      if(!reduce){ p.x += u * .02; p.y += v * .02; }
      if(++p.a > 140 || p.x < 0 || p.x > W || p.y < 0 || p.y > H || z < 2){
        for(let t = 0; t < 6; t++){ p.x = Math.random() * W; p.y = Math.random() * H; z = cfg.mode === "sol" ? z0 : z0 - sol(p.x, p.y); if(z >= 2) break; }
        p.a = Math.random() * 40 | 0; [u, v] = wind(p.x, p.y, Math.max(z, 2), state.hour);
      }
      const o = i * 3;
      if(z < 2){ pp[o] = p.x - W/2; pp[o+1] = -50; pp[o+2] = -(p.y - H/2); continue; }
      pp[o] = p.x - W/2; pp[o+1] = sol(p.x, p.y) * ex + Math.max(.15, z * ex * 1.4); pp[o+2] = -(p.y - H/2);
      const c = couleur(p.x, p.y, z, Math.hypot(u, v));
      pc[o] = c[0]/255; pc[o+1] = c[1]/255; pc[o+2] = c[2]/255;
    }
    pg.setDrawRange(0, actives);
    pg.attributes.position.needsUpdate = true; pg.attributes.color.needsUpdate = true;
    return true;
  }

  // ---------------------------------------------------------------- V3D10 : fluidité
  let tPrec = performance.now(), nIm = 0, tFen = tPrec, lent = 0, ips = 60;
  const ipsEl = pan.querySelector("#aw3d-ips");
  function mesurer(){
    const t = performance.now();
    if(t - tPrec > 250){ nIm = 0; tFen = t; }          // la vue était cachée : nouvelle fenêtre
    nIm++;
    if(t - tFen >= 1000){
      ips = nIm * 1000 / (t - tFen); nIm = 0; tFen = t;
      if(ips < 30 && host.offsetParent){
        lent++;
        if(lent >= 2){                                 // deux secondes sous 30 images/s : on allège
          if(renderer.getPixelRatio() > 1){ renderer.setPixelRatio(1); renderer.setSize(host.clientWidth, host.clientHeight, false); }
          else if(actives > 800) actives = Math.round(actives * .7);
          lent = 0;
        }
      } else lent = 0;
      ipsEl.textContent = `${Math.round(ips)} images/s · ${actives.toLocaleString("fr-FR")} particules`;
    }
    tPrec = t;
  }

  // ---------------------------------------------------------------- recalculs (heure, journée, réglages)
  let sale = true, cle = "", tCalc = 0;
  function image(){
    mesurer(); tournerNord();
    const k = `${state.hour}|${ctx.z3()}|${cfg.mode}|${cfg.couleur}|${ctx.ex()}|${A.jour ? (A.jour.date || "") + (A.jour.instants || []).length : "ex"}`;
    if(k !== cle) sale = true;
    const t = performance.now();
    if(sale && t - tCalc > 350){
      cle = k; sale = false; tCalc = t;
      if(cfg.couches.lignes) construireLignes();
      if(cfg.couches.colonnes) construireColonnes();
      if(cfg.couches.horizontale) construireHorizontale();
      placerMarqueurs();
    }
    if(+hTl.value !== state.hour){ hTl.value = state.hour; hTt.textContent = `${state.hour} h`; }
  }

  // ---------------------------------------------------------------- V3D5 : fiche d'un point
  function fiche(x, y){
    const z = Math.max(2, zSol(x, y));
    openSheet(x, y, Math.round(z));
    const sh = document.getElementById("sheet"); if(!sh) return true;
    const s = incert(x, y, z), [rc, rl] = relClass(s);
    const prochesSt = STATIONS.map(st => [st.n, Math.hypot(x - st.p[0], y - st.p[1])]).sort((a, b) => a[1] - b[1])[0];
    const mesure = prochesSt[1] < .75;
    // statut, fiabilité, incertitude et station la plus proche à la place des pastilles de la maquette
    const pastilles = sh.querySelector(".pill.m"), ligne0 = pastilles && pastilles.closest(".row");
    if(ligne0) ligne0.innerHTML = `<span class="pill ${mesure ? "good" : "m"}">${mesure ? "Point mesuré (station)" : "Point estimé (calculé)"}</span>`
      + `<span class="pill ${rc}">Fiabilité ${rl}</span>`;
    const bloc = document.createElement("p");
    bloc.style.cssText = "margin:0;font-size:15px";
    bloc.innerHTML = `Incertitude : ± ${fmtSpeed(s)} (écart-type, ${sigmaModele() ? "calibré sur l'ensemble du modèle" : "estimé selon la distance aux stations"}). `
      + `Station la plus proche : <b>${prochesSt[0]}</b>, à ${prochesSt[1].toFixed(1).replace(".", ",")} km.`
      + (cfg.mode === "alt" ? ` Altitude ${ctx.z3()} m, soit ${Math.round(z)} m au-dessus du sol ici.` : "");
    const svg = sh.querySelector('svg[aria-label="Profil vertical du vent"]');
    if(svg){
      // profil avec l'intervalle à 80 % (vitesse ± 1,28 écart-type)
      const pr = PROFIL_Z.map(zz => { const [u, v] = wind(x, y, zz, state.hour), sp = Math.hypot(u, v), sg = incert(x, y, zz);
        return [kmh(sp), kmh(Math.max(0, sp - Z80 * sg)), kmh(sp + Z80 * sg)]; });
      const mx = Math.max(...pr.map(q => q[2])) * 1.1, Wp = 300, Hp = 170;
      const X = s2 => 34 + s2 / mx * (Wp - 44), Yz = zz => Hp - 18 - Math.log10(zz / 10) / 2 * (Hp - 30);
      const bande = pr.map((q, i) => `${X(q[2]).toFixed(1)},${Yz(PROFIL_Z[i]).toFixed(1)}`).concat(pr.slice().reverse().map((q, i) => `${X(q[1]).toFixed(1)},${Yz(PROFIL_Z[PROFIL_Z.length-1-i]).toFixed(1)}`)).join(" ");
      const ligne = pr.map((q, i) => `${X(q[0]).toFixed(1)},${Yz(PROFIL_Z[i]).toFixed(1)}`).join(" ");
      svg.innerHTML = [10, 100, 1000].map(zz => `<line x1="34" x2="${Wp-8}" y1="${Yz(zz)}" y2="${Yz(zz)}" stroke="var(--line)"/><text x="30" y="${Yz(zz)+4}" text-anchor="end">${zz} m</text>`).join("")
        + `<polygon points="${bande}" fill="var(--accent)" opacity=".18"/>`
        + `<polyline points="${ligne}" fill="none" stroke="var(--accent)" stroke-width="2.5"/>`
        + pr.map((q, i) => `<circle cx="${X(q[0])}" cy="${Yz(PROFIL_Z[i])}" r="3" fill="var(--accent)"/>`).join("")
        + `<text x="${Wp-8}" y="${Hp-4}" text-anchor="end">km/h →</text>`;
      const leg = document.createElement("div"); leg.className = "muted"; leg.style.fontSize = "13px";
      leg.textContent = `Bande claire : intervalle à 80 % (${sigmaModele() ? "incertitude du modèle" : "incertitude estimée"}).`;
      svg.after(leg);
    }
    const p = sh.querySelector("p");
    if(p) p.replaceWith(bloc); else (ligne0 || sh.querySelector("header")).after(bloc);
    return true;
  }

  // ---------------------------------------------------------------- commandes
  const exIn = pan.querySelector("#aw3d-ex"), exT = pan.querySelector("#aw3d-ext");
  exIn.oninput = () => { cfg.fact = +exIn.value; exT.textContent = `× ${cfg.fact}`; appliquerExageration(cfg.fact); };
  pan.querySelectorAll('input[name="aw3d-mode"]').forEach(r => r.onchange = () => {
    const ancien = cfg.mode; cfg.mode = r.value; if(ancien === cfg.mode) return;
    poserHauteurs(cfg.mode === "sol" ? MOYEU_M : 500);
  });
  const coul = pan.querySelector("#aw3d-coul");
  coul.onchange = () => { cfg.couleur = coul.value; poserLegende(); sale = true; };
  function appliquerCouches(){
    const c = cfg.couches;
    pts.visible = c.particules; lignes.visible = fleches.visible = c.lignes; horiz.visible = c.horizontale;
    colonnes.visible = mats.visible = c.colonnes;
    for(const o of objets.stations) o.m.visible = c.stations;
    for(const o of objets.turb) o.m.visible = c.eoliennes;
    moyeux.visible = c.eoliennes; lblMoyeu.hidden = !c.eoliennes; lMoyeu.cache = !c.eoliennes;
    aneo.visible = c.aneo; parcelle.visible = c.parcelle; lParc.cache = !c.parcelle;
    for(const o of ctx.cutObjs()) o.visible = c.coupes;
    for(const l of labels) if(l.st) l.cache = !c.stations;
    sale = true;
  }
  pan.querySelectorAll("input[data-c]").forEach(b => b.onchange = () => { cfg.couches[b.dataset.c] = b.checked; appliquerCouches(); });
  const hTl = pan.querySelector("#aw3d-h"), hTt = pan.querySelector("#aw3d-ht"), bRej = pan.querySelector("#aw3d-rejouer");
  hTl.oninput = () => { setHour(+hTl.value); hTt.textContent = `${hTl.value} h`; };
  let tRej = null;
  bRej.onclick = () => {
    if(tRej){ clearInterval(tRej); tRej = null; bRej.textContent = "Rejouer la journée"; return; }
    let h = 0; bRej.textContent = "Arrêter";
    const pas = () => { setHour(h); h++; if(h > 23){ clearInterval(tRej); tRej = null; bRej.textContent = "Rejouer la journée"; } };
    pas(); tRej = setInterval(pas, reduce ? 1800 : 900);
  };

  // état initial : × 5, hauteur 120 m au-dessus du sol (cahier V3D1, V3D3)
  appliquerExageration(cfg.fact);
  poserHauteurs(MOYEU_M); poserLegende(); appliquerCouches();
  return { particules, image, fiche, zSol,
           texteHauteur: (x, y) => cfg.mode === "sol" ? `à ${ctx.z3()} m du sol`
             : `à ${ctx.z3()} m d'altitude (${Math.max(0, Math.round(zSol(x, y)))} m du sol)`,
           cfg };
};
})();
