// ATOM Wind : interface en français, arabe et anglais (cahier v12, E17 ; TB7 : espace agriculteur en
// français et en arabe). La page est écrite en français ; ce script traduit ce qu'elle affiche, à la volée,
// avec les dictionnaires i18n/ar.json et i18n/en.json (texte français -> traduction).
//   - une clé sans {n} est un texte exact ;
//   - une clé avec {0}, {1}... est un motif : « Vent {0}, à {1} m du sol » reconnaît « Vent nord, à 120 m
//     du sol », et les valeurs capturées sont traduites à leur tour (« nord » -> « شمال ») ;
//   - un texte absent du dictionnaire reste en français (dictionnaires à compléter avec
//     scripts/extraire_textes_atom_wind.py).
// Les nœuds de texte et les attributs aria-label, title et placeholder sont traduits, puis surveillés
// (MutationObserver) : ce que la page redessine est retraduit. Le texte dessiné dans un canvas ne l'est pas.
// Le choix est gardé dans localStorage (« aw-langue ») ; changer de langue recharge la page.
(() => {
"use strict";
const A = window.ATOM = window.ATOM || {};
const CLE = "aw-langue", LANGUES = ["fr", "ar", "en"];
let langue = "fr";
try { const l = localStorage.getItem(CLE); if(LANGUES.includes(l)) langue = l; } catch(e) {}
A.langue = () => langue;
A.choisirLangue = l => {
  if(!LANGUES.includes(l) || l === langue) return;
  try { localStorage.setItem(CLE, l); } catch(e) {}
  location.reload();
};

function brancherBoutons(){
  document.querySelectorAll("[data-lang]").forEach(b => {
    b.setAttribute("aria-pressed", String(b.dataset.lang === langue));
    b.onclick = () => A.choisirLangue(b.dataset.lang);
  });
}
if(document.readyState === "loading") document.addEventListener("DOMContentLoaded", brancherBoutons); else brancherBoutons();
if(langue === "fr") return;

document.documentElement.lang = langue;
if(langue === "ar") document.documentElement.dir = "rtl";

// ------------------------------------------------------------------ dictionnaire
const exacts = new Map(), motifs = [], memo = new Map();
const normal = t => t.replace(/\s+/g, " ").trim();
const echapper = t => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
function compiler(dico){
  for(const [fr, tr] of Object.entries(dico)){
    if(typeof tr !== "string" || !tr) continue;
    const k = normal(fr);
    if(!/\{\d+\}/.test(k)){ exacts.set(k, tr); continue; }
    const noms = [], morceaux = k.split(/(\{\d+\})/);
    // une valeur peut être vide (« {2} » final sans texte) : espaces souples autour des valeurs
    const re = morceaux.map(m => { const p = /^\{(\d+)\}$/.exec(m); if(p){ noms.push(p[1]); return "([\\s\\S]*?)"; }
      return echapper(m).replace(/^ | $/g, "\\s*"); }).join("");
    const fixe = morceaux.filter(m => !/^\{\d+\}$/.test(m)).join("").length;
    if(fixe < 2) continue;                       // « {0} {1} » : trop vague, reconnaîtrait tout
    motifs.push({ re: new RegExp(`^${re}$`), noms, tr, fixe });
  }
  motifs.sort((a, b) => b.fixe - a.fixe);       // le motif le plus précis d'abord
}
function traduire(t, prof = 0){
  const k = normal(t);
  if(!k || !/[A-Za-zÀ-ÿ]/.test(k)) return null;
  if(memo.has(k)) return memo.get(k);
  let r = exacts.get(k) ?? null;
  if(r === null && prof < 3){
    for(const m of motifs){
      const x = m.re.exec(k); if(!x) continue;
      r = m.tr;
      m.noms.forEach((n, i) => { const v = x[i+1]; r = r.split(`{${n}}`).join(traduire(v, prof + 1) ?? v); });
      break;
    }
  }
  // la page met parfois une majuscule à un mot du dictionnaire (« grande éolienne » -> « Grande éolienne »)
  if(r === null && prof < 3 && k[0] !== k[0].toLowerCase()){
    const b = traduire(k[0].toLowerCase() + k.slice(1), prof + 1);
    if(b) r = b[0].toUpperCase() + b.slice(1);
  }
  if(memo.size > 20000) memo.clear();
  memo.set(k, r);
  if(r === null && prof === 0 && nonTraduits.size < 3000) nonTraduits.add(k);
  return r;
}
// textes affichés sans traduction (pour compléter les dictionnaires : ATOM.nonTraduits dans la console)
const nonTraduits = new Set();
A.nonTraduits = nonTraduits;
A.traduire = t => traduire(t) ?? t;

// ------------------------------------------------------------------ application au document
const faits = new WeakMap();                    // nœud -> texte écrit par nous
const origines = new WeakMap();                 // nœud -> texte français écrit par la page
const ATTRS = ["aria-label", "title", "placeholder"];
function texte(n){
  const v = n.nodeValue;
  if(faits.get(n) === v) return;
  const p = n.parentNode;
  if(!p || p.nodeName === "SCRIPT" || p.nodeName === "STYLE" || p.nodeName === "TEXTAREA") return;
  origines.set(n, v);
  const tr = traduire(v);
  if(tr === null) return;
  const avant = /^\s*/.exec(v)[0], apres = /\s*$/.exec(v)[0];
  const nv = avant + tr + apres;
  faits.set(n, nv);
  if(nv !== v) n.nodeValue = nv;
}
// le texte français d'un élément tel que la page l'a écrit (pour le code qui relit ce qui est affiché)
A.texteFr = el => {
  if(!el) return "";
  let s = ""; const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  for(let n = w.nextNode(); n; n = w.nextNode()) s += origines.get(n) ?? n.nodeValue;
  return s;
};
function attributs(el){
  for(const a of ATTRS){
    const v = el.getAttribute(a); if(!v) continue;
    const marque = `aw-${a}`;
    if(el[marque] === v) continue;
    const tr = traduire(v);
    if(tr !== null){ el[marque] = tr; if(tr !== v) el.setAttribute(a, tr); }
  }
}
function parcourir(racine){
  if(racine.nodeType === 3){ texte(racine); return; }
  if(racine.nodeType !== 1) return;
  if(racine.nodeName === "SCRIPT" || racine.nodeName === "STYLE") return;
  attributs(racine);
  const w = document.createTreeWalker(racine, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
  for(let n = w.nextNode(); n; n = w.nextNode()){
    if(n.nodeType === 3) texte(n); else attributs(n);
  }
}
let enAttente = new Set(), prevu = false;
function vider(){
  prevu = false;
  const lot = enAttente; enAttente = new Set();
  for(const n of lot) if(n.isConnected) parcourir(n);
}
function demarrer(){
  parcourir(document.body);
  document.title = A.traduire(document.title);
  new MutationObserver(ms => {
    for(const m of ms){
      if(m.type === "characterData") enAttente.add(m.target);
      else if(m.type === "attributes") enAttente.add(m.target);
      else m.addedNodes.forEach(n => enAttente.add(n));
    }
    // une fois par image : une page qui redessine souvent ne paie qu'une passe
    if(!prevu){ prevu = true; requestAnimationFrame(vider); setTimeout(() => { if(prevu) vider(); }, 200); }
  }).observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTRS });
}

fetch(`i18n/${langue}.json`, { cache: "no-cache" })
  .then(r => r.ok ? r.json() : Promise.reject(new Error(r.status)))
  .then(d => {
    compiler(d);
    if(document.readyState === "loading") document.addEventListener("DOMContentLoaded", demarrer); else demarrer();
  })
  .catch(e => console.warn("ATOM Wind : traduction indisponible, page en français", e));
})();
