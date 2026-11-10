import { renderTerritory } from '../territory-map';
export function startWorkspace(){
"use strict";
// ---------------- géographie (km) ----------------
// Zone réelle du Grand Tunis (maquette) ; ou le domaine du modèle servi quand ce n'est pas le Grand Tunis
// (phase 8 nationale : emprise, projection, blocs et stations de /atom-wind/grille, ATOM.domaine()).
const DOM = (window.ATOM && ATOM.domaine) ? ATOM.domaine() : null;
const LAT0 = DOM ? DOM.lat0 : 36.395, LON0 = DOM ? DOM.lon0 : 10.04, KX = DOM ? DOM.kx : 89.42, KY = DOM ? DOM.ky : 110.97, W = DOM ? DOM.w : 68, H = DOM ? DOM.h : 56;
const ZONE = DOM ? DOM.nom : "Grand Tunis";
const xy = (lat, lon) => [(lon - LON0) * KX, (lat - LAT0) * KY];
const poly = pts => pts.map(([a, b]) => xy(a, b));
// côte, lacs, villes et reliefs schématiques de la journée d'exemple : Grand Tunis seulement (le relief réel
// de l'API les remplace ; sur un autre domaine, l'eau vient de la grille)
const GULF = DOM ? [] : poly([[36.95,10.29],[36.90,10.29],[36.878,10.325],[36.85,10.33],[36.818,10.305],[36.79,10.29],[36.765,10.285],[36.735,10.33],[36.715,10.40],[36.72,10.46],[36.745,10.52],[36.80,10.555],[36.845,10.585],[36.90,10.625],[36.95,10.625]]);
const HAMM = DOM ? [] : poly([[36.36,10.60],[36.40,10.61],[36.43,10.68],[36.45,10.745],[36.48,10.79],[36.51,10.84],[36.36,10.84]]);
const LAKES = DOM ? [] : [[...xy(36.81,10.235), 4.6, 2.2], [...xy(36.775,10.14), 4.0, 2.4]];
const URB = DOM ? [] : [[...xy(36.80,10.18),6],[...xy(36.75,10.23),3],[...xy(36.87,10.31),2.5],[...xy(36.455,10.72),2],[...xy(36.725,10.34),1.6]];
const HILLS = DOM ? [] : [[36.61,10.33,720,2.6],[36.705,10.375,520,1.9],[36.36,10.10,950,6.5],[36.76,10.64,480,5],[36.58,10.43,240,4],[36.86,10.12,230,4],[36.50,10.22,260,5]].map(([a,b,h,s]) => [...xy(a,b),h,s]);
function inPoly(x, y, P){ let c=false; for(let i=0,j=P.length-1;i<P.length;j=i++){ const [xi,yi]=P[i],[xj,yj]=P[j]; if(((yi>y)!==(yj>y)) && x < (xj-xi)*(y-yi)/(yj-yi)+xi) c=!c; } return c; }
function waterType(x, y){ if(inPoly(x,y,GULF)||inPoly(x,y,HAMM)) return 1; for(const [cx,cy,rx,ry] of LAKES){ const dx=(x-cx)/rx, dy=(y-cy)/ry; if(dx*dx+dy*dy<1) return 2; } return 0; }
function elevRaw(x, y){ let h = 18 + 6*Math.sin(x*.31)*Math.cos(y*.27) + 4*Math.sin(x*.9+y*.4); for(const [cx,cy,hh,s] of HILLS){ const d2=(x-cx)**2+(y-cy)**2; h += hh*Math.exp(-d2/(2*s*s)); } return Math.max(2,h); }
function urbanAt(x,y){ for(const [cx,cy,r] of URB) if((x-cx)**2+(y-cy)**2<r*r) return true; return false; }

// grille à 0,5 km (Grand Tunis) ; celle des blocs de l'API sur un autre domaine (national : 4 km)
const G = DOM ? DOM.g : 2, NX = DOM ? DOM.nx : W*G, NY = DOM ? DOM.ny : H*G;
const AIRP = xy(36.851,10.227); // aéroport de Tunis-Carthage : terrain dégagé (pistes, herbe)
const elev = new Float32Array(NX*NY), wat = new Uint8Array(NX*NY), z0 = new Float32Array(NX*NY), sdist = new Float32Array(NX*NY);
for(let j=0;j<NY;j++) for(let i=0;i<NX;i++){ const x=(i+.5)/G, y=(j+.5)/G, k=j*NX+i, w=waterType(x,y); wat[k]=w; elev[k]= w ? 0 : elevRaw(x,y); z0[k]= w===1 ? .0002 : w===2 ? .001 : Math.hypot(x-AIRP[0],y-AIRP[1])<1.8 ? .03 : urbanAt(x,y) ? .8 : elev[k]>300 ? .4 : .08; }
if(window.ATOM) ATOM.appliquerGrille({NX, NY, G, elev, wat, z0, LAKES}); // relief réel (API)
// distance signée à la côte marine (positive dans les terres) : deux passes de chanfrein
(function(){ const INF=1e9, dl=new Float32Array(NX*NY).fill(INF), ds=new Float32Array(NX*NY).fill(INF);
  for(let k=0;k<NX*NY;k++){ if(wat[k]===1) dl[k]=0; else ds[k]=0; }
  for(const d of [dl,ds]){ const s=1/G, s2=s*1.414;
    for(let j=0;j<NY;j++) for(let i=0;i<NX;i++){ const k=j*NX+i; if(i>0) d[k]=Math.min(d[k],d[k-1]+s); if(j>0){ d[k]=Math.min(d[k],d[k-NX]+s); if(i>0) d[k]=Math.min(d[k],d[k-NX-1]+s2); if(i<NX-1) d[k]=Math.min(d[k],d[k-NX+1]+s2);} }
    for(let j=NY-1;j>=0;j--) for(let i=NX-1;i>=0;i--){ const k=j*NX+i; if(i<NX-1) d[k]=Math.min(d[k],d[k+1]+s); if(j<NY-1){ d[k]=Math.min(d[k],d[k+NX]+s); if(i<NX-1) d[k]=Math.min(d[k],d[k+NX+1]+s2); if(i>0) d[k]=Math.min(d[k],d[k+NX-1]+s2);} } }
  for(let k=0;k<NX*NY;k++) sdist[k] = wat[k]===1 ? -ds[k] : dl[k]; })();
const gx = new Float32Array(NX*NY), gy = new Float32Array(NX*NY), smooth = new Float32Array(NX*NY);
for(let j=0;j<NY;j++) for(let i=0;i<NX;i++){ const k=j*NX+i, il=Math.max(0,i-1), ir=Math.min(NX-1,i+1), jd=Math.max(0,j-1), ju=Math.min(NY-1,j+1);
  let a=sdist[j*NX+ir]-sdist[j*NX+il], b=sdist[ju*NX+i]-sdist[jd*NX+i]; const n=Math.hypot(a,b)||1; gx[k]=a/n; gy[k]=b/n;
  let s=0,c=0; for(let dj=-6;dj<=6;dj+=2) for(let di=-6;di<=6;di+=2){ const ii=Math.min(NX-1,Math.max(0,i+di)), jj=Math.min(NY-1,Math.max(0,j+dj)); s+=elev[jj*NX+ii]; c++; } smooth[k]=s/c; }
const idx = (x,y) => Math.min(NY-1,Math.max(0,Math.floor(y*G)))*NX + Math.min(NX-1,Math.max(0,Math.floor(x*G)));

// ---------------- modèle de vent d'exemple (journée d'été avec brise de mer) ----------------
function synoptic(h){ const day=Math.max(0, Math.sin((h-8)/12*Math.PI)), th = (305 + 40*day) * Math.PI/180, S = 3.0 + 2.2*day; return [-S*Math.sin(th), -S*Math.cos(th)]; }
function breeze(h){ if(h>=9 && h<=20) return 5.2*Math.sin((h-9)/11*Math.PI); if(h>=22 || h<=6) return -0.9; return 0; }
function wind(x, y, z, h){
  if(window.ATOM && ATOM.pret()) return ATOM.vent(x, y, z, h);
  const k = idx(x,y), [bu,bv] = synoptic(h), B = breeze(h), sd = sdist[k];
  const reach = sd > 0 ? Math.exp(-sd/13) : Math.exp(sd/9);
  const fz = Math.exp(-z/380), ret = -0.35 * smoothstep(350, 900, z);
  const br = B * reach * (fz + ret);
  const zz = Math.max(z, 1), zr = z0[k], low = Math.log(Math.min(zz,150)/zr)/Math.log(10/zr);
  const sp = 1 + Math.max(-.25, Math.min(.45, .0016*(elev[k]-smooth[k]))) * Math.exp(-z/400);
  let u = (bu*low + br*gx[k]*Math.max(.6,low*.8)) * sp, v = (bv*low + br*gy[k]*Math.max(.6,low*.8)) * sp;
  const a = smoothstep(150, 1000, z), veer = 20*Math.PI/180*a, gu = bu*2.4, gv = bv*2.4;
  const ru = gu*Math.cos(veer) - gv*Math.sin(veer), rv = gu*Math.sin(veer) + gv*Math.cos(veer);
  u = (1-a)*u + a*(ru + br*gx[k]); v = (1-a)*v + a*(rv + br*gy[k]);
  return [u, v];
}
function smoothstep(a,b,x){ const t=Math.min(1,Math.max(0,(x-a)/(b-a))); return t*t*(3-2*t); }
function gustF(x,y,z){ if(window.ATOM && ATOM.facteurRafale){ const f=ATOM.facteurRafale(x,y,z,state.hour); if(f) return f; } const zr=z0[idx(x,y)], g0 = zr>.5 ? 1.65 : zr>.05 ? 1.45 : 1.22; return 1 + (g0-1)*Math.exp(-z/300); }

// fiabilité
const STATIONS = DOM ? DOM.stations.map(s => ({n:s.nom, p:xy(s.lat,s.lon)})) : [{n:"Tunis-Carthage", p:xy(36.8504,10.2156)},{n:"Zaghouan", p:xy(36.4333,10.0833)},{n:"Nabeul", p:xy(36.45,10.7333)}];
const ANEO = (window.ATOM && ATOM.planXY(xy)) || (() => { const pts=[]; const cand=[]; for(let j=2;j<NY;j+=4) for(let i=2;i<NX;i+=4){ const k=j*NX+i; if(!wat[k] && elev[k]<650) cand.push([(i+.5)/G,(j+.5)/G]); }
  const all = STATIONS.map(s=>s.p); for(let n=0;n<36;n++){ let best=null, bd=-1; for(const c of cand){ let d=1e9; for(const p of all.concat(pts)) d=Math.min(d,Math.hypot(c[0]-p[0],c[1]-p[1])); if(d>bd){bd=d; best=c;} } pts.push(best); } return pts; })();
let relMode = "reel";
function sigma(x,y,mode=relMode){ let d=1e9, who=null; for(const s of STATIONS){ const dd=Math.hypot(x-s.p[0],y-s.p[1]); if(dd<d){d=dd; who=s.n;} }
  if(mode==="reel" && window.ATOM && ATOM.aSigma()){ const sg=ATOM.sigma(x,y,ACTS[state.act].z,state.hour); if(sg!==null && isFinite(sg)) return {s:sg, d, who}; }
  if(mode==="aneo"){ for(const p of ANEO){ const dd=Math.hypot(x-p[0],y-p[1]); if(dd<d){d=dd; who="un capteur ANEO";} } return {s:.35+.75*(1-Math.exp(-d/3)), d, who}; }
  return {s:.45+1.9*(1-Math.exp(-d/11)), d, who}; }
const relClass = s => s < .9 ? ["good","bonne"] : s < 1.5 ? ["mid","moyenne"] : ["bad","faible"];

// ---------------- unités et mots simples ----------------
let unit = "kmh";
const kmh = ms => ms*3.6;
const fmtSpeed = ms => unit==="kmh" ? `${Math.round(kmh(ms))} km/h` : `${(ms).toFixed(1).replace(".",",")} m/s`;
const DIRS = ["du nord","du nord-est","de l'est","du sud-est","du sud","du sud-ouest","de l'ouest","du nord-ouest"];
const fromDeg = (u,v) => (Math.atan2(-u,-v)*180/Math.PI + 360) % 360;
const dirName = (u,v) => DIRS[Math.round(fromDeg(u,v)/45)%8];
function beaufort(k){ const T=[[2,"Calme : la fumée monte tout droit."],[6,"Très léger : la fumée penche un peu."],[12,"Léger : on sent le vent sur le visage."],[20,"Petite brise : les feuilles bougent sans arrêt."],[29,"Jolie brise : la poussière s'envole, les petites branches bougent."],[39,"Bonne brise : les petits arbres se balancent."],[50,"Vent frais : les grosses branches bougent."],[62,"Grand frais : les arbres entiers bougent, marcher face au vent est pénible."],[75,"Coup de vent : des branches cassent."],[999,"Fort coup de vent : dégâts possibles."]]; return T.find(t=>k<t[0])[1]; }
const COL = [[0,[191,227,242]],[8,[89,195,160]],[16,[182,216,74]],[24,[244,196,48]],[34,[240,138,44]],[46,[224,65,75]],[62,[167,58,158]]];
function speedColor(k){ for(let i=1;i<COL.length;i++) if(k<=COL[i][0]){ const [a,ca]=COL[i-1],[b,cb]=COL[i], t=(k-a)/(b-a); return ca.map((c,j)=>Math.round(c+(cb[j]-c)*t)); } return COL[COL.length-1][1]; }
const arrowSvg = (deg,size=26,color="currentColor") => `<svg class="windarrow" width="${size}" height="${size}" viewBox="0 0 24 24" style="transform:rotate(${deg}deg)" aria-hidden="true"><path d="M12 3l6 9h-4v9h-4v-9H6z" fill="${color}"/></svg>`;
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// ---------------- activités : l'entrée de l'utilisateur ----------------
const ICONS = {
  spray:'<path d="M4 20h7M7 20V9h4l3-3h5M14 6v5M17 13c0 1.5-1 2-1 3M20 12c0 1.5-1 2-1 3M17 18c0 1.5-1 2-1 3" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/>',
  irr:'<path d="M12 3c3 4 5 7 5 10a5 5 0 0 1-10 0c0-3 2-6 5-10z" stroke="currentColor" stroke-width="2" fill="none"/>',
  turbine:'<path d="M12 10v11M9 21h6M12 10L12 3M12 10l6 3.5M12 10l-6 3.5" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/><circle cx="12" cy="10" r="1.6" fill="currentColor"/>'
};
const ACTS = {
  spray:{n:"Traiter mes cultures", z:10, kind:"speed", rule:k => k<3 ? ["warn","Attention : air trop calme"] : k<=15 ? ["go","Oui, vous pouvez traiter"] : k<=19 ? ["warn","Attention : le produit peut dériver"] : ["stop","Non, vent trop fort pour traiter"],
    todo:{go:"Traitez maintenant ; gardez une bande sans traitement près des voisins et des points d'eau.", warn:"Utilisez des buses anti-dérive et baissez la rampe, ou attendez le prochain créneau vert.", stop:"Reportez le traitement : le produit partirait chez les voisins."}},
  irr:{n:"Arroser par aspersion", z:10, kind:"speed", rule:k => k<=15 ? ["go","Oui, bon moment pour arroser"] : k<=25 ? ["warn","Attention : l'eau est emportée en partie"] : ["stop","Non, trop de vent pour l'aspersion"],
    todo:{go:"Arrosez maintenant : l'eau tombera là où il faut.", warn:"Préférez le goutte-à-goutte ou attendez le soir, quand le vent baisse.", stop:"Attendez : une grande partie de l'eau serait perdue."}},
  turbine:{n:"Produire avec une éolienne", z:80, kind:"power", rule:k => k<11 ? ["info","Production très faible : vent trop faible"] : k<90 ? ["go","Bonne production"] : ["stop","Arrêt de sécurité : vent trop fort"],
    todo:{go:"Bonne période pour produire ; reportez la maintenance.", info:"Bon moment pour la maintenance : l'éolienne produit peu.", stop:"L'éolienne se met en sécurité automatiquement."}}
};
// lieux : ceux du Grand Tunis, ou les stations du domaine servi (national : les 12 aéroports)
const PLACES = DOM ? (DOM.stations.length ? DOM.stations.map(s => [s.nom, s.lat, s.lon]) : [[`${ZONE} (centre)`, LAT0 + H/2/KY, LON0 + W/2/KX]]) : [["Tunis-Carthage (aéroport)",36.8504,10.2156],["Tunis centre",36.80,10.18],["La Marsa",36.878,10.325],["Ben Arous",36.75,10.23],["Hammam-Lif",36.728,10.34],["Mornag (vignes)",36.68,10.29],["Djebel Ressas",36.61,10.33],["Soliman",36.69,10.49],["Grombalia (agrumes)",36.60,10.50],["Korbous",36.82,10.57],["Nabeul",36.455,10.715],["Zaghouan (station)",36.4333,10.0833],["Mohamedia",36.68,10.16]];

const P0 = DOM ? (PLACES.find(p => /Tunis/.test(p[0])) || PLACES[0]) : null;
const state = { hour:15, act:"spray", pin: P0 ? xy(P0[1],P0[2]) : xy(36.68,10.29), pinName: P0 ? P0[0] : "Mornag (vignes)" };
function evalAt(actKey, x, y, h){
  const a = ACTS[actKey], [u,v] = wind(x,y,a.z,h), sp = Math.hypot(u,v), g = sp*gustF(x,y,a.z);
  const val = a.kind==="gust" ? kmh(g) : kmh(sp), [cls,title] = a.rule(val);
  return {u,v,sp,g,val,cls,title};
}

// ---------------- carte 2D animée ----------------
const MAPS=[];
function makeMap(canvasId, boxId, opts){
  const cv = document.getElementById(canvasId), box = document.getElementById(boxId), ctx = cv.getContext("2d");
  const base = document.createElement("canvas"), trail = document.createElement("canvas");
  let w=0, h=0, dpr=1, parts=[], hover=null; const ripples=[];
  function size(){ dpr = Math.min(2, window.devicePixelRatio||1); w = box.clientWidth; h = Math.round(w*H/W); if(!w) return; cv.width=base.width=trail.width=w*dpr; cv.height=base.height=trail.height=h*dpr; cv.style.height=h+"px"; drawBase(); parts = Array.from({length: Math.round(1300*w/700)}, () => newPart()); }
  const toPx = (x,y) => [x/W*w*dpr, (1-y/H)*h*dpr];
  function newPart(){ return {x:Math.random()*W, y:Math.random()*H, age:Math.random()*60|0}; }
  function drawBase(){
    const b = base.getContext("2d"), RW=272, RH=224, img=b.createImageData(RW,RH);
    const hex = s => { const m = s.match(/#(..)(..)(..)/); return m ? [1,2,3].map(i=>parseInt(m[i],16)) : [128,128,128]; };
    const lo=hex(css("--land-lo")), hi=hex(css("--land-hi")), wa=hex(css("--water")), la=hex(css("--lake")), ur=hex(css("--urban"));
    for(let j=0;j<RH;j++) for(let i=0;i<RW;i++){ const x=(i+.5)/RW*W, y=(1-(j+.5)/RH)*H, k=idx(x,y); let c;
      if(wat[k]===1) c=wa; else if(wat[k]===2) c=la; else { const t=Math.min(1,elev[k]/700); c=lo.map((q,n)=>q+(hi[n]-q)*Math.sqrt(t)); if(z0[k]>.7) c=c.map((q,n)=>q*.5+ur[n]*.5);
        const kE=idx(x+.5,y), kN=idx(x,y+.5), sh = Math.max(-.22,Math.min(.22,((elev[kE]-elev[k]) + (elev[kN]-elev[k]))*.0011)); c=c.map(q=>Math.min(255,q*(1-sh))); }
      if(opts.rel){ const s=sigma(x,y).s, t=Math.min(1,Math.max(0,(s-.45)/1.6)); const col = t<.3 ? [31,157,85] : t<.6 ? [217,138,0] : [210,58,58]; c=c.map((q,n)=>q*(.55+.2*t)+col[n]*(.45-.2*t)); if(t>.6) c=c.map(q=>q+(255-q)*.35*(t-.6)/.4); }
      const o=(j*RW+i)*4; img.data[o]=c[0]; img.data[o+1]=c[1]; img.data[o+2]=c[2]; img.data[o+3]=255; }
    const tmp=document.createElement("canvas"); tmp.width=RW; tmp.height=RH; tmp.getContext("2d").putImageData(img,0,0);
    b.imageSmoothingEnabled=true; b.drawImage(tmp,0,0,base.width,base.height);
  }
  function overlay(){
    ctx.save(); ctx.font = `700 ${12*dpr}px ${css("--f-body")}`; ctx.textAlign="center";
    for(const s of STATIONS){ const [px,py]=toPx(...s.p); ctx.fillStyle=css("--info"); ctx.beginPath(); ctx.arc(px,py,6*dpr,0,7); ctx.fill(); ctx.strokeStyle="#fff"; ctx.lineWidth=2*dpr; ctx.stroke(); }
    if(opts.rel && relMode==="aneo") for(const p of ANEO){ const [px,py]=toPx(...p); ctx.fillStyle=css("--sand"); ctx.beginPath(); ctx.arc(px,py,4*dpr,0,7); ctx.fill(); }
    if(opts.marks) for(const m of opts.marks()){ const [px,py]=toPx(...m.p), r=(m.r||2.2)/W*w*dpr, col=css(m.cls==="stop"?"--stop":m.cls==="warn"?"--warn":m.cls==="go"?"--go":"--info"), puls=m.sel?1+.18*Math.sin(performance.now()/220):1;
      ctx.globalAlpha=.25; ctx.fillStyle=col; ctx.beginPath(); ctx.arc(px,py,r*puls,0,7); ctx.fill(); ctx.globalAlpha=1; ctx.strokeStyle=col; ctx.lineWidth=(m.sel?4:2.5)*dpr; ctx.beginPath(); ctx.arc(px,py,r*puls,0,7); ctx.stroke();
      if(m.label){ ctx.fillStyle=css("--ink"); ctx.fillText(m.label,px,py-r-6*dpr); } }
    const pinP = typeof opts.pin==="function" ? opts.pin() : opts.pin ? state.pin : null;
    if(pinP){ const [px,py]=toPx(...pinP), t=performance.now()/600; ctx.strokeStyle=css("--accent"); ctx.lineWidth=2*dpr; ctx.globalAlpha=.6; ctx.beginPath(); ctx.arc(px,py,(10+6*(t%1))*dpr,0,7); ctx.stroke(); ctx.globalAlpha=1;
      ctx.fillStyle=css("--accent"); ctx.beginPath(); ctx.moveTo(px,py); ctx.arc(px,py-16*dpr,8*dpr,Math.PI*.8,Math.PI*.2,false); ctx.closePath(); ctx.fill(); ctx.fillStyle="#fff"; ctx.beginPath(); ctx.arc(px,py-16*dpr,3.2*dpr,0,7); ctx.fill(); }
    if(hover){ const [px,py]=toPx(...hover); ctx.strokeStyle=css("--ink"); ctx.globalAlpha=.55; ctx.lineWidth=1*dpr; ctx.setLineDash([4*dpr,4*dpr]); ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,cv.height); ctx.moveTo(0,py); ctx.lineTo(cv.width,py); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha=1; ctx.lineWidth=2.5*dpr; ctx.strokeStyle="#fff"; ctx.beginPath(); ctx.arc(px,py,9*dpr,0,7); ctx.stroke(); ctx.strokeStyle=css("--accent"); ctx.lineWidth=1.5*dpr; ctx.beginPath(); ctx.arc(px,py,9*dpr,0,7); ctx.stroke(); }
    const now=performance.now(); for(let i=ripples.length-1;i>=0;i--){ const q=ripples[i], a=(now-q.t)/700; if(a>1){ ripples.splice(i,1); continue; } const [px,py]=toPx(...q.p); ctx.globalAlpha=1-a; ctx.strokeStyle=css("--accent"); ctx.lineWidth=3*dpr; ctx.beginPath(); ctx.arc(px,py,(6+38*a)*dpr,0,7); ctx.stroke(); ctx.globalAlpha=1; }
    ctx.restore();
  }
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let frameId=0,lastFrame=performance.now();
  function schedule(){if(!frameId&&!document.hidden&&box.offsetParent)frameId=requestAnimationFrame(frame);}
  function frame(now=performance.now()){
    frameId=0;
    if(!w||!box.offsetParent||document.hidden)return;
    const elapsed=Math.min(2,(now-lastFrame)/(1000/60));lastFrame=now;
    const tc = trail.getContext("2d"), z = opts.z();
    tc.globalCompositeOperation="destination-out"; tc.fillStyle="rgba(0,0,0,.07)"; tc.fillRect(0,0,trail.width,trail.height); tc.globalCompositeOperation="source-over"; tc.lineWidth=2.2*dpr; tc.lineCap="round";
    if(!reduce) for(const p of parts){ const [u,v]=wind(p.x,p.y,z,state.hour), sp=Math.hypot(u,v), k=kmh(sp), dt=.055*elapsed;
      const nx=p.x+u*dt, ny=p.y+v*dt, [x1,y1]=toPx(p.x,p.y), [x2,y2]=toPx(nx,ny), c=speedColor(k); tc.strokeStyle=`rgb(${c[0]},${c[1]},${c[2]})`; tc.beginPath(); tc.moveTo(x1,y1); tc.lineTo(x2,y2); tc.stroke();
      p.x=nx; p.y=ny; if(++p.age>80 || p.x<0||p.x>W||p.y<0||p.y>H) Object.assign(p,newPart(),{age:0}); }
    else { tc.clearRect(0,0,trail.width,trail.height); for(let y=2;y<H;y+=4) for(let x=2;x<W;x+=4){ const [u,v]=wind(x,y,z,state.hour), c=speedColor(kmh(Math.hypot(u,v))), [x1,y1]=toPx(x,y), [x2,y2]=toPx(x+u*.5,y+v*.5); tc.strokeStyle=`rgb(${c})`; tc.lineWidth=2*dpr; tc.beginPath(); tc.moveTo(x1,y1); tc.lineTo(x2,y2); tc.stroke(); } }
    ctx.clearRect(0,0,cv.width,cv.height); ctx.drawImage(base,0,0); ctx.drawImage(trail,0,0); overlay();
    if(!reduce)schedule();
  }
  box.addEventListener("click", e => { const r=cv.getBoundingClientRect(), x=(e.clientX-r.left)/r.width*W, y=(1-(e.clientY-r.top)/r.height)*H; ripples.push({p:[x,y], t:performance.now()}); opts.onClick && opts.onClick(x,y); });
  box.addEventListener("mousemove", e => { const r=cv.getBoundingClientRect(), x=(e.clientX-r.left)/r.width*W, y=(1-(e.clientY-r.top)/r.height)*H; if(x<0||x>W||y<0||y>H) return; hover=[x,y];
    const z=opts.z(), k=idx(x,y), [u,v]=wind(x,y,z,state.hour), sp=Math.hypot(u,v), [rc,rl]=relClass(sigma(x,y).s), deg=(fromDeg(u,v)+180)%360;
    showTip(`<div style="display:flex;gap:8px;align-items:center"><b>${fmtSpeed(sp)}</b>${arrowSvg(deg,18,"currentColor")}</div>Vent ${dirName(u,v)}, à ${z} m du sol<br><span class="sub">${wat[k]===1?"En mer":wat[k]===2?"Lac ou sebkha":`Sol à ${Math.round(elev[k])} m`} · fiabilité ${rl} · près de ${nearestName([x,y])}</span>`, e.clientX, e.clientY, "map"); });
  box.addEventListener("mouseleave", () => { hover=null; hideTip("map"); });
  const observer=new ResizeObserver(()=>{size();schedule();});observer.observe(box);
  const visibility=new IntersectionObserver(entries=>{if(entries[0].isIntersecting){lastFrame=performance.now();schedule();}else{cancelAnimationFrame(frameId);frameId=0;}});visibility.observe(box);
  box.addEventListener("mousemove",schedule);box.addEventListener("click",schedule);box.addEventListener("mouseleave",schedule);
  addEventListener("visibilitychange",schedule);addEventListener("atom:state",schedule);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change",()=>{drawBase();schedule();});
  cv.tabIndex=0;cv.setAttribute("role","img");cv.setAttribute("aria-label","Carte du vent. Flèches pour déplacer la sélection, Entrée pour inspecter. Le lieu peut aussi être choisi dans la liste.");
  cv.addEventListener("keydown",e=>{if(!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","Enter"].includes(e.key))return;e.preventDefault();const p=state.pin.slice();if(e.key==="ArrowLeft")p[0]--;if(e.key==="ArrowRight")p[0]++;if(e.key==="ArrowUp")p[1]++;if(e.key==="ArrowDown")p[1]--;p[0]=Math.max(0,Math.min(W,p[0]));p[1]=Math.max(0,Math.min(H,p[1]));opts.onClick?.(...p);schedule();});
  addEventListener("pagehide",()=>{cancelAnimationFrame(frameId);observer.disconnect();visibility.disconnect();},{once:true});
  schedule();
  const api = { redraw: ()=>{drawBase();schedule();} }; MAPS.push(api); return api;
}

// ---------------- MON VENT ----------------
const townSel = document.getElementById("town");
townSel.innerHTML = `<option value="">Choisir ma ville ou mon lieu…</option>` + PLACES.map((p,i)=>`<option value="${i}">${p[0]}</option>`).join("") + `<option value="pin" hidden>Ma position sur la carte</option>`;
townSel.value = "5";
townSel.addEventListener("change", () => { if(townSel.value===""||townSel.value==="pin") return; const p=PLACES[+townSel.value]; state.pin=xy(p[1],p[2]); state.pinName=p[0]; renderAnswer(); if(typeof syncLeafletPin==="function") syncLeafletPin(true); });
const map1 = makeMap("c1","map1",{ pin:true, z:()=>ACTS[state.act].z, onClick:(x,y) => placePin(x,y) });
function placePin(x,y){ const k=idx(x,y); state.pin=[x,y]; let near=PLACES.map(p=>[p[0],Math.hypot(x-xy(p[1],p[2])[0],y-xy(p[1],p[2])[1])]).sort((a,b)=>a[1]-b[1])[0];
  state.pinName = wat[k]===1 ? `En mer, à ${Math.round(near[1])} km de ${near[0].split(" (")[0]}` : near[1]<2 ? near[0] : `À ${Math.round(near[1])} km de ${near[0].split(" (")[0]}`; townSel.value="pin"; townSel.querySelector('[value="pin"]').hidden=false; document.getElementById("hint1").textContent="Position placée. Touchez ailleurs pour la déplacer."; renderAnswer(); if(typeof syncLeafletPin==="function") syncLeafletPin(); }
document.getElementById("acts").innerHTML = Object.entries(ACTS).map(([k,a]) => `<button class="act" data-a="${k}" aria-pressed="${k===state.act}"><svg width="28" height="28" viewBox="0 0 24 24" aria-hidden="true">${ICONS[k]}</svg>${a.n}</button>`).join("");
document.getElementById("acts").addEventListener("click", e => { const b=e.target.closest(".act"); if(!b) return; state.act=b.dataset.a; document.querySelectorAll(".act").forEach(x=>x.setAttribute("aria-pressed", x===b)); renderAnswer(); });

const VERD_ICON = {go:'<path d="M5 12l5 5 9-10" stroke="#fff" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>', warn:'<path d="M12 6v8M12 17.5v.5" stroke="#fff" stroke-width="3" stroke-linecap="round"/>', stop:'<path d="M7 7l10 10M17 7L7 17" stroke="#fff" stroke-width="3" stroke-linecap="round"/>', info:'<path d="M12 11v7M12 6.5v.5" stroke="#fff" stroke-width="3" stroke-linecap="round"/>'};
function power(ms){ if(ms<3||ms>=25) return 0; if(ms>=12) return 2; return 2*(ms**3-27)/(1728-27); }
let whyOpen=false;
function renderAnswer(){
  const a=ACTS[state.act], [x,y]=state.pin, r=evalAt(state.act,x,y,state.hour), rel=sigma(x,y), [rc,rl]=relClass(rel.s), deg=(fromDeg(r.u,r.v)+180)%360;
  const extra = a.kind==="power" ? `<div class="fact"><div class="k">Production d'une éolienne de 2 MW</div><div class="v">${(power(r.sp)*1000).toFixed(0)} kW</div></div>` : `<div class="fact"><div class="k">Rafales à ${a.z} m</div><div class="v">${fmtSpeed(r.g)}</div></div>`;
  document.getElementById("answer").innerHTML = `<div class="answer ${r.cls}" role="status" aria-live="polite">
    <div class="verdict"><div class="badge-big"><svg width="28" height="28" viewBox="0 0 24 24">${VERD_ICON[r.cls]}</svg></div><div><div class="muted" style="font-size:14px;font-weight:700">${state.pinName} · ${state.hour} h · ${a.n.toLowerCase()}</div><h2>${r.title}</h2></div></div>
    <p>${a.todo[r.cls]||a.todo.go}</p>
    <div class="facts">
      <div class="fact"><div class="k">Vent à ${a.z} m du sol</div><div class="v">${fmtSpeed(r.sp)}</div></div>
      ${extra}
      <div class="fact"><div class="k">Il vient</div><div class="v" style="font-size:17px">${arrowSvg(deg,22,css("--accent"))}${dirName(r.u,r.v)}</div></div>
      <div class="fact"><div class="k">Fiabilité <span class="tip" tabindex="0" data-tip="Valeur calculée par ATOM Wind à partir de ${STATIONS.length} stations et de la météo de grande échelle. La station la plus proche (${rel.who}) est à ${Math.round(rel.d)} km. Erreur probable : ± ${Math.round(kmh(rel.s))} km/h.">?</span></div><div class="v" style="font-size:17px"><span class="pill ${rc}">${rl}</span></div></div>
    </div>
    <p class="muted" style="font-size:15px">${beaufort(kmh(r.sp))}</p>
    <button class="btn whybtn" id="whybtn" aria-expanded="${whyOpen}">Pourquoi ce vent ?</button>
    <div id="whybox" ${whyOpen?"":"hidden"}>${explainHtml(x,y,a.z,state.hour)}</div>
  </div>`;
  document.getElementById("whybtn").onclick=()=>{ whyOpen=!whyOpen; renderAnswer(); };
  updateSound();
  renderDay();
  if(window.ATOM?.workspace) publishWorkspace();
}
function renderDay(){
  const [x,y]=state.pin, a=ACTS[state.act], rows=[];
  for(let h=0;h<24;h++) rows.push(evalAt(state.act,x,y,h));
  const mx = Math.max(...rows.map(r=>r.val), 20);
  document.getElementById("day1").innerHTML = rows.map((r,h)=>`<button class="${r.cls}${h===state.hour?" now":""}" style="height:${Math.round(18+70*r.val/mx)}px" aria-label="${h} h : ${r.title}, ${Math.round(r.val)} km/h" data-h="<b>${h} h</b><br>${r.title}<br><span class=&quot;sub&quot;>${fmtSpeed(r.val/3.6)}</span>">​</button>`).join("");
  document.getElementById("axis1").innerHTML = Array.from({length:24},(_,h)=>`<span>${h%3===0?h:""}</span>`).join("");
  document.querySelectorAll("#day1 button").forEach((b,h)=>b.onclick=()=>setHour(h));
  const good=[]; let s=null; rows.forEach((r,h)=>{ if(r.cls==="go"){ if(s===null) s=h; } else if(s!==null){ good.push([s,h]); s=null; } }); if(s!==null) good.push([s,24]);
  document.getElementById("win1").innerHTML = good.length ? `<span style="font-weight:700">Meilleurs créneaux :</span>` + good.map(([a,b])=>`<span class="pill good">${a} h – ${b} h</span>`).join("") : `<span class="pill bad">Aucun créneau favorable aujourd'hui</span>`;
}

// ---------------- heure, lecture, unités ----------------
const hourIn=document.getElementById("hour"), hourTxt=document.getElementById("hourtxt");
function setHour(h){ state.hour=h; hourIn.value=h; hourTxt.textContent=`${h} h`; const sun=document.getElementById("sun"); if(sun){ sun.style.left=`${h/23*100}%`; sun.className=`sun ${h>=6&&h<=19?"day":"night"}`; } renderAnswer(); renderAlerts(); renderEol(); renderTB(); renderMyT(); renderAgri(); if(lastCut) renderCut(...lastCut); animateKpis("kpis"); animateKpis("tbkpis"); }
hourIn.addEventListener("input", () => setHour(+hourIn.value));
let playT=null; document.getElementById("play").onclick = e => { if(playT){ clearInterval(playT); playT=null; e.target.textContent="Rejouer la journée"; return; } e.target.textContent="Pause"; let h=0; setHour(0); playT=setInterval(()=>{ h=(h+1)%24; setHour(h); if(h===23){ clearInterval(playT); playT=null; e.target.textContent="Rejouer la journée"; } }, 700); };
function setUnit(u){ unit=u; document.getElementById("u-kmh").setAttribute("aria-pressed",u==="kmh"); document.getElementById("u-ms").setAttribute("aria-pressed",u==="ms"); renderAnswer(); renderAlerts(); renderEol(); renderTB(); renderMyT(); }
document.getElementById("u-kmh").onclick=()=>setUnit("kmh"); document.getElementById("u-ms").onclick=()=>setUnit("ms");

// ---------------- ALERTES ----------------
let ALERTS=[], selAlert=-1, MY=[];
try{ MY=JSON.parse(localStorage.getItem("aw-alertes")||"[]"); if(!Array.isArray(MY)) MY=[]; }catch(e){ MY=[]; }
const saveMy=()=>{ try{ localStorage.setItem("aw-alertes",JSON.stringify(MY)); }catch(e){} };
const LV={stop:"Danger",warn:"Attention",info:"Information",go:"Tout va bien"}, PILL={stop:"bad",warn:"mid",info:"m",go:"good"};
function buildAlerts(){
  const L=[];
  const mp=xy(36.68,10.29); let last=-1; for(let h=6;h<20;h++) if(evalAt("spray",...mp,h).cls==="go") last=h;
  const sprayNow = evalAt("spray",...mp,state.hour);
  L.push({cls: sprayNow.cls==="stop"?"warn":"info", p:mp, at:last+1, t:"Créneau de traitement des vignes", where:"Mornag", when: last>0 ? `favorable jusqu'à ${last+1} h environ` : "pas de créneau favorable aujourd'hui", what:"Au-delà, le vent dépasse 15 km/h et le produit peut dériver.", todo:"Traiter tôt le matin ; reprendre en fin de soirée si le vent retombe."});
  { const hub=Array.from({length:24},(_,h)=>TURB.reduce((m,t)=>Math.max(m,Math.hypot(...wind(...t.p,80,h))),0));
    const cut=hub.map((v,h)=>v>=25?h:-1).filter(h=>h>=0), calm=hub.map((v,h)=>h>=7&&h<=18&&v<4?h:-1).filter(h=>h>=0);
    if(cut.length) L.push({cls:"stop", p:TURB[1].p, r:2, at:cut[0], t:"Arrêt de sécurité des éoliennes", where:"Parc virtuel du djebel Ressas", when:`à partir de ${cut[0]} h (vent au moyeu au-dessus de 90 km/h)`, what:"Au-delà de 25 m/s au moyeu, les éoliennes se mettent en drapeau.", todo:"Prévoir la perte de production et l'inspection après l'épisode."});
    else if(calm.length) L.push({cls:"info", p:TURB[1].p, r:2, at:calm[0], t:"Créneau de maintenance du parc", where:"Parc virtuel du djebel Ressas", when:`de ${calm[0]} h à ${calm[calm.length-1]+1} h (vent faible au moyeu)`, what:"La production est presque nulle : c'est le moment d'intervenir sans perte.", todo:"Planifier les interventions sur ce créneau."});
    else { const pk=hub.indexOf(Math.max(...hub)); L.push({cls:"go", p:TURB[1].p, r:2, at:pk, t:"Bonne journée de production", where:"Parc virtuel du djebel Ressas", when:`pic vers ${pk} h (${fmtSpeed(hub[pk])} au moyeu)`, what:"Le vent au moyeu reste dans la plage de production.", todo:"Reporter la maintenance à un jour plus calme."}); } }
  L.push({cls:"info", p:STATIONS[2].p, t:"Station de Nabeul : vent mesuré plus faible que la zone", where:"Nabeul (station de l'INM)", when:"toute la journée", what:"L'anémomètre est abrité en ville ; il mesure moins que le vent réel alentour.", todo:"Aucune action : ATOM Wind tient compte de cet écart dans la fiabilité."});
  for(const m of MY){ const pl=PLACES[m.place], p=xy(pl[1],pl[2]), a=ACTS[m.act]; let hit=null;
    for(let k=0;k<24;k++){ const h=(state.hour+k)%24, r=evalAt(m.act,...p,h); if(r.cls==="stop"||r.cls==="warn"){ hit={h,r}; break; } }
    m.res = hit ? `se déclenche à ${hit.h} h (${LV[hit.r.cls].toLowerCase()})` : "rien à signaler dans les 24 h";
    if(hit) L.push({cls:hit.r.cls, p, at:hit.h, mine:true, t:`Votre alerte : ${a.n.toLowerCase()}`, where:pl[0], when:`à ${hit.h} h : ${hit.r.title.toLowerCase()} (${fmtSpeed(a.kind==="gust"?hit.r.g:hit.r.sp)})`, what:`Alerte créée par vous, envoi par ${m.chan}.`, todo:a.todo[hit.r.cls]||a.todo.go}); }
  if(window.ATOM && ATOM.alertesService) L.push(...ATOM.alertesService(xy));
  const ord={stop:0,warn:1,info:2,go:3}; L.sort((a,b)=>ord[a.cls]-ord[b.cls]);
  return L;
}
function countdown(a){ if(a.at===undefined || a.at<0) return ""; const d=a.at-state.hour, col=css(a.cls==="stop"?"--stop":a.cls==="warn"?"--warn":"--info");
  const txt = d>0 ? `Dans ${d} h` : d===0 ? "Maintenant" : "Passé"; const pct = d<=0 ? 100 : Math.max(5, 100-d/12*100);
  return `<div class="cdown" style="color:${col}"><span>${txt}</span><div class="bar"><i style="width:${pct}%;background:${col}"></i></div></div>`; }
function renderAlerts(){
  ALERTS = buildAlerts(); if(selAlert>=ALERTS.length) selAlert=-1;
  document.getElementById("alertlist").innerHTML = ALERTS.map((a,i)=>`<article class="alert ${a.cls}${i===selAlert?" sel":""}" data-i="${i}" tabindex="0"><div class="ic"><svg width="24" height="24" viewBox="0 0 24 24">${VERD_ICON[a.cls]}</svg></div><div><span class="pill ${PILL[a.cls]}">${LV[a.cls]}</span>${a.mine?' <span class="pill m">Mon alerte</span>':''}<h3 style="margin-top:6px">${a.t}</h3><dl><dt>Où</dt><dd>${a.where}</dd><dt>Quand</dt><dd>${a.when}</dd><dt>Pourquoi</dt><dd>${a.what}</dd></dl><div class="todo">Que faire : ${a.todo}</div>${a.cls!=="info"?countdown(a):""}</div></article>`).join("");
  document.querySelectorAll("#alertlist .alert").forEach(el=>{ const f=()=>{ selAlert=+el.dataset.i; renderAlerts(); }; el.onclick=f; el.onkeydown=e=>{ if(e.key==="Enter") f(); }; });
  const n=ALERTS.filter(a=>a.cls==="stop"||a.cls==="warn").length; const ac=document.getElementById("alertcount"); ac.textContent=n; ac.hidden=!n;
  document.getElementById("al-mine").innerHTML = MY.length ? MY.map((m,i)=>`<div class="it"><span style="min-width:0"><b>${ACTS[m.act].n}</b> · ${PLACES[m.place][0]}<br><span class="muted" style="font-size:14px">${m.res||""} · par ${m.chan}</span></span><button class="x sm" data-del="${i}" aria-label="Supprimer cette alerte">×</button></div>`).join("") : `<span class="muted">Aucune alerte personnelle pour l'instant.</span>`;
  document.querySelectorAll("[data-del]").forEach(b=>b.onclick=()=>{ MY.splice(+b.dataset.del,1); saveMy(); renderAlerts(); renderTB(); });
}
// ---------------- ÉOLIEN ----------------
const TURB = [["E1",36.618,10.315],["E2",36.612,10.33],["E3",36.605,10.345]].map(([n,a,b])=>({n,p:xy(a,b)}));
const eo={day:0}; const EODAYS=["aujourd'hui","demain","après-demain"];
const windE=(x,y,z,h)=>windD(x,y,z,h,eo.day);
function renderEol(){
  const now = TURB.map(t=>{ const [u,v]=windE(...t.p,80,state.hour), sp=Math.hypot(u,v); return {...t,u,v,sp,P:power(sp)}; });
  const tot = now.reduce((s,t)=>s+t.P,0), day=Array.from({length:24},(_,h)=>TURB.reduce((s,t)=>s+power(Math.hypot(...windE(...t.p,80,h))),0)), energy=day.reduce((a,b)=>a+b,0);
  const avg = now.reduce((s,t)=>s+t.sp,0)/3;
  document.getElementById("kpis").innerHTML = [
    [`Production à ${state.hour} h ${EODAYS[eo.day]}`, `${tot.toFixed(1).replace(".",",")} MW`, `${Math.round(tot/6*100)} % de la capacité`],
    ["Vent au moyeu (80 m)", fmtSpeed(avg), `vient ${dirName(now[0].u,now[0].v)}`],
    [`Énergie prévue ${EODAYS[eo.day]}`, `${energy.toFixed(0)} MWh`, `≈ ${Math.round(energy*1000/8)} foyers pour une journée`],
    ["État du parc", "3 / 3 en service", "Aucune alerte d'arrêt"]].map(([k,v,s])=>`<div class="card kpi"><div class="k">${k}</div><div class="v">${v}</div><div class="s muted">${s}</div></div>`).join("");
  const Wd=640,Hd=200,pl=36,pb=24, mx=6, X=h=>pl+h*(Wd-pl-8)/23, Y=p=>Hd-pb-p/mx*(Hd-pb-10);
  const path = day.map((p,h)=>`${h?"L":"M"}${X(h).toFixed(1)},${Y(p).toFixed(1)}`).join("");
  document.getElementById("prodchart").innerHTML = `<div class="production-unit">Puissance (MW)</div><div class="tblwrap"><svg viewBox="0 0 ${Wd} ${Hd}" style="width:100%;min-width:320px;height:auto" role="img" aria-label="Production du parc heure par heure">
    ${[0,2,4,6].map(p=>`<line x1="${pl}" x2="${Wd-8}" y1="${Y(p)}" y2="${Y(p)}" stroke="var(--line)"/><text x="${pl-6}" y="${Y(p)+4}" text-anchor="end">${p}</text>`).join("")}
    ${[0,6,12,18,23].map(h=>`<text x="${X(h)}" y="${Hd-6}" text-anchor="${h===0?"start":h===23?"end":"middle"}">${h} h</text>`).join("")}
    <path d="${path}L${X(23)},${Y(0)}L${X(0)},${Y(0)}Z" fill="var(--accent)" opacity=".18"/><path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2.5"/>
    <line x1="${X(state.hour)}" x2="${X(state.hour)}" y1="10" y2="${Hd-pb}" stroke="var(--sand)" stroke-width="2" stroke-dasharray="4 3"/><circle cx="${X(state.hour)}" cy="${Y(day[state.hour])}" r="5" fill="var(--sand)"/><line class="hl" y1="10" y2="${Hd-pb}" stroke="var(--ink)" stroke-opacity=".35" visibility="hidden"/></svg></div>`;
  hoverChart("prodchart", Wd, pl, 8, h=>`<b>${h} h</b><br>Production du parc : ${day[h].toFixed(1).replace(".",",")} MW<br><span class="sub">Cliquez pour aller à cette heure</span>`);
  document.getElementById("turbs").innerHTML = now.map(t=>{ const dur = t.P>0 ? Math.max(.6, 4.5-t.sp/4) : 0; return `<div class="card" style="display:grid;grid-template-columns:auto minmax(0,1fr);gap:12px;align-items:center">
    <svg width="70" height="90" viewBox="0 0 70 90" aria-hidden="true"><path d="M33 34h4l2 54h-8z" fill="var(--muted)"/><g class="rotor" style="animation-duration:${dur||0}s;${dur?"":"animation:none"}"><path d="M35 32L35 4M35 32L59 46M35 32L11 46" stroke="var(--ink)" stroke-width="4" stroke-linecap="round"/><circle cx="35" cy="32" r="4" fill="var(--accent)"/></g></svg>
    <div><h3>Éolienne ${t.n}</h3><div><b>${(t.P*1000).toFixed(0)} kW</b> <span class="muted">sur 2 000</span></div><div class="muted" style="font-size:14px">Vent au moyeu ${fmtSpeed(t.sp)}, ${dirName(t.u,t.v)}</div><span class="pill ${t.P>0?"good":"mid"}">${t.P>0?"En production":"À l'arrêt, vent trop faible"}</span></div></div>`; }).join("");
}

// ---------------- alertes : formulaire et carte ----------------
document.getElementById("al-place").innerHTML = PLACES.map((p,i)=>`<option value="${i}">${p[0]}</option>`).join("");
document.getElementById("al-act").innerHTML = Object.entries(ACTS).map(([k,a])=>`<option value="${k}">${a.n}</option>`).join("");
document.getElementById("al-add").onclick = () => { MY.push({place:+document.getElementById("al-place").value, act:document.getElementById("al-act").value, chan:document.getElementById("al-chan").value}); saveMy(); renderAlerts(); renderTB(); };
makeMap("c3","map3",{ z:()=>10, marks:()=>ALERTS.map((a,i)=>({p:a.p, r:a.r, cls:a.cls, sel:i===selAlert})), onClick:()=>{} });

// ---------------- tableau de bord ----------------
const DSTAT = {};
function domainStats(h){ if(DSTAT[h]) return DSTAT[h]; let n=0, su=0, sv=0, ss=0, gmax=0, gwhere=null;
  for(let y=1;y<H;y+=2) for(let x=1;x<W;x+=2){ if(wat[idx(x,y)]) continue; const [u,v]=wind(x,y,10,h), sp=Math.hypot(u,v), g=sp*gustF(x,y,10); n++; su+=u; sv+=v; ss+=sp; if(g>gmax){ gmax=g; gwhere=[x,y]; } }
  return DSTAT[h]={mean:ss/n, u:su/n, v:sv/n, gmax, gwhere}; }
function nearestName(p){ return PLACES.map(q=>[q[0],Math.hypot(p[0]-xy(q[1],q[2])[0],p[1]-xy(q[1],q[2])[1])]).sort((a,b)=>a[1]-b[1])[0][0].split(" (")[0]; }
let relGood=null;
function renderTB(){
  const d=domainStats(state.hour), prod=TURB.reduce((s,t)=>s+power(Math.hypot(...wind(...t.p,80,state.hour))),0), na=ALERTS.filter(a=>a.cls==="stop"||a.cls==="warn").length;
  if(!relGood){ const keep=relMode; relMode="reel"; const a=relStats(); relMode="aneo"; const b=relStats(); relMode=keep; relGood=[a,b]; }
  const reading=fmtSpeed(d.mean).split(" ");
  document.getElementById("tbhero").innerHTML=`<div class="territory-average"><span>Moyenne du territoire</span><b>${reading[0]} <small>${reading.slice(1).join(" ")}</small></b></div>`;
  renderTerritory(document.getElementById("tbmap"),{
    water:[GULF,HAMM],lakes:LAKES,hills:HILLS,selected:state.pin,
    width:W,height:H,zone:DOM ? (DOM.national ? "la Tunisie" : DOM.nom) : "le Grand Tunis",seaLabel:DOM ? null : "Golfe de Tunis",
    sample:(x,y)=>wind(x,y,10,state.hour),
    places:PLACES.map(p=>{const point=xy(p[1],p[2]);return {name:p[0],point,speed:fmtSpeed(Math.hypot(...wind(...point,10,state.hour))),selected:Math.hypot(point[0]-state.pin[0],point[1]-state.pin[1])<.01};}),
    select:selectTerritoryPlace
  });
  const placeSelector=document.getElementById("tb-place");
  if(!placeSelector.options.length)placeSelector.innerHTML=PLACES.map((p,i)=>`<option value="${i}">${p[0]}</option>`).join("")+`<option value="pin" hidden>Point choisi sur la carte</option>`;
  const selectedPlace=PLACES.findIndex(p=>Math.hypot(xy(p[1],p[2])[0]-state.pin[0],xy(p[1],p[2])[1]-state.pin[1])<.01);
  placeSelector.value=selectedPlace<0?"pin":String(selectedPlace);
  placeSelector.onchange=()=>selectTerritoryPlace(+placeSelector.value);
  document.getElementById("tblocations").innerHTML=[1,2,5,6,10].filter(i=>PLACES[i]).map(i=>{const p=PLACES[i],point=xy(p[1],p[2]);return `<button data-territory-place="${i}" aria-pressed="${i===selectedPlace}"><span>${p[0].split(" (")[0]}</span><b>${fmtSpeed(Math.hypot(...wind(...point,10,state.hour)))}</b></button>`;}).join("");
  document.querySelectorAll("[data-territory-place]").forEach(button=>button.onclick=()=>{const index=button.dataset.territoryPlace,keyboard=button.matches(":focus-visible");selectTerritoryPlace(+index);if(keyboard)document.querySelector(`[data-territory-place="${index}"]`)?.focus();});
  const [pu,pv]=wind(...state.pin,10,state.hour),pointSpeed=Math.hypot(pu,pv),pointRel=sigma(...state.pin,"reel"),[quality,qualityLabel]=relClass(pointRel.s),pointReading=fmtSpeed(pointSpeed).split(" ");
  document.getElementById("tbpoint").innerHTML=`<div class="point-reading"><span>Vent à 10 m du sol · ${state.hour} h</span><strong>${pointReading[0]}<small> ${pointReading.slice(1).join(" ")}</small></strong><p>${arrowSvg(fromDeg(pu,pv),20)}Il vient ${dirName(pu,pv)}</p></div><dl class="point-facts"><div><dt>Rafales estimées</dt><dd>${fmtSpeed(pointSpeed*gustF(...state.pin,10))}</dd></div><div><dt>Fiabilité indicative</dt><dd><span class="point-quality ${quality}">${qualityLabel}</span></dd></div></dl>`;
  document.getElementById("tb-profile").onclick=()=>openSheet(...state.pin,10);
  document.getElementById("tbkpis").innerHTML = [
    ["Parc virtuel", `${prod.toFixed(1).replace(".",",")} MW`, `${Math.round(prod/6*100)} % de la capacité`],
    ["Points de vigilance", `${na}`, na ? "danger ou attention" : "aucune alerte importante"],
    ["Couverture indicative", `${relGood[0]} %`, `${relGood[1]} % avec le réseau prévu`],
    ...((window.ATOM && ATOM.kpisReels) ? ATOM.kpisReels(state.hour) : [])
  ].map(([k,v,note])=>{const parts=String(v).split(" "),number=parts.shift(),unit=parts.join(" ");return `<div class="overview-metric"><span class="metric-label">${k}</span><strong><span>${number}</span>${unit?` <small>${unit}</small>`:""}</strong><span class="metric-note">${note}</span></div>`;}).join("");
  document.getElementById("tb-source-note").textContent=window.ATOM?.pret()?"Calcul du modèle, avec interpolation entre les instants disponibles. Il ne s’agit pas de mesures à chaque point.":"Journée d’exemple. Les conditions actuelles ne sont pas affichées.";
  const days=Array.from({length:24},(_,h)=>domainStats(h));
  const Wd=760,Hd=310,pl=42,pb=33,factor=unit==="kmh"?3.6:1,chartUnit=unit==="kmh"?"km/h":"m/s";
  const mx=Math.ceil(Math.max(...days.map(x=>x.gmax*factor))/5)*5;
  const X=h=>pl+h*(Wd-pl-16)/23,Y=k=>Hd-pb-k/mx*(Hd-pb-24);
  const line=f=>days.map((x,h)=>`${h?"L":"M"}${X(h).toFixed(1)},${Y(f(x)).toFixed(1)}`).join("");
  const peak=days.reduce((best,d,i)=>d.gmax>days[best].gmax?i:best,0);
  document.getElementById("tbday").innerHTML=`<div class="day-fact"><span>Le pic de rafales</span><b>${fmtSpeed(days[peak].gmax)} vers ${peak} h</b></div><div class="tblwrap"><svg viewBox="0 0 ${Wd} ${Hd}" role="img" aria-label="Vent moyen et rafales maximales, en ${chartUnit}">
    <rect x="${X(state.hour)-10}" y="9" width="20" height="${Hd-pb-9}" rx="10" fill="var(--selected-hour)"/>
    ${[0,mx/2,mx].map(k=>`<path d="M${pl},${Y(k)}H${Wd-16}" stroke="var(--chart-grid)"/><text x="${pl-10}" y="${Y(k)+4}" text-anchor="end">${Math.round(k)}</text>`).join("")}
    ${days.map((d,h)=>`<path d="M${X(h)},${Y(d.mean*factor)}V${Y(d.gmax*factor)}" stroke="var(--gust-chart)" stroke-opacity=".65" stroke-width="1.5"/><circle cx="${X(h)}" cy="${Y(d.gmax*factor)}" r="2.2" fill="var(--gust-chart)"/>`).join("")}
    <path d="${line(d=>d.mean*factor)}L${X(23)},${Y(0)}L${X(0)},${Y(0)}Z" fill="var(--wind-chart)" opacity=".11"/>
    <path d="${line(d=>d.mean*factor)}" fill="none" stroke="var(--wind-chart)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
    ${days.map((d,h)=>`<circle cx="${X(h)}" cy="${Y(d.mean*factor)}" r="${h===state.hour?5:2.8}" fill="${h===state.hour?"var(--ink)":"var(--wind-chart)"}"/>`).join("")}
    ${[0,6,12,18,23].map(h=>`<text x="${X(h)}" y="${Hd-8}" text-anchor="${h===0?"start":h===23?"end":"middle"}">${h} h</text>`).join("")}
    <line class="hl" y1="8" y2="${Hd-pb}" stroke="var(--ink)" stroke-opacity=".35" visibility="hidden"/>
  </svg></div><div class="chart-key"><span class="mean-key">Vent moyen</span><span class="gust-key">Rafales maximales</span><span>${chartUnit}</span></div><p class="chart-instruction">Touchez une heure pour la lire.</p>`;
  if(window.ATOM && ATOM.rendreSerieMoyeu) ATOM.rendreSerieMoyeu(xy);
  hoverChart("tbday",Wd,pl,16,h=>`<b>${h} h</b><br>Vent moyen : ${fmtSpeed(days[h].mean)}<br>Rafale la plus forte : ${fmtSpeed(days[h].gmax)}`);
  document.getElementById("tbalerts").innerHTML = ALERTS.slice(0,3).map((a,i)=>`<a class="it" href="#alertes" data-tb-alert="${i}"><span style="min-width:0"><span class="pill ${PILL[a.cls]}">${LV[a.cls]}</span> <b>${esc(a.t)}</b><br><span class="muted">${esc(a.where)} · ${esc(a.when)}</span></span><span class="alert-row-arrow" aria-hidden="true">↗</span></a>`).join("") || `<span class="muted">Aucune alerte.</span>`;
  document.querySelectorAll("[data-tb-alert]").forEach(link=>link.onclick=()=>{selAlert=+link.dataset.tbAlert;renderAlerts();});
  const C=[["Anémomètre de Nabeul",2.42,2.57],["Anémomètre de Tunis-Carthage",2.70,2.81],["Anémomètre de Zaghouan",2.28,2.16],["Ballons-sondes, 0 à 250 m",2.22,2.29]];
  const bx=k=>Math.round(k/12*100);
  document.getElementById("tbcomp").innerHTML = C.map(([n,a,b])=>`<div style="display:grid;gap:3px;margin-bottom:10px"><b style="font-size:15px">${n}</b>
    <div style="display:grid;grid-template-columns:90px minmax(0,1fr) 86px;gap:8px;align-items:center;font-size:14px"><span>ATOM Wind</span><span style="height:12px;border-radius:6px;background:var(--accent);width:${bx(kmh(a))}%"></span><span style="font-family:var(--f-data)">${kmh(a).toFixed(1).replace(".",",")} km/h</span></div>
    <div style="display:grid;grid-template-columns:90px minmax(0,1fr) 86px;gap:8px;align-items:center;font-size:14px"><span class="muted">ERA5</span><span style="height:12px;border-radius:6px;background:var(--muted);opacity:.5;width:${bx(kmh(b))}%"></span><span style="font-family:var(--f-data)" class="muted">${kmh(b).toFixed(1).replace(".",",")} km/h</span></div></div>`).join("") + `<p class="muted" style="font-size:13px;margin:0">ATOM Wind fait mieux qu'ERA5 à Nabeul, à Tunis-Carthage et sur les ballons-sondes ; un peu moins bien à Zaghouan, en bord de zone.</p>`;
  document.getElementById("tbnet").innerHTML = [
    ["Stations de l'INM dans la zone","3","Tunis-Carthage, Zaghouan, Nabeul : vent à 10 m, toutes les 30 min à 3 h"],
    ["Mesure en altitude","1","Ballons-sondes de Tunis-Carthage, deux fois par jour"],
    ["Réseau ANEO prévu",`${ANEO.length}<small>capteurs</small>`,`Fiabilité bonne sur ${relGood[1]} % des terres, contre ${relGood[0]} % aujourd'hui`]].map(([k,v,s2])=>`<div class="kpi"><div class="k">${k}</div><div class="v" style="font-size:26px">${v}</div><div class="s muted">${s2}</div></div>`).join("");
}

function selectTerritoryPlace(index){
  if(!PLACES[index])return;
  townSel.value=String(index);townSel.dispatchEvent(new Event("change"));renderTB();
}
document.querySelector('[data-context="agri"]').addEventListener("click",()=>{
  const nearest=PLACES.map((p,i)=>({i,d:Math.hypot(xy(p[1],p[2])[0]-state.pin[0],xy(p[1],p[2])[1]-state.pin[1])})).sort((a,b)=>a.d-b.d)[0];
  ag.place=nearest.i;document.getElementById("ag-place").value=String(nearest.i);renderAgri();
});

// ---------------- éolien : mon éolienne, rotor, rose ----------------
const TMOD={s:{n:"petite éolienne",z:30,pr:.1,d:20},m:{n:"éolienne moyenne",z:50,pr:.8,d:50},l:{n:"grande éolienne",z:80,pr:2,d:90}};
const myT={p:xy(36.69,10.40), m:"l"};
const powerM=(ms,pr)=> ms<3||ms>=25 ? 0 : ms>=12 ? pr : pr*(ms**3-27)/(1728-27);
const dayEnergy=(p,m)=>{ let e=0; for(let h=0;h<24;h++) e+=powerM(Math.hypot(...windE(...p,TMOD[m].z,h)),TMOD[m].pr); return e; };
function renderMyT(){
  const t=TMOD[myT.m], e=dayEnergy(myT.p,myT.m), ref=dayEnergy(TURB[1].p,myT.m), cf=e/(t.pr*24), ratio=ref>0?e/ref:0, k=idx(...myT.p);
  const verdict = wat[k]===1 ? ["info","En mer : ce prototype estime surtout les éoliennes à terre."] : cf>.3 ? ["go","Très bon emplacement"] : cf>.18 ? ["warn","Emplacement moyen"] : ["stop","Emplacement peu venté"];
  document.getElementById("myt").innerHTML = `<div class="answer ${verdict[0]}" style="padding:14px"><div class="verdict"><div class="badge-big" style="width:42px;height:42px"><svg width="22" height="22" viewBox="0 0 24 24">${VERD_ICON[verdict[0]]}</svg></div><div><div class="muted" style="font-size:13px;font-weight:700">${t.n[0].toUpperCase()+t.n.slice(1)} près de ${nearestName(myT.p)}, sol à ${Math.round(elev[k])} m</div><h2 style="font-size:21px">${verdict[1]}</h2></div></div>
    <div class="facts"><div class="fact"><div class="k">Énergie sur la journée</div><div class="v">${e>=1?e.toFixed(1).replace(".",",")+" MWh":Math.round(e*1000)+" kWh"}</div></div><div class="fact"><div class="k">Taux de charge</div><div class="v">${Math.round(cf*100)} %</div></div><div class="fact"><div class="k">Par rapport à la crête du djebel Ressas</div><div class="v">${Math.round(ratio*100)} %</div></div></div>
    <p class="muted" style="font-size:14px">${ratio<.7?"Le vent est plus faible ici que sur la crête : une tour plus haute ou un site plus exposé produirait davantage.":"Cet emplacement produit presque autant que la crête, la meilleure zone de l'exemple."} Estimation pour ${EODAYS[eo.day]}.</p></div>`;
  // rotor
  const z0h=t.z, R=t.d/2, hs=[]; for(let i=0;i<=8;i++){ const z=Math.max(5,z0h-R+i*(2*R/8)); hs.push([z,Math.hypot(...windE(...myT.p,z,state.hour))]); }
  const top=hs[8][1], bot=hs[0][1], hub=Math.hypot(...windE(...myT.p,z0h,state.hour)), cx=110, cy=110, rp=82;
  const bands=hs.slice(0,8).map(([z,sp],i)=>{ const y=cy+rp-(i+1)*(2*rp/8), c=speedColor(kmh((sp+hs[i+1][1])/2)); return `<rect x="${cx-rp}" y="${y.toFixed(1)}" width="${2*rp}" height="${(2*rp/8+.5).toFixed(1)}" fill="rgb(${c})"/>`; }).join("");
  document.getElementById("rotor").innerHTML = `<div style="display:grid;grid-template-columns:auto minmax(0,1fr);gap:14px;align-items:center"><svg width="220" height="260" viewBox="0 0 220 260" aria-hidden="true"><defs><clipPath id="disk"><circle cx="${cx}" cy="${cy}" r="${rp}"/></clipPath></defs><path d="M106 ${cy}h8l5 148h-18z" fill="var(--muted)"/><g clip-path="url(#disk)">${bands}</g><circle cx="${cx}" cy="${cy}" r="${rp}" fill="none" stroke="var(--ink)" stroke-width="2"/><g class="rotor" style="animation-duration:${hub>3?Math.max(.8,5-hub/3):0}s"><path d="M${cx} ${cy}V${cy-rp+4}M${cx} ${cy}L${cx+71} ${cy+41}M${cx} ${cy}L${cx-71} ${cy+41}" stroke="var(--ink)" stroke-width="5" stroke-linecap="round" opacity=".75"/></g><circle cx="${cx}" cy="${cy}" r="6" fill="var(--accent)"/></svg>
    <div style="display:grid;gap:8px"><div><span class="muted">En haut des pales (${Math.round(z0h+R)} m)</span><br><b style="font-size:20px">${fmtSpeed(top)}</b></div><div><span class="muted">Au moyeu (${z0h} m)</span><br><b style="font-size:20px">${fmtSpeed(hub)}</b></div><div><span class="muted">En bas des pales (${Math.round(Math.max(5,z0h-R))} m)</span><br><b style="font-size:20px">${fmtSpeed(bot)}</b></div>
    <p style="margin:0">En haut des pales, le vent est <b>${Math.round((top/bot-1)*100)} %</b> plus fort qu'en bas. Cet écart fatigue les pales : seule une carte du vent en 3D permet de le connaître.</p></div></div>`;
  // rose
  const sec=Array.from({length:8},()=>({n:0,s:0})); for(let h=0;h<24;h++){ const [u,v]=windE(...myT.p,z0h,h), i=Math.round(fromDeg(u,v)/45)%8; sec[i].n++; sec[i].s+=Math.hypot(u,v); }
  const mxn=Math.max(...sec.map(x=>x.n)), c0=120, rr=95, NAMES=["N","NE","E","SE","S","SO","O","NO"];
  const wedges=sec.map((x,i)=>{ if(!x.n) return ""; const r=18+(rr-18)*x.n/mxn, a0=(i*45-20)*Math.PI/180, a1=(i*45+20)*Math.PI/180, c=speedColor(kmh(x.s/x.n));
    return `<path d="M${c0} ${c0} L${(c0+r*Math.sin(a0)).toFixed(1)} ${(c0-r*Math.cos(a0)).toFixed(1)} A${r.toFixed(1)} ${r.toFixed(1)} 0 0 1 ${(c0+r*Math.sin(a1)).toFixed(1)} ${(c0-r*Math.cos(a1)).toFixed(1)} Z" fill="rgb(${c})" stroke="var(--surface)" stroke-width="2" data-h="<b>Vent ${DIRS[i]}</b><br>${x.n} h sur 24, ${fmtSpeed(x.s/x.n)} en moyenne"></path>`; }).join("");
  const best=sec.map((x,i)=>[x.n,i]).sort((a,b)=>b[0]-a[0])[0][1];
  document.getElementById("rose").innerHTML = `<div style="display:grid;grid-template-columns:auto minmax(0,1fr);gap:14px;align-items:center"><svg width="240" height="240" viewBox="0 0 240 240" role="img" aria-label="Rose des vents"><circle cx="${c0}" cy="${c0}" r="${rr}" fill="none" stroke="var(--line)"/><circle cx="${c0}" cy="${c0}" r="${rr/2}" fill="none" stroke="var(--line)"/>${wedges}${NAMES.map((n,i)=>`<text x="${c0+(rr+14)*Math.sin(i*Math.PI/4)}" y="${c0-(rr+14)*Math.cos(i*Math.PI/4)+4}" text-anchor="middle" style="font-weight:700">${n}</text>`).join("")}</svg>
    <p style="margin:0">${EODAYS[eo.day][0].toUpperCase()+EODAYS[eo.day].slice(1)}, au moyeu de votre éolienne, le vent vient surtout <b>${DIRS[best]}</b> (${sec[best].n} heures sur 24). La longueur indique la durée, la couleur la force. Orientez les rangées d'éoliennes face à ce vent.</p></div>`;
}
document.getElementById("tm").onchange=e=>{ myT.m=e.target.value; renderMyT(); renderGis(); };
// ---- carte du gisement
const gisCache={};
function gisGrid(m,d){ const key=m+"|"+d; if(gisCache[key]) return gisCache[key]; const t=TMOD[m], g=new Float32Array(W*H);
  for(let j=0;j<H;j++) for(let i=0;i<W;i++){ const x=i+.5, y=j+.5; if(wat[idx(x,y)]){ g[j*W+i]=-1; continue; } let e=0; for(let h=0;h<24;h+=2) e+=powerM(Math.hypot(...windD(x,y,t.z,h,d)),t.pr); g[j*W+i]=e*2/(t.pr*24); }
  return gisCache[key]=g; }
const GRAMP=[[0,[215,227,232]],[.1,[159,211,181]],[.2,[232,211,106]],[.32,[240,160,60]],[.45,[217,71,47]]];
const gcol=v=>{ for(let i=1;i<GRAMP.length;i++) if(v<=GRAMP[i][0]){ const [a,ca]=GRAMP[i-1],[b,cb]=GRAMP[i], q=(v-a)/(b-a); return ca.map((c,k)=>Math.round(c+(cb[k]-c)*q)); } return GRAMP[GRAMP.length-1][1]; };
function renderGis(){ const cv=document.getElementById("gis"); if(!cv || !cv.clientWidth) return; const g=gisGrid(myT.m,eo.day), S=8;
  const small=document.createElement("canvas"); small.width=W; small.height=H; const sc=small.getContext("2d"), img=sc.createImageData(W,H), wa=css("--water").match(/#(..)(..)(..)/).slice(1).map(h=>parseInt(h,16));
  for(let j=0;j<H;j++) for(let i=0;i<W;i++){ const v=g[j*W+i], c=v<0?wa:gcol(v), o=((H-1-j)*W+i)*4; img.data[o]=c[0]; img.data[o+1]=c[1]; img.data[o+2]=c[2]; img.data[o+3]=255; }
  sc.putImageData(img,0,0); cv.width=W*S; cv.height=H*S; const c=cv.getContext("2d"); c.imageSmoothingEnabled=true; c.drawImage(small,0,0,W*S,H*S);
  const cells=[]; for(let j=0;j<H;j++) for(let i=0;i<W;i++) if(g[j*W+i]>=0) cells.push([g[j*W+i],i+.5,j+.5]); cells.sort((a,b)=>b[0]-a[0]);
  const best=[]; for(const q of cells){ if(best.every(b=>Math.hypot(b[1]-q[1],b[2]-q[2])>6)) best.push(q); if(best.length===3) break; }
  const P=(x,y)=>[x/W*cv.width,(1-y/H)*cv.height];
  for(const s2 of STATIONS){ const [px,py]=P(...s2.p); c.fillStyle="#3a6fd2"; c.beginPath(); c.arc(px,py,7,0,7); c.fill(); c.strokeStyle="#fff"; c.lineWidth=2; c.stroke(); }
  c.font="700 22px sans-serif"; c.textAlign="center"; c.textBaseline="middle";
  best.forEach((b,n)=>{ const [px,py]=P(b[1],b[2]); c.fillStyle="#14232b"; c.beginPath(); c.arc(px,py,15,0,7); c.fill(); c.strokeStyle="#fff"; c.lineWidth=3; c.stroke(); c.fillStyle="#fff"; c.fillText(String(n+1),px,py+1); });
  { const [px,py]=P(...myT.p); c.strokeStyle="#0a6e7d"; c.lineWidth=5; c.beginPath(); c.arc(px,py,13,0,7); c.stroke(); c.fillStyle="#0a6e7d"; c.beginPath(); c.arc(px,py,5,0,7); c.fill(); }
  document.getElementById("gisbest").innerHTML = `<b>Meilleurs emplacements ${EODAYS[eo.day]} :</b> `+best.map((b,n)=>`<button class="btn" data-gb="${n}" style="margin:4px 6px 0 0">${n+1}. près de ${nearestName([b[1],b[2]])} · ${Math.round(b[0]*100)} %</button>`).join("");
  document.querySelectorAll("[data-gb]").forEach(bt=>bt.onclick=()=>{ const b=best[+bt.dataset.gb]; myT.p=[b[1],b[2]]; renderMyT(); renderGis(); }); }
document.getElementById("gis").addEventListener("click",e=>{ const r=e.target.getBoundingClientRect(), x=(e.clientX-r.left)/r.width*W, y=(1-(e.clientY-r.top)/r.height)*H; myT.p=[x,y]; renderMyT(); renderGis(); });
document.getElementById("gis").addEventListener("mousemove",e=>{ const r=e.target.getBoundingClientRect(), x=(e.clientX-r.left)/r.width*W, y=(1-(e.clientY-r.top)/r.height)*H; if(x<0||x>W||y<0||y>H) return; const v=gisGrid(myT.m,eo.day)[Math.min(H-1,Math.floor(y))*W+Math.min(W-1,Math.floor(x))];
  showTip(v<0?"En mer":`<b>Taux de charge ${Math.round(v*100)} %</b><br><span class="sub">${TMOD[myT.m].n}, moyeu à ${TMOD[myT.m].z} m · près de ${nearestName([x,y])}</span>`, e.clientX, e.clientY, "gis"); });
document.getElementById("gis").addEventListener("mouseleave",()=>hideTip("gis"));
new ResizeObserver(()=>renderGis()).observe(document.getElementById("gis").parentElement);
document.getElementById("eo-day").innerHTML=["Aujourd'hui","Demain","Après-demain"].map((n,i)=>`<button class="chip" data-d="${i}" aria-pressed="${i===0}">${n}</button>`).join("");
document.getElementById("eo-day").onclick=e=>{ const b=e.target.closest(".chip"); if(!b) return; eo.day=+b.dataset.d; document.querySelectorAll("#eo-day .chip").forEach(x=>x.setAttribute("aria-pressed",x===b)); renderEol(); renderMyT(); renderGis(); };
makeMap("c4","map4",{ pin:()=>myT.p, z:()=>TMOD[myT.m].z, marks:()=>TURB.map(t=>({p:t.p, r:.8, cls:"go"})), onClick:(x,y)=>{ myT.p=[x,y]; renderMyT(); } });

// ---------------- coupe verticale ----------------
let lastCut=null;
function renderCut(A,B){
  lastCut=[A,B]; const card=document.getElementById("cutcard"); card.hidden=false;
  const cv=document.getElementById("cutcv"), w=Math.max(320,cv.clientWidth||600), h=260, dpr=Math.min(2,devicePixelRatio||1);
  cv.width=w*dpr; cv.height=h*dpr; cv.style.height=h+"px"; const ctx=cv.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  const L=Math.hypot(B[0]-A[0],B[1]-A[1])||1, tx=(B[0]-A[0])/L, ty=(B[1]-A[1])/L, ZMAX=1300, pl=50, pr=12, pt=10, pb=26, W2=w-pl-pr, H2=h-pt-pb, N=Math.round(W2/4), M=Math.round(H2/4);
  ctx.fillStyle=css("--surface-2"); ctx.fillRect(0,0,w,h);
  const at=s=>{ const x=A[0]+(B[0]-A[0])*s, y=A[1]+(B[1]-A[1])*s, k=idx(x,y); return [x,y,k,wat[k]?0:elev[k]]; };
  for(let i=0;i<N;i++){ const [x,y,k,g]=at((i+.5)/N);
    for(let j=0;j<M;j++){ const zasl=(j+.5)/M*ZMAX, zagl=zasl-g, px=pl+i*W2/N, py=pt+H2-(j+1)*H2/M;
      if(zagl<0) ctx.fillStyle = wat[k] ? css("--water") : css("--land-hi"); else if(zagl>1000) ctx.fillStyle=css("--surface-2"); else { const c=speedColor(kmh(Math.hypot(...wind(x,y,Math.max(10,zagl),state.hour)))); ctx.fillStyle=`rgb(${c[0]},${c[1]},${c[2]})`; }
      ctx.fillRect(px,py,W2/N+.7,H2/M+.7); } }
  ctx.strokeStyle=css("--ink"); ctx.fillStyle=css("--ink"); ctx.lineWidth=1.6;
  for(let i=5;i<N;i+=12) for(const zagl of [40,200,450,750]){ const [x,y,,g]=at((i+.5)/N), zasl=g+zagl; if(zasl>ZMAX) continue; const [u,v]=wind(x,y,zagl,state.hour), al=u*tx+v*ty, len=Math.max(-16,Math.min(16,al*2)), px=pl+(i+.5)*W2/N, py=pt+H2-zasl/ZMAX*H2;
    if(Math.abs(len)<2){ ctx.beginPath(); ctx.arc(px,py,2,0,7); ctx.fill(); continue; } ctx.beginPath(); ctx.moveTo(px-len/2,py); ctx.lineTo(px+len/2,py); ctx.stroke(); const sg=Math.sign(len); ctx.beginPath(); ctx.moveTo(px+len/2,py); ctx.lineTo(px+len/2-sg*5,py-3.5); ctx.lineTo(px+len/2-sg*5,py+3.5); ctx.closePath(); ctx.fill(); }
  ctx.fillStyle=css("--muted"); ctx.font=`11px ${css("--f-data")}`; ctx.textAlign="right";
  for(const z of [0,400,800,1200]){ const py=pt+H2-z/ZMAX*H2; ctx.fillText(`${z} m`,pl-6,py+4); }
  ctx.textAlign="left"; ctx.font=`700 13px ${css("--f-body")}`; ctx.fillStyle=css("--ink"); ctx.fillText("A",pl+2,h-8); ctx.textAlign="right"; ctx.fillText("B",w-pr-2,h-8); ctx.textAlign="center"; ctx.font=`11px ${css("--f-data")}`; ctx.fillStyle=css("--muted"); ctx.fillText(`${L.toFixed(1).replace(".",",")} km`,pl+W2/2,h-8);
  document.getElementById("cuttxt").textContent = `De A (près de ${nearestName(A)}) à B (près de ${nearestName(B)}), à ${state.hour} h. La couleur donne la force du vent du sol jusqu'à 1 000 m au-dessus du sol ; le relief est en brun, la mer en bleu. On voit le vent accélérer au-dessus des sommets et changer près de la côte.`;
}
document.getElementById("cutclose").onclick=()=>{ document.getElementById("cutcard").hidden=true; lastCut=null; };

// ---------------- AGRICULTEUR ----------------
const CROPS = { tomate:["Tomate",1.15], pdt:["Pomme de terre",1.15], pasteque:["Pastèque, melon",1.0], piment:["Piment, poivron",1.05], fraise:["Fraise",0.85], agrumes:["Agrumes",0.65], olivier:["Olivier",0.7], vigne:["Vigne",0.7], artichaut:["Artichaut",1.0], palmier:["Palmier dattier",0.95] };
const MILIEUX = { champ:"Plein champ", serre:"Serre ou tunnel", verger:"Verger" };
const AGDAYS = ["Aujourd'hui","Demain","Après-demain"];
const ag = { place:5, crop:"tomate", area:2, eff:.9, milieu:"champ", day:0 };
const agP = () => { const q=PLACES[ag.place]; return xy(q[1],q[2]); };
// Prototype : demain et après-demain sont des variantes simulées de la journée d'exemple. En service : prévision GFS à +24 h et +48 h, affinée par ATOM Wind.
const DAYMOD = [{f:1,rot:0,dT:0,dRH:0},{f:1.35,rot:20,dT:4,dRH:-16},{f:.7,rot:-30,dT:-2,dRH:6}];
function windD(x,y,z,h,d){ if(window.ATOM && ATOM.pretJour(d)) return ATOM.ventJour(d,x,y,z,h); const [u,v]=wind(x,y,z,h), m=DAYMOD[d], a=m.rot*Math.PI/180; return [m.f*(u*Math.cos(a)-v*Math.sin(a)), m.f*(u*Math.sin(a)+v*Math.cos(a))]; }
const Tair = (h,x,y,d=0) => (window.ATOM && ATOM.aT(d)) ? ATOM.tJour(d,x,y,h) : 26.5 + 6.5*Math.sin((h-9)*Math.PI/12) - .0065*elev[idx(x,y)] + DAYMOD[d].dT;
const RHair = (h,x,y,d=0) => (window.ATOM && ATOM.aT(d)) ? ATOM.hrJour(d,x,y,h) : Math.max(15, Math.min(98, 62 - 18*Math.sin((h-9)*Math.PI/12) + (sdist[idx(x,y)]<4 ? 8 : 0) + DAYMOD[d].dRH));
const esat = T => .6108*Math.exp(17.27*T/(T+237.3));
function et0Day(x,y,d=0,du=0){ // FAO-56 Penman-Monteith, pas journalier ; du = vent ajouté (m/s) pour mesurer l'effet du vent
  const Ts=[], RH=[], U=[]; for(let h=0;h<24;h++){ Ts.push(Tair(h,x,y,d)); RH.push(RHair(h,x,y,d)); U.push(Math.hypot(...windD(x,y,10,h,d))*0.748 + du); }
  const Tmax=Math.max(...Ts), Tmin=Math.min(...Ts), T=(Tmax+Tmin)/2, u2=U.reduce((a,b)=>a+b)/24, rh=RH.reduce((a,b)=>a+b)/24, z=elev[idx(x,y)];
  const D=4098*esat(T)/((T+237.3)**2), P=101.3*((293-.0065*z)/293)**5.26, g=.000665*P, es=(esat(Tmax)+esat(Tmin))/2, ea=es*rh/100;
  const J=(window.ATOM && ATOM.doyJour(d)) || 200+d, phi=(DOM ? LAT0 + y/KY : 36.65)*Math.PI/180, dr=1+.033*Math.cos(2*Math.PI*J/365), dl=.409*Math.sin(2*Math.PI*J/365-1.39), ws=Math.acos(-Math.tan(phi)*Math.tan(dl));
  const Ra=24*60/Math.PI*.082*dr*(ws*Math.sin(phi)*Math.sin(dl)+Math.cos(phi)*Math.cos(dl)*Math.sin(ws)), Rs=(window.ATOM && ATOM.rsJour(d)!==null) ? ATOM.rsJour(d) : .72*Ra, Rso=(.75+2e-5*z)*Ra;
  const Rnl=4.903e-9*(((Tmax+273.16)**4+(Tmin+273.16)**4)/2)*(.34-.14*Math.sqrt(ea))*(1.35*Rs/Rso-.35), Rn=.77*Rs-Rnl;
  const et0=(.408*D*Rn + g*900/(T+273)*u2*(es-ea))/(D+g*(1+.34*u2));
  return {et0, Tmax, Tmin, u2, rh};
}
function hourBars(id, rows, legend){ const el=document.getElementById(id); if(!el) return; const mx=Math.max(...rows.map(r=>r.v),1);
  el.innerHTML = `<div class="day">${rows.map((r,h)=>`<button class="${r.cls}${h===state.hour?" now":""}" style="height:${Math.round(16+60*r.v/mx)}px" data-h="<b>${h} h</b><br>${r.t}" aria-label="${h} h : ${r.t}">​</button>`).join("")}</div><div class="dayaxis">${Array.from({length:24},(_,h)=>`<span>${h%3===0?h:""}</span>`).join("")}</div>${legend||""}`;
  el.querySelectorAll(".day button").forEach((b,h)=>b.onclick=()=>setHour(h)); }
function windowsOf(rows, cls="go"){ const out=[]; let st=null; rows.forEach((r,h)=>{ if(r.cls===cls){ if(st===null) st=h; } else if(st!==null){ out.push([st,h]); st=null; } }); if(st!==null) out.push([st,24]); return out; }
const winTxt = ws => ws.length ? ws.map(([a,b])=>`<span class="pill good">${a} h – ${b} h</span>`).join(" ") : `<span class="pill bad">Aucun créneau favorable</span>`;
const SRC_ATOM = '<span class="srctag atom">Vent : ATOM Wind</span>', SRC_GFS = '<span class="srctag">Température, humidité : GFS</span>', SRC_FAO = '<span class="srctag">Méthode FAO-56</span>';
const AIC = { stop:'<path d="M12 5v9M12 18v.5" stroke="#fff" stroke-width="3" stroke-linecap="round"/>', warn:'<path d="M12 6v8M12 17.5v.5" stroke="#fff" stroke-width="3" stroke-linecap="round"/>', go:'<path d="M5 12l5 5 9-10" stroke="#fff" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>' };
const spanTxt = hs => { const w=[]; let a=null; hs.forEach((h,i)=>{ if(a===null) a=h; if(hs[i+1]!==h+1){ w.push(a===h?`${a} h`:`${a} h à ${h+1} h`); a=null; } }); return w.join(", "); };
function renderAgri(){
  const SRC_ATOM = window.ATOM?.pretJour(ag.day) || (ag.day===0&&window.ATOM?.pret()) ? '<span class="srctag atom">Vent : ATOM Wind</span>' : '<span class="srctag">Vent : simulation</span>';
  const SRC_GFS = window.ATOM?.aT(ag.day) ? '<span class="srctag">Température, humidité : GFS</span>' : '<span class="srctag">Température, humidité : exemple</span>';
  const [x,y]=agP(), d=ag.day, c=CROPS[ag.crop], jour=AGDAYS[d].toLowerCase(), fmt1=v=>v.toFixed(1).replace(".",","), place=PLACES[ag.place][0].split(" (")[0];
  const H=[]; for(let h=0;h<24;h++){ const [u,v]=windD(x,y,10,h,d), [u2,v2]=windD(x,y,2,h,d), k=kmh(Math.hypot(u,v)); H.push({u,v,k:Math.round(k), k2:Math.round(kmh(Math.hypot(u2,v2))), g:Math.round(k*gustF(x,y,10)), T:Tair(h,x,y,d), RH:RHair(h,x,y,d)}); }
  const rel=sigma(x,y), [rc,rl]=relClass(rel.s);
  // ---- alertes du jour
  const gmax=Math.max(...H.map(r=>r.g)), hg=H.findIndex(r=>r.g===gmax);
  const hot=H.map((r,h)=>r.T>=35&&r.RH<=30&&r.k>=20?h:-1).filter(h=>h>=0), warm=H.map((r,h)=>r.T>=32&&r.RH<=40&&r.k>=15?h:-1).filter(h=>h>=0);
  const AL=[];
  if(gmax>=60) AL.push(["stop",`Vent fort ${jour} : rafales jusqu'à ${gmax} km/h vers ${hg} h`,"Fermez et arrimez serres et tunnels, reportez la taille et tous les traitements, vérifiez les brise-vent et les tuteurs.",SRC_ATOM]);
  else if(gmax>=45) AL.push(["warn",`Rafales assez fortes ${jour} : jusqu'à ${gmax} km/h vers ${hg} h`,"Vérifiez la fixation des films plastiques et des filets ; évitez les travaux en hauteur à ce moment.",SRC_ATOM]);
  if(hot.length) AL.push(["stop",`Vent chaud et sec ${jour} : ${spanTxt(hot)}`,"Arrosez tôt le matin et augmentez la dose, ne traitez pas, ombrez et aérez les serres : les plantes perdent beaucoup d'eau et les fleurs peuvent tomber.",SRC_ATOM+SRC_GFS]);
  else if(warm.length) AL.push(["warn",`Air chaud, sec et venté ${jour} : ${spanTxt(warm)}`,"Prévoyez l'arrosage avant ces heures et surveillez les jeunes plants.",SRC_ATOM+SRC_GFS]);
  if(!AL.length) AL.push(["go",`Pas d'alerte vent pour ${jour} à ${place}`,"Ni rafales fortes, ni vent chaud et sec prévus.",SRC_ATOM]);
  document.getElementById("ag-alerts").innerHTML = AL.map(([cls,tt,todo,src])=>`<div class="alert ${cls}" style="cursor:default"><div class="ic"><svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">${AIC[cls]}</svg></div><div><h3>${tt}</h3><div class="todo">${todo}</div><div class="srcrow">${src}</div></div></div>`).join("");
  // ---- traitement
  const rowsT=H.map(r=>({v:r.k, cls: r.k>19?"stop" : (r.k>15||r.g>30||r.k<3)?"warn":"go", t:`${r.k} km/h, rafales ${r.g} km/h`}));
  // ---- eau
  const e=et0Day(x,y,d), etc=e.et0*c[1], gross=etc/ag.eff, m3ha=gross*10, tot=m3ha*ag.area, plus=Math.round((et0Day(x,y,d,10/3.6).et0/e.et0-1)*100);
  const asp = ag.eff<.8 && ag.eff>.7;
  const rowsIrr=H.map(r=>({v:r.k*.8+Math.max(0,r.T-22)*2, cls:(asp ? (r.k>25||r.T>32) : r.T>32) ? "stop" : (asp ? (r.k>15||r.T>28) : r.T>29) ? "warn" : "go", t:`${r.k} km/h, ${Math.round(r.T)} °C`}));
  document.getElementById("ag-top").innerHTML = `
    <div class="card"><h3>Quand traiter ${jour} ?</h3>${window.ATOM && ATOM.htmlProba ? ATOM.htmlProba(d, x, y) : ""}<div id="ag-trbars"></div><p style="margin:8px 0 4px">Créneaux conseillés : ${winTxt(windowsOf(rowsT))}</p>
      <p class="muted" style="font-size:14px;margin:0">Non au-dessus de 19 km/h à 10 m ; attention entre 15 et 19 km/h, si les rafales dépassent 30 km/h, ou sous 3 km/h. Vent au niveau de la rampe (2 m) à ${state.hour} h : ${H[state.hour].k2} km/h. Fiabilité ici : <span class="pill ${rc}">${rl}</span></p><div class="srcrow">${SRC_ATOM}</div></div>
    <div class="answer info"><div class="verdict"><div class="badge-big"><svg width="28" height="28" viewBox="0 0 24 24"><path class="drop" d="M12 3c3 4 5 7 5 10a5 5 0 0 1-10 0c0-3 2-6 5-10z" fill="#fff"/></svg></div><div><div class="muted" style="font-size:14px;font-weight:700">${place} · ${c[0]} · ${MILIEUX[ag.milieu].toLowerCase()} · ${jour}</div><div class="big">Apportez ${Math.round(m3ha)} m³ par hectare</div></div></div>
      <p>Soit <b>${Math.round(tot).toLocaleString("fr-FR")} m³</b> pour vos ${fmt1(ag.area)} ha (${fmt1(gross)} mm d'eau). ${ag.milieu==="serre"?"Sous serre, le besoin réel est souvent 20 à 30 % plus faible : ajustez selon l'aération.":""}</p>
      <div class="facts"><div class="fact"><div class="k">Évaporation de référence <span class="tip" tabindex="0" data-tip="Quantité d'eau qu'un gazon perdrait ce jour-là, calculée avec le vent, la chaleur, l'humidité et le soleil (méthode FAO-56).">?</span></div><div class="v">${fmt1(e.et0)} mm</div></div>
        <div class="fact"><div class="k">Besoin de la culture</div><div class="v">${fmt1(etc)} mm</div></div>
        <div class="fact"><div class="k">Vent moyen à 2 m</div><div class="v">${fmtSpeed(e.u2)}</div></div>
        <div class="fact"><div class="k">Chaleur du jour</div><div class="v">${Math.round(e.Tmin)} à ${Math.round(e.Tmax)} °C</div></div></div>
      <p class="muted" style="font-size:14px">Le vent compte : avec 10 km/h de vent en plus, cette parcelle aurait besoin d'environ ${plus} % d'eau en plus.</p><div class="srcrow">${SRC_ATOM}${SRC_GFS}${SRC_FAO}</div></div>`;
  const cards=[];
  cards.push(`<div class="card"><h3>Quand arroser ${jour} ?</h3><div id="ag-irrbars"></div><p style="margin:8px 0 4px">Meilleurs moments : ${winTxt(windowsOf(rowsIrr))}</p><p class="muted" style="font-size:14px;margin:0">${asp?"En aspersion, le vent emporte l'eau et la chaleur l'évapore : arrosez quand le vent est faible et l'air frais.":"Avec ce mode d'arrosage, l'heure compte moins ; évitez seulement les heures les plus chaudes."}</p><div class="srcrow">${SRC_ATOM}${SRC_GFS}</div></div>`);
  if(ag.milieu==="serre"){ const r=H[state.hour], lee=DIRS[(Math.round(fromDeg(r.u,r.v)/45)+4)%8].replace(/^du |^de l'|^de /,"");
    cards.push(`<div class="card"><h3>Aérer la serre ${d===0?"":jour+" "}à ${state.hour} h</h3><p style="margin:0 0 6px"><b>${r.g>50?"Fermez les ouvrants : rafales trop fortes.":`Ouvrez les ouvrants côté ${lee}`}</b></p><p class="muted" style="font-size:14px;margin:0">Le vent viendra ${dirName(r.u,r.v)} à ${r.k} km/h, rafales ${r.g} km/h. ${r.g>50?"Le film et les arceaux risquent de souffrir.":"En ouvrant à l'abri du vent, l'air se renouvelle sans courant violent sur les plantes."}</p><div class="srcrow">${SRC_ATOM}</div></div>`); }
  let rowsB=null;
  if(ag.milieu==="verger"){ rowsB=H.map((r,h)=>{ const day=h>=7&&h<=19; return {v:day?Math.max(1,30-r.k):1, cls:!day?"info":(r.k<20&&r.T>=15&&r.T<=35)?"go":"warn", t:`${r.k} km/h`}; });
    cards.push(`<div class="card"><h3>Les abeilles butineront-elles ${jour} ?</h3><div id="ag-beebars"></div><p style="margin:8px 0 0">Bons moments pour la pollinisation : ${winTxt(windowsOf(rowsB))}</p><p class="muted" style="font-size:14px;margin:4px 0 0">Les abeilles sortent peu au-dessus de 20 km/h de vent. Ne traitez jamais pendant ces créneaux.</p><div class="srcrow">${SRC_ATOM}</div></div>`); }
  document.getElementById("ag-cards").innerHTML = cards.join("");
  hourBars("ag-trbars",rowsT); hourBars("ag-irrbars",rowsIrr); if(rowsB) hourBars("ag-beebars",rowsB);
}
document.getElementById("ag-place").innerHTML = PLACES.map((p,i)=>`<option value="${i}">${p[0]}</option>`).join(""); document.getElementById("ag-place").value="5";
document.getElementById("ag-crop").innerHTML = Object.entries(CROPS).map(([k,c])=>`<option value="${k}">${c[0]}</option>`).join("");
document.getElementById("ag-milieu").innerHTML = Object.entries(MILIEUX).map(([k,n])=>`<button class="chip" data-m="${k}" aria-pressed="${k===ag.milieu}">${n}</button>`).join("");
document.getElementById("ag-day").innerHTML = AGDAYS.map((n,i)=>`<button class="chip" data-d="${i}" aria-pressed="${i===ag.day}">${n}</button>`).join("");
document.getElementById("ag-milieu").onclick=e=>{ const b=e.target.closest(".chip"); if(!b) return; ag.milieu=b.dataset.m; document.querySelectorAll("#ag-milieu .chip").forEach(x=>x.setAttribute("aria-pressed",x===b)); renderAgri(); };
document.getElementById("ag-day").onclick=e=>{ const b=e.target.closest(".chip"); if(!b) return; ag.day=+b.dataset.d; document.querySelectorAll("#ag-day .chip").forEach(x=>x.setAttribute("aria-pressed",x===b)); renderAgri(); };
document.getElementById("ag-place").onchange=e=>{ ag.place=+e.target.value; renderAgri(); };
document.getElementById("ag-crop").onchange=e=>{ ag.crop=e.target.value; renderAgri(); };
document.getElementById("ag-area").oninput=e=>{ const v=parseFloat(e.target.value); if(v>0){ ag.area=v; renderAgri(); } };
document.getElementById("ag-irr").onchange=e=>{ ag.eff=+e.target.value; renderAgri(); };

// ---------------- FIABILITÉ ----------------
const map2 = makeMap("c2","map2",{ rel:true, z:()=>10, onClick:(x,y)=>openSheet(x,y,10) });
function relStats(){ let n=0, g=0; for(let j=0;j<NY;j+=2) for(let i=0;i<NX;i+=2){ if(wat[j*NX+i]) continue; n++; if(sigma((i+.5)/G,(j+.5)/G).s<.9) g++; } return Math.round(g/n*100); }
function renderRel(){ const a=relStats(); const other = relMode==="reel" ? (relMode="aneo", relStats()) : null; if(other!==null) relMode="reel";
  document.getElementById("relstats").innerHTML = (window.ATOM ? ATOM.noteFiabilite() : "") + (relMode==="reel"
    ? `<h3>Aujourd'hui, avec 3 stations</h3><div class="kpi"><div class="v">${a} %</div><div class="s">des terres avec une fiabilité bonne</div></div><p style="margin-bottom:0">Avec le réseau ANEO prévu (${ANEO.length} capteurs, points orange), ce chiffre passerait à <b>${other} %</b>. Le plan de déploiement place chaque capteur là où il apporte le plus.</p>`
    : `<h3>Avec le réseau ANEO prévu</h3><div class="kpi"><div class="v">${a} %</div><div class="s">des terres avec une fiabilité bonne</div></div><p style="margin-bottom:0">Les points orange montrent les ${ANEO.length} emplacements proposés, dans l'ordre où les installer. Zones d'eau exclues.</p>`);
}
function setRel(m){ relMode=m; document.getElementById("m-reel").setAttribute("aria-pressed",m==="reel"); document.getElementById("m-aneo").setAttribute("aria-pressed",m==="aneo"); map2.redraw(); renderRel(); renderAnswer(); }
document.getElementById("m-reel").onclick=()=>setRel("reel"); document.getElementById("m-aneo").onclick=()=>setRel("aneo");

// ---------------- fiche d'un point ----------------
const sheet=document.getElementById("sheet");
function openSheet(x,y,zSel){
  const k=idx(x,y), [u,v]=wind(x,y,zSel,state.hour), sp=Math.hypot(u,v), rel=sigma(x,y), [rc,rl]=relClass(rel.s), deg=(fromDeg(u,v)+180)%360;
  const Z=[10,25,50,100,150,200,300,400,500,650,800,1000], prof=Z.map(z=>kmh(Math.hypot(...wind(x,y,z,state.hour)))), mx=Math.max(...prof)*1.15;
  const Wp=300,Hp=170, X=s=>34+s/mx*(Wp-44), Yz=z=>Hp-18-Math.log10(z/10)/2*(Hp-30);
  const pts=prof.map((s,i)=>`${X(s).toFixed(1)},${Yz(Z[i]).toFixed(1)}`).join(" ");
  let near=PLACES.map(p=>[p[0],Math.hypot(x-xy(p[1],p[2])[0],y-xy(p[1],p[2])[1])]).sort((a,b)=>a[1]-b[1])[0];
  sheet.innerHTML = `<header><div><div class="muted" style="font-size:13px;font-weight:700">${wat[k]===1?"En mer":`Altitude du sol : ${Math.round(elev[k])} m`} · près de ${near[0].split(" (")[0]}</div><h2 style="font-size:24px">${fmtSpeed(sp)} à ${zSel} m</h2></div><button class="x" aria-label="Fermer">×</button></header>
    <div class="row">${arrowSvg(deg,30,css("--accent"))}<b>Vent ${dirName(u,v)}</b></div>
    <div class="muted">${beaufort(kmh(sp))}</div>
    <details><summary style="cursor:pointer;font-weight:700">Pourquoi ce vent ?</summary>${explainHtml(x,y,zSel,state.hour)}</details>
    <div class="row"><span class="pill m">Valeur calculée</span><span class="pill ${rc}">Fiabilité ${rl}</span></div>
    <p style="margin:0;font-size:15px">Calculée à partir des stations et de la météo de grande échelle. Le capteur le plus proche (${rel.who}) est à ${Math.round(rel.d)} km. Erreur probable : ± ${Math.round(kmh(rel.s))} km/h.</p>
    <div><b style="font-size:15px">Le vent selon la hauteur</b><svg viewBox="0 0 ${Wp} ${Hp}" style="width:100%;height:auto" role="img" aria-label="Profil vertical du vent">
      ${[10,100,1000].map(z=>`<line x1="34" x2="${Wp-8}" y1="${Yz(z)}" y2="${Yz(z)}" stroke="var(--line)"/><text x="30" y="${Yz(z)+4}" text-anchor="end">${z} m</text>`).join("")}
      <polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="2.5"/>${prof.map((s,i)=>`<circle cx="${X(s)}" cy="${Yz(Z[i])}" r="3" fill="var(--accent)"/>`).join("")}
      <text x="${Wp-8}" y="${Hp-4}" text-anchor="end">km/h →</text></svg></div>`;
  sheet.hidden=false; sheet.querySelector(".x").onclick=()=>sheet.hidden=true;
}

// ---------------- CARTE 3D ----------------
let update3dColors=null; const H3=[10,30,50,80,150,300,500,1000]; let z3=80, relOn=false,terrainStarting=null,sceneMode="terrain";
function setSceneMode(mode){
  sceneMode=mode;document.getElementById("v-carte3d").dataset.sceneMode=mode;
  document.getElementById("terrain-pane").hidden=mode!=="terrain";
  document.getElementById("three").hidden=mode!=="analysis";
  document.getElementById("analysis-terrain-note").hidden=mode!=="analysis";
  document.getElementById("terrain-mode-map").setAttribute("aria-pressed",mode==="terrain");
  document.getElementById("terrain-mode-analysis").setAttribute("aria-pressed",mode==="analysis");
  if(window.ATOM?.workspace)publishWorkspace();
  if(mode==="analysis"&&!threeReady){threeReady=true;ATOM.loadThree().then(init3d).catch(e=>{threeReady=false;toast("La scène 3D n’a pas pu se charger. Réessayez.");console.error(e);});}
  if(mode==="terrain"&&!terrainStarting){terrainStarting=ATOM.loadTerrain().catch(error=>{terrainStarting=null;document.getElementById("terrain-status").textContent="Le terrain n’a pas pu se charger. Vérifiez les fichiers importés, puis réessayez.";document.getElementById("terrain-retry").hidden=false;console.error(error);});}
}
document.getElementById("terrain-mode-map").onclick=()=>setSceneMode("terrain");
document.getElementById("terrain-mode-analysis").onclick=()=>setSceneMode("analysis");
document.getElementById("terrain-retry").onclick=()=>{document.getElementById("terrain-retry").hidden=true;setSceneMode("terrain");};
function init3d(){
  const host=document.getElementById("three");
  if(!window.THREE){ host.insertAdjacentHTML("beforeend",`<div class="nogl"><div class="card"><b>La vue 3D n'a pas pu se charger.</b><br><span class="muted">Vérifiez la connexion, puis rechargez la page.</span></div></div>`); return; }
  let renderer; try{ renderer=new THREE.WebGLRenderer({antialias:true, alpha:true}); }catch(e){ host.insertAdjacentHTML("beforeend",`<div class="nogl"><div class="card"><b>Votre appareil n'affiche pas la 3D.</b><br><span class="muted">Utilisez l'onglet « Mon vent ».</span></div></div>`); return; }
  renderer.setPixelRatio(Math.min(1.75,devicePixelRatio)); renderer.outputColorSpace=THREE.SRGBColorSpace; renderer.toneMapping=THREE.ACESFilmicToneMapping; renderer.toneMappingExposure=1.15; host.prepend(renderer.domElement);
  // K3 : échelle de la caméra, du brouillard et des repères quand le domaine dépasse le Grand Tunis (national : × 12)
  const K3 = DOM ? Math.max(W, H) / 68 : 1;
  const scene=new THREE.Scene(), cam=new THREE.PerspectiveCamera(45,host.clientWidth/host.clientHeight,.1,500*K3); let EX=11/1000, AW3=null; const OBJ3={stations:[],turb:[]};
  scene.fog=new THREE.FogExp2(0x21382e,.004/K3); scene.add(new THREE.HemisphereLight(0xeaf1de,0x344838,2.1)); const sun=new THREE.DirectionalLight(0xfff1cf,2.4); sun.position.set(-30,40,20); scene.add(sun);
  const SEG=DOM ? [NX, NY] : [136,112], geo=new THREE.PlaneGeometry(W,H,SEG[0],SEG[1]); geo.rotateX(-Math.PI/2);
  const pos=geo.attributes.position, cols=new Float32Array(pos.count*3);
  for(let i=0;i<pos.count;i++){ const x=pos.getX(i)+W/2, y=H/2-pos.getZ(i), k=idx(x,y); pos.setY(i, wat[k]?-.02:elev[k]*EX); pos.setX(i,x-W/2); pos.setZ(i,-(y-H/2)); }
  geo.computeVertexNormals(); geo.setAttribute("color",new THREE.BufferAttribute(cols,3));
  const mesh=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({vertexColors:true,roughness:.95,metalness:0})); scene.add(mesh);
  function colorTerrain(){ const lo=new THREE.Color(0x839c67), mid=new THREE.Color(0xb5ab77), hi=new THREE.Color(0xd8d0a4), wa=new THREE.Color(0x487e7b), la=new THREE.Color(0x608f7b), ur=new THREE.Color(0x9eaa8b);
    for(let i=0;i<pos.count;i++){ const x=pos.getX(i)+W/2, y=-pos.getZ(i)+H/2, k=idx(x,y); let c;
      if(wat[k]===1) c=wa.clone(); else if(wat[k]===2) c=la.clone(); else { const t=Math.min(1,elev[k]/700); c = t<.5 ? lo.clone().lerp(mid,t*2) : mid.clone().lerp(hi,(t-.5)*2); if(z0[k]>.7) c.lerp(ur,.6); }
      if(relOn && !wat[k]){ const s=sigma(x,y).s, t=Math.min(1,Math.max(0,(s-.45)/1.6)); c.lerp(new THREE.Color(t<.3?0x1f9d55:t<.6?0xd98a00:0xd23a3a),.5); c.lerp(new THREE.Color(0xffffff),Math.max(0,t-.5)*.6); }
      cols[i*3]=c.r; cols[i*3+1]=c.g; cols[i*3+2]=c.b; } geo.attributes.color.needsUpdate=true; }
  colorTerrain();
  // particules
  const N=Math.min(2800,Math.max(800,Math.round(host.clientWidth*host.clientHeight/260))), pp=new Float32Array(N*3), pc=new Float32Array(N*3), P=[]; const pg=new THREE.BufferGeometry(); pg.setAttribute("position",new THREE.BufferAttribute(pp,3)); pg.setAttribute("color",new THREE.BufferAttribute(pc,3));
  const pts=new THREE.Points(pg,new THREE.PointsMaterial({size:.16*K3, vertexColors:true, transparent:true, opacity:.8, depthWrite:false})); scene.add(pts);
  for(let i=0;i<N;i++) P.push({x:Math.random()*W, y:Math.random()*H, a:Math.random()*90|0});
  // stations et éoliennes
  for(const s of STATIONS){ const k=idx(...s.p), m=new THREE.Mesh(new THREE.CylinderGeometry(.18*K3,.18*K3,1.2*K3,10),new THREE.MeshLambertMaterial({color:0x3a6fd2})); m.position.set(s.p[0]-W/2,elev[k]*EX+.6*K3,-(s.p[1]-H/2)); scene.add(m); OBJ3.stations.push({m,p:s.p}); }
  for(const t of TURB){ const k=idx(...t.p), m=new THREE.Mesh(new THREE.CylinderGeometry(.06,.1,80*EX*4,8),new THREE.MeshLambertMaterial({color:0xf2f2f2})); m.position.set(t.p[0]-W/2,elev[k]*EX+80*EX*2,-(t.p[1]-H/2)); scene.add(m); OBJ3.turb.push({m,p:t.p}); }
  const pin=new THREE.Mesh(new THREE.ConeGeometry(.5,1.6,16),new THREE.MeshLambertMaterial({color:0x0a6e7d})); pin.rotation.x=Math.PI; pin.visible=false; scene.add(pin);
  const labels=[...STATIONS.map(s=>({t:s.n,p:s.p,st:true})), ...(DOM ? [] : [{t:"Djebel Ressas",p:xy(36.61,10.33)},{t:"Golfe de Tunis",p:xy(36.84,10.42)},{t:"Tunis",p:xy(36.80,10.18)}]),{t:"Parc éolien virtuel",p:xy(36.612,10.33)}]
    .map(l=>{ const d=document.createElement("div"); d.className="lbl3d"+(l.st?" st":""); d.textContent=l.t; host.appendChild(d); const k=idx(...l.p); return {d, p:l.p, st:!!l.st, v:new THREE.Vector3(l.p[0]-W/2, (wat[k]?0:elev[k]*EX)+1.6, -(l.p[1]-H/2))}; });
  const ctl=new THREE.OrbitControls(cam,renderer.domElement); ctl.enableDamping=true; ctl.dampingFactor=.09; ctl.screenSpacePanning=true; ctl.zoomSpeed=.8; ctl.panSpeed=.75; ctl.maxPolarAngle=Math.PI*.47; ctl.minDistance=12*(DOM ? 2 : 1); ctl.maxDistance=280*K3;
  const home=()=>{ cam.position.set(-20,68,82).multiplyScalar(Math.max(1,.95/cam.aspect)*K3); ctl.target.set(0,2,0); ctl.update(); }; home();
  document.getElementById("recenter").onclick=home;
  const rb=document.getElementById("relbtn"); rb.onclick=()=>{ relOn=!relOn; rb.setAttribute("aria-pressed",relOn); rb.textContent=relOn?"Masquer la fiabilité":"Afficher la fiabilité"; colorTerrain(); };
  const hs=document.getElementById("h3"); hs.oninput=()=>{ z3=H3[+hs.value]; document.getElementById("h3txt").textContent=`${z3.toLocaleString("fr-FR")} m`;if(window.ATOM?.workspace)publishWorkspace(); };
  const ray=new THREE.Raycaster(), m2=new THREE.Vector2(); let down=null, hovPend=null;
  const hring=new THREE.Mesh(new THREE.RingGeometry(.55,.8,32),new THREE.MeshBasicMaterial({color:0xffffff, transparent:true, opacity:.9, side:THREE.DoubleSide, depthTest:false})); hring.rotation.x=-Math.PI/2; hring.visible=false; hring.renderOrder=5; scene.add(hring);
  renderer.domElement.addEventListener("pointermove", e => { if(e.buttons){ hideTip("3d"); hring.visible=false; return; } if(hovPend) return; hovPend=requestAnimationFrame(()=>{ hovPend=null;
    const r=renderer.domElement.getBoundingClientRect(); m2.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1); ray.setFromCamera(m2,cam); const hit=ray.intersectObject(mesh)[0];
    if(!hit){ hideTip("3d"); hring.visible=false; return; } const x=hit.point.x+W/2, y=-hit.point.z+H/2, k=idx(x,y), zh=AW3?Math.max(2,AW3.zSol(x,y)):z3, [u,v]=wind(x,y,zh,state.hour), sp=Math.hypot(u,v), deg=(fromDeg(u,v)+180)%360;
    hring.position.set(hit.point.x, hit.point.y+.08, hit.point.z); hring.visible=true;
    showTip(`<div style="display:flex;gap:8px;align-items:center"><b>${fmtSpeed(sp)}</b>${arrowSvg(deg,18,"currentColor")}</div>Vent ${dirName(u,v)}, ${AW3?AW3.texteHauteur(x,y):`à ${z3} m du sol`}<br><span class="sub">${wat[k]===1?"En mer":`Sol à ${Math.round(elev[k])} m`} · près de ${nearestName([x,y])} · cliquez pour le détail</span>`, e.clientX, e.clientY, "3d"); }); });
  renderer.domElement.addEventListener("pointerleave", () => { hideTip("3d"); hring.visible=false; });
  renderer.domElement.addEventListener("pointerdown",e=>down=[e.clientX,e.clientY]);
  let cutMode=false, cutPts=[], cutObjs=[]; const cb=document.getElementById("cutbtn"), cap=document.getElementById("cap3");
  const clearCut=()=>{ cutObjs.forEach(o=>{scene.remove(o);o.geometry?.dispose();o.material?.dispose();}); cutObjs=[]; cutPts=[]; };
  cb.onclick=()=>{ cutMode=!cutMode; clearCut(); cb.setAttribute("aria-pressed",cutMode); cb.textContent=cutMode?"Touchez 2 points du relief":"Tracer une coupe"; cap.hidden=!cutMode; cap.textContent="Touchez le point de départ A, puis le point d'arrivée B."; };
  function drawCut(A,B){ const pts=[]; for(let i=0;i<=80;i++){ const s=i/80, x=A[0]+(B[0]-A[0])*s, y=A[1]+(B[1]-A[1])*s, k=idx(x,y); pts.push(new THREE.Vector3(x-W/2,(wat[k]?0:elev[k]*EX)+.35,-(y-H/2))); }
    const ln=new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0xc4842f})); scene.add(ln); cutObjs.push(ln); }
  renderer.domElement.addEventListener("pointerup",e=>{ if(!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>6) return; const r=renderer.domElement.getBoundingClientRect(); m2.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1); ray.setFromCamera(m2,cam); const hit=ray.intersectObject(mesh)[0]; if(!hit) return; const x=hit.point.x+W/2, y=-hit.point.z+H/2;
    if(seedMode){ launchSeed(x,y,hit.point); return; }
    if(cutMode){ cutPts.push([x,y]); const mk=new THREE.Mesh(new THREE.SphereGeometry(.45,16,12),new THREE.MeshLambertMaterial({color:0xc4842f})); mk.position.set(hit.point.x,hit.point.y+.5,hit.point.z); scene.add(mk); cutObjs.push(mk);
      if(cutPts.length===2){ drawCut(...cutPts); renderCut(...cutPts); cutMode=false; cb.setAttribute("aria-pressed",false); cb.textContent="Nouvelle coupe"; cap.hidden=true; document.getElementById("cutcard").scrollIntoView({behavior:"smooth",block:"nearest"}); } else cap.textContent="Touchez maintenant le point d'arrivée B."; return; }
    pin.position.set(hit.point.x,hit.point.y+1.1,hit.point.z); pin.visible=true; placePin(x,y); if(AW3 && AW3.fiche(x,y)) return; openSheet(x,y,z3); });
  const BREEZE={8:"8 h : le matin, le vent de terre faiblit.",10:"10 h : la terre chauffe plus vite que la mer.",12:"12 h : la brise de mer se lève et entre dans les terres.",14:"14 h : brise de mer forte sur toute la côte du golfe.",16:"16 h : la brise est à son maximum et pénètre à 10 ou 15 km.",18:"18 h : le soleil baisse, la brise faiblit.",20:"20 h : le vent tombe près de la côte.",22:"22 h : la nuit, une faible brise de terre souffle vers la mer."};
  let bT=null; const bb=document.getElementById("breeze");
  bb.onclick=()=>{ if(bT){ clearInterval(bT); bT=null; bb.textContent="Voir la brise de mer"; cap.hidden=true; return; } let h=8; hs.value=0; hs.oninput(); bb.textContent="Arrêter"; cap.hidden=false;
    const step=()=>{ setHour(h); const k=Object.keys(BREEZE).map(Number).filter(q=>q<=h).pop(); cap.textContent=BREEZE[k]; h++; if(h>22){ clearInterval(bT); bT=null; bb.textContent="Voir la brise de mer"; setTimeout(()=>{ if(!bT) cap.hidden=true; },2500); } };
    step(); bT=setInterval(step,1100); };
  function resize(){ const w=host.clientWidth, h=host.clientHeight; renderer.setSize(w,h,false); cam.aspect=w/h; cam.updateProjectionMatrix(); }
  const sceneObserver=new ResizeObserver(()=>{if(host.clientWidth&&host.clientHeight) resize();}); sceneObserver.observe(host);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", colorTerrain);
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches, tmp=new THREE.Vector3();
  // ---- voyage d'un objet lâché dans le vent
  let seedMode=false, seed=null; const sb=document.getElementById("seedbtn"), st=document.getElementById("seedtype");
  const SEEDS={graine:{n:"la graine",c:0xfff1a8,z0:15}, sable:{n:"le grain de sable",c:0xe0a457,z0:5}, ballon:{n:"le ballon",c:0xff5a5f,z0:10}};
  const TOWARD=["le nord","le nord-est","l'est","le sud-est","le sud","le sud-ouest","l'ouest","le nord-ouest"];
  const NMAX=900, tpos=new Float32Array(NMAX*3), tgeo=new THREE.BufferGeometry(); tgeo.setAttribute("position",new THREE.BufferAttribute(tpos,3)); tgeo.setDrawRange(0,0);
  const trail=new THREE.Line(tgeo,new THREE.LineBasicMaterial({color:0xffd34d})); scene.add(trail);
  const ball=new THREE.Mesh(new THREE.SphereGeometry(.5,20,14),new THREE.MeshBasicMaterial({color:0xfff1a8})); ball.visible=false; scene.add(ball);
  const glow=new THREE.Mesh(new THREE.SphereGeometry(1.1,20,14),new THREE.MeshBasicMaterial({color:0xffd34d, transparent:true, opacity:.25, depthWrite:false})); ball.add(glow);
  const yOf=(x,y,z)=>{ const k=idx(x,y); return (wat[k]?0:elev[k])*EX + Math.max(.15, z*EX*1.4); };
  sb.onclick=()=>{ seedMode=!seedMode; sb.setAttribute("aria-pressed",seedMode); sb.textContent=seedMode?"Touchez le relief…":"Lâcher dans le vent"; cap.hidden=!seedMode; cap.textContent=`Touchez le relief pour lâcher ${SEEDS[st.value].n}.`; };
  function launchSeed(x,y,p){ seedMode=false; sb.setAttribute("aria-pressed",false); sb.textContent="Lâcher encore";
    const T=SEEDS[st.value]; seed={x,y,x0:x,y0:y,z:T.z0,type:st.value,t:0,dist:0,n:0,zmax:T.z0,done:false}; ball.material.color.setHex(T.c); ball.visible=true; tgeo.setDrawRange(0,0); cap.hidden=false; }
  function stepSeed(){ if(!seed || seed.done) return; const dt=6;
    for(let s2=0;s2<(reduce?50:1);s2++){
      const [u,v]=wind(seed.x,seed.y,seed.z,state.hour), sp=Math.hypot(u,v);
      seed.x+=u*dt/1000; seed.y+=v*dt/1000; seed.dist+=sp*dt/1000; seed.t+=dt;
      if(seed.type==="ballon") seed.z=Math.min(1000,seed.z+2.2*dt); else if(seed.type==="graine") seed.z=Math.max(4,Math.min(160,seed.z+(Math.sin(seed.t/240)*.35+.05*(sp-3))*dt)); else seed.z=Math.max(2,Math.min(30,seed.z+(sp>6?.15:-.25)*dt));
      seed.zmax=Math.max(seed.zmax,seed.z);
      if(seed.n<NMAX){ tpos[seed.n*3]=seed.x-W/2; tpos[seed.n*3+1]=yOf(seed.x,seed.y,seed.z); tpos[seed.n*3+2]=-(seed.y-H/2); seed.n++; tgeo.setDrawRange(0,seed.n); tgeo.attributes.position.needsUpdate=true; }
      const out = seed.x<0||seed.x>W||seed.y<0||seed.y>H;
      if(out || seed.t>=3600 || (seed.type==="sable" && sp<2.5 && seed.t>600)){ seed.done=true;
        const dx=seed.x-seed.x0, dy=seed.y-seed.y0, dd=Math.hypot(dx,dy), dir=TOWARD[Math.round(((Math.atan2(dx,dy)*180/Math.PI)+360)%360/45)%8], mins=Math.round(seed.t/60);
        cap.textContent = `${SEEDS[seed.type].n[0].toUpperCase()+SEEDS[seed.type].n.slice(1)} a voyagé ${dd.toFixed(1).replace(".",",")} km vers ${dir} en ${mins} min${out?" (sortie de la zone)":""}, jusqu'à ${Math.round(seed.zmax)} m du sol, près de ${nearestName([Math.min(W,Math.max(0,seed.x)),Math.min(H,Math.max(0,seed.y))])}.`;
        break; } }
    ball.position.set(seed.x-W/2, yOf(seed.x,seed.y,seed.z), -(seed.y-H/2)); glow.scale.setScalar(1+.15*Math.sin(performance.now()/200));
    if(!seed.done) cap.textContent=`${SEEDS[seed.type].n[0].toUpperCase()+SEEDS[seed.type].n.slice(1)} vole depuis ${Math.round(seed.t/60)} min : ${seed.dist.toFixed(1).replace(".",",")} km parcourus, à ${Math.round(seed.z)} m du sol.`; }
  let sceneFrame=0, lastFrame=0;
  const stats={frames:0,renderMs:0};
  const schedule=()=>{if(!sceneFrame&&!document.hidden&&host.offsetParent)sceneFrame=requestAnimationFrame(tick);};
  function tick(now=performance.now()){
    sceneFrame=0;
    if(document.hidden||!host.offsetParent) return;
    if(now-lastFrame<1000/45){schedule();return;}
    lastFrame=now; const start=performance.now();
    if(!ATOM.scenePaused&&!reduce) stepSeed();
    if(!(AW3 && AW3.particules())&&!ATOM.scenePaused) for(let i=0;i<N;i++){ const p=P[i], [u,v]=wind(p.x,p.y,z3,state.hour), sp=Math.hypot(u,v); if(!reduce){ p.x+=u*.02; p.y+=v*.02; } if(++p.a>140||p.x<0||p.x>W||p.y<0||p.y>H){ p.x=Math.random()*W; p.y=Math.random()*H; p.a=0; }
      const k=idx(p.x,p.y), gz=(wat[k]?0:elev[k]); pp[i*3]=p.x-W/2; pp[i*3+1]=gz*EX+Math.max(.15,z3*EX*1.4); pp[i*3+2]=-(p.y-H/2); const c=speedColor(kmh(sp)); pc[i*3]=c[0]/255; pc[i*3+1]=c[1]/255; pc[i*3+2]=c[2]/255; }
    pg.attributes.position.needsUpdate=true; pg.attributes.color.needsUpdate=true; if(AW3) AW3.image(); const moving=ctl.update(); renderer.render(scene,cam); stats.frames++;stats.renderMs=performance.now()-start; if((!ATOM.scenePaused&&!reduce)||moving) schedule();
    const labelRects=[]; const r=renderer.domElement.getBoundingClientRect(); for(const l of labels){ tmp.copy(l.v).project(cam); let vis=tmp.z<1&&Math.abs(tmp.x)<1&&Math.abs(tmp.y)<1;
      const lx=(tmp.x+1)/2*r.width,ly=(1-tmp.y)/2*r.height,lw=Math.min(160,l.d.textContent.length*6.4);
      const bounds={x:lx-lw/2,y:ly-20,w:lw,h:22};
      if(labelRects.some(b=>bounds.x<b.x+b.w&&bounds.x+bounds.w>b.x&&bounds.y<b.y+b.h&&bounds.y+bounds.h>b.y))vis=false;
      if(vis&&!l.cache)labelRects.push(bounds);
      l.d.hidden=!vis||!!l.cache; if(vis){ l.d.style.left=((tmp.x+1)/2*r.width)+"px"; l.d.style.top=((1-tmp.y)/2*r.height)+"px"; } } }
  if(window.ATOM && ATOM.vue3d) try{ AW3=ATOM.vue3d({THREE, scene, cam, ctl, renderer, host, mesh, geo, pos, pts, pg, pp, pc, P, N, labels, objets:OBJ3, pin, hring, hs, H3, NX, NY, G, ex:()=>EX, setEx:v=>{EX=v}, z3:()=>z3, cutObjs:()=>cutObjs, wind, sigma, relClass, idx, elev, wat, W, H, STATIONS, TURB, ANEO, state, speedColor, kmh, fmtSpeed, dirName, fromDeg, nearestName, openSheet, setHour, echelle:K3}); }catch(e){ console.error("vue 3D complète", e); AW3=null; }
  ATOM.sceneStats=()=>({...stats,drawCalls:renderer.info.render.calls,triangles:renderer.info.render.triangles,geometries:renderer.info.memory.geometries,textures:renderer.info.memory.textures,pixelRatio:renderer.getPixelRatio()});
  ATOM.invalidateScene=schedule;
  ctl.addEventListener("change",schedule);
  host.addEventListener("input",schedule); host.addEventListener("click",schedule); host.addEventListener("pointermove",schedule); host.addEventListener("pointerleave",schedule);
  addEventListener("atom:state",schedule); addEventListener("visibilitychange",schedule);
  const visibilityObserver=new IntersectionObserver(entries=>{if(entries[0].isIntersecting)schedule();else{cancelAnimationFrame(sceneFrame);sceneFrame=0;} }); visibilityObserver.observe(host);
  addEventListener("pagehide",()=>{cancelAnimationFrame(sceneFrame);sceneObserver.disconnect();visibilityObserver.disconnect();ctl.dispose();scene.traverse(o=>{o.geometry?.dispose();if(Array.isArray(o.material))o.material.forEach(m=>m.dispose());else o.material?.dispose();});renderer.dispose();},{once:true});
  schedule(); update3dColors=()=>{colorTerrain();schedule();};
}

// ---------------- navigation par onglets ----------------
let threeReady=false; let currentView="monvent";
// ---------------- comptes, rôles et accès ----------------
const ROLES = {
  public:{n:"Grand public", auto:true, home:"monvent", d:"Vent près de chez soi, alertes"},
  agri:{n:"Agriculteur", auto:true, home:"agri", d:"Traitements, irrigation, alertes vent"},
  eolien:{n:"Exploitant éolien", auto:false, home:"eolien", d:"Production prévue, rotor, gisement"},
  admin:{n:"Administrateur", auto:false, home:"admin", d:"Comptes, rôles, clés d'API"}
};
const ACCESS = { monvent:"*", carte3d:"*", alertes:"*", agri:["agri","admin"], tableau:["eolien","admin"], eolien:["eolien","admin"], fiabilite:["eolien","admin"], admin:["admin"], reseau:["admin"] };
const TABNAME = { monvent:"Mon vent", agri:"Agriculteur", tableau:"Tableau de bord", carte3d:"Carte 3D", alertes:"Alertes", eolien:"Éolien", fiabilite:"Fiabilité", admin:"Administration", reseau:"Réseau ANEO" };
const store = { get(k,d){ try{ const v=localStorage.getItem(k); return v?JSON.parse(v):d; }catch(e){ return d; } }, set(k,v){ try{ localStorage.setItem(k,JSON.stringify(v)); }catch(e){} } };
let USERS = store.get("aw-users", null);
if(!Array.isArray(USERS)){ USERS = [
  {id:"u1", name:"Admin ATOM Wind", email:"admin@demo.atomwind.tn", role:"admin", status:"actif", demo:true, created:"2026-10-01"},
  {id:"u2", name:"Hédi, agriculteur à Mornag", email:"agri@demo.atomwind.tn", role:"agri", status:"actif", demo:true, created:"2026-10-05"},
  {id:"u3", name:"Société Vent du Cap", email:"eolien@demo.atomwind.tn", role:"eolien", status:"actif", demo:true, created:"2026-10-08"},
  {id:"u4", name:"Domaine de Grombalia", email:"grombalia@demo.atomwind.tn", role:"public", requested:"eolien", status:"en attente", demo:true, created:"2026-10-20"} ]; store.set("aw-users",USERS); }
USERS = USERS.filter(u=>!(u.demo && (u.role==="rd"||u.requested==="chantier"))); if(!USERS.some(u=>u.demo&&u.role==="eolien")) USERS.push({id:"u3", name:"Société Vent du Cap", email:"eolien@demo.atomwind.tn", role:"eolien", status:"actif", demo:true, created:"2026-10-08"});
USERS.forEach(u=>{ if(!ROLES[u.role]) u.role="public"; if(u.requested && !ROLES[u.requested]) u.requested=null; });
let KEYS = store.get("aw-keys", []); if(!Array.isArray(KEYS)) KEYS=[];
let me = USERS.find(u=>u.id===store.get("aw-session",null) && u.status!=="désactivé") || null;
const roleOf = () => me ? me.role : "visiteur";
const allowed = v => ACCESS[v]==="*" || (Array.isArray(ACCESS[v]) && ACCESS[v].includes(roleOf()));
const toastEl=document.getElementById("toast"); let toastT=null;
function toast(msg){ toastEl.textContent=msg; toastEl.hidden=false; clearTimeout(toastT); toastT=setTimeout(()=>toastEl.hidden=true,3200); }
async function hashPw(salt,pw){ try{ const b=await crypto.subtle.digest("SHA-256", new TextEncoder().encode(salt+":"+pw)); return [...new Uint8Array(b)].map(x=>x.toString(16).padStart(2,"0")).join(""); }catch(e){ let h=0; for(const c of salt+pw) h=(h*31+c.charCodeAt(0))|0; return "f"+h; } }
const esc = t => String(t).replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const initials = n => n.split(/\s+/).filter(Boolean).slice(0,2).map(w=>w[0].toUpperCase()).join("");
function applyAccess(){ document.querySelectorAll("nav.tabs a").forEach(a=>{ a.hidden = !allowed(a.dataset.v); });document.querySelectorAll("[data-access-view]").forEach(a=>{a.hidden=!allowed(a.dataset.accessView);}); renderUserBox(); }
function renderUserBox(){ const ub=document.getElementById("userbox");
  if(!me){ ub.innerHTML=`<button class="btn" id="b-login">Se connecter</button><button class="btn primary" id="b-reg">Créer un compte</button>`; document.getElementById("b-login").onclick=()=>openAuth("login"); document.getElementById("b-reg").onclick=()=>openAuth("register"); return; }
  ub.innerHTML=`<button class="btn" id="b-user" aria-label="Compte de ${esc(me.name)}" aria-haspopup="true" aria-expanded="false"><span class="avatar">${esc(initials(me.name))}</span><span style="text-align:left;line-height:1.15"><span style="display:block">${esc(me.name.split(",")[0])}</span><span class="muted" style="font-size:12px;font-weight:400">${ROLES[me.role].n}${me.requested?" · demande en cours":""}</span></span></button><div class="menu card" id="umenu" hidden>
    <div><b>${esc(me.name)}</b><br><span class="muted" style="font-size:14px">${esc(me.email)}</span></div>
    <div><span class="pill m">${ROLES[me.role].n}</span>${me.requested?` <span class="pill mid">${ROLES[me.requested].n} : en attente de validation</span>`:""}</div>
    <div class="muted" style="font-size:14px">Accès : ${Object.keys(ACCESS).filter(allowed).map(v=>TABNAME[v]).join(", ")}</div>
    <button class="btn" id="b-out">Se déconnecter</button></div>`;
  const bu=document.getElementById("b-user"), mn=document.getElementById("umenu");
  bu.onclick=e=>{ e.stopPropagation(); mn.hidden=!mn.hidden; bu.setAttribute("aria-expanded",!mn.hidden); };
  document.addEventListener("click",e=>{ if(!e.target.closest("#userbox")) mn.hidden=true; },{once:true});
  document.getElementById("b-out").onclick=logout; }
function login(u, msg){ me=u; store.set("aw-session",u.id); applyAccess(); closeAuth(); const home=ROLES[u.role].home; navigate(home,true); toast(msg||`Bienvenue, ${u.name.split(",")[0]}`); }
function logout(){ if(window.ATOM && ATOM.deconnexionServeur) ATOM.deconnexionServeur(); me=null; store.set("aw-session",null); applyAccess(); navigate("monvent",true); toast("Vous êtes déconnecté."); }
const authEl=document.getElementById("authm");
function closeAuth(){ authEl.hidden=true; }
function openAuth(mode){
  const roleOpts = Object.entries(ROLES).filter(([k])=>k!=="admin").map(([k,r])=>`<option value="${k}">${r.n} : ${r.d}${r.auto?"":" (validation par l'équipe)"}</option>`).join("");
  authEl.innerHTML = `<div class="card authcard" role="dialog" aria-modal="true" aria-label="Connexion à ATOM Wind">
    <div class="row" style="justify-content:space-between"><div class="brand" style="margin:0"><svg width="30" height="30" viewBox="0 0 38 38" aria-hidden="true"><circle cx="19" cy="19" r="18" fill="var(--accent)"/><path d="M8 15c6-4 11 3 18-1M8 21c7-4 12 3 22-1M10 27c5-3 9 2 14-1" stroke="var(--accent-ink)" stroke-width="2.4" fill="none" stroke-linecap="round"/></svg><b>ATOM Wind</b></div><button class="x" id="a-close" aria-label="Fermer">×</button></div>
    <div class="tabs2"><button id="t-login" aria-pressed="${mode==="login"}">Se connecter</button><button id="t-reg" aria-pressed="${mode==="register"}">Créer un compte</button></div>
    ${mode==="login" ? `<form class="form" id="f-login" novalidate>
      <label for="l-mail">Adresse e-mail<input class="inp" id="l-mail" type="email" autocomplete="username" required></label>
      <label for="l-pw">Mot de passe<input class="inp" id="l-pw" type="password" autocomplete="current-password" required></label>
      <div class="err" id="l-err" role="alert"></div>
      <button class="btn primary" type="submit">Se connecter</button>
      <button class="btn" type="button" id="l-forgot">Mot de passe oublié ?</button></form>
      <div><b style="font-size:14px">Démonstration : entrer directement en tant que</b><div class="demo" style="margin-top:6px">${["agri","eolien","admin"].map(r=>`<button class="btn" data-demo="${r}">${ROLES[r].n}</button>`).join("")}</div></div>`
    : `<form class="form" id="f-reg" novalidate>
      <label for="r-name">Nom ou raison sociale<input class="inp" id="r-name" autocomplete="name" required></label>
      <label for="r-mail">Adresse e-mail<input class="inp" id="r-mail" type="email" autocomplete="email" required></label>
      <label for="r-pw">Mot de passe (8 caractères au moins, dont un chiffre)<input class="inp" id="r-pw" type="password" autocomplete="new-password" required></label>
      <label for="r-pw2">Confirmer le mot de passe<input class="inp" id="r-pw2" type="password" autocomplete="new-password" required></label>
      <label for="r-role">Je suis<select class="inp" id="r-role">${roleOpts}</select></label>
      <label for="r-place">Mon lieu principal<select class="inp" id="r-place">${PLACES.map((p,i)=>`<option value="${i}">${p[0]}</option>`).join("")}</select></label>
      <label class="check" for="r-ok"><input type="checkbox" id="r-ok"> J'accepte les conditions d'utilisation et la politique de confidentialité d'ATOM Wind.</label>
      <div class="err" id="r-err" role="alert"></div>
      <button class="btn primary" type="submit">Créer mon compte</button></form>`}
    <p class="muted" style="font-size:12px;margin:0">Prototype : les comptes sont enregistrés seulement dans ce navigateur, mot de passe chiffré (empreinte). En production, l'authentification se fait sur le serveur ATOM Wind.</p></div>`;
  authEl.hidden=false;
  document.getElementById("a-close").onclick=closeAuth;
  document.getElementById("t-login").onclick=()=>openAuth("login"); document.getElementById("t-reg").onclick=()=>openAuth("register");
  if(mode==="login"){
    document.getElementById("l-mail").focus();
    document.getElementById("l-forgot").onclick=()=>{ document.getElementById("l-err").textContent="Prototype : en production, un lien de réinitialisation est envoyé par e-mail."; };
    authEl.querySelectorAll("[data-demo]").forEach(b=>b.onclick=()=>{ let u=USERS.find(x=>x.demo && x.role===b.dataset.demo && x.status==="actif"); if(!u){ u={id:"d"+b.dataset.demo, name: ROLES[b.dataset.demo].n, email:`${b.dataset.demo}@demo.atomwind.tn`, role:b.dataset.demo, status:"actif", demo:true, created:new Date().toISOString().slice(0,10)}; USERS.push(u); store.set("aw-users",USERS); } if(u) login(u, `Démonstration : connecté en tant que ${ROLES[u.role].n}`); });
    document.getElementById("f-login").onsubmit=async e=>{ e.preventDefault(); const mail=document.getElementById("l-mail").value.trim().toLowerCase(), pw=document.getElementById("l-pw").value, err=document.getElementById("l-err");
      const u=USERS.find(x=>x.email===mail);
      if(!mail||!pw){ err.textContent="Saisissez votre e-mail et votre mot de passe."; return; }
      if(window.ATOM && ATOM.comptesServeur && ATOM.comptesServeur()){ const r=await ATOM.connexionServeur(mail,pw); if(r.erreur){ err.textContent=r.erreur; return; } const su=ATOM.utilisateurLocal(r.compte,USERS); store.set("aw-users",USERS); login(su); return; }
      if(!u || u.demo || !u.salt || await hashPw(u.salt,pw)!==u.hash){ err.textContent="E-mail ou mot de passe incorrect."; return; }
      if(u.status==="désactivé"){ err.textContent="Ce compte est désactivé. Contactez l'équipe ATOM Wind."; return; }
      login(u); };
  } else {
    document.getElementById("r-name").focus();
    document.getElementById("f-reg").onsubmit=async e=>{ e.preventDefault(); const g=id=>document.getElementById(id), err=g("r-err");
      const name=g("r-name").value.trim(), mail=g("r-mail").value.trim().toLowerCase(), pw=g("r-pw").value, pw2=g("r-pw2").value, role=g("r-role").value;
      if(name.length<2){ err.textContent="Indiquez votre nom."; return; }
      if(!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(mail)){ err.textContent="Cette adresse e-mail n'est pas valide."; return; }
      if(USERS.some(u=>u.email===mail)){ err.textContent="Un compte existe déjà avec cette adresse. Connectez-vous."; return; }
      if(pw.length<8 || !/\d/.test(pw)){ err.textContent="Le mot de passe doit faire 8 caractères au moins, dont un chiffre."; return; }
      if(pw!==pw2){ err.textContent="Les deux mots de passe ne sont pas identiques."; return; }
      if(!g("r-ok").checked){ err.textContent="Acceptez les conditions d'utilisation pour continuer."; return; }
      if(window.ATOM && ATOM.comptesServeur && ATOM.comptesServeur()){ const r=await ATOM.inscriptionServeur(name,mail,pw,role); if(r.erreur){ err.textContent=r.erreur; return; } const c=await ATOM.connexionServeur(mail,pw); if(c.erreur){ err.textContent=c.erreur; return; } const su=ATOM.utilisateurLocal(c.compte,USERS); su.place=+g("r-place").value; store.set("aw-users",USERS); login(su, su.requested ? `Compte créé. Votre accès « ${ROLES[role].n} » sera validé par l'équipe ; en attendant, vous avez l'accès Grand public.` : "Compte créé. Bienvenue sur ATOM Wind !"); return; }
      const salt=[...crypto.getRandomValues(new Uint8Array(12))].map(x=>x.toString(16).padStart(2,"0")).join(""), auto=ROLES[role].auto;
      const u={id:"u"+Date.now(), name, email:mail, role: auto?role:"public", requested: auto?null:role, status: auto?"actif":"en attente", salt, hash:await hashPw(salt,pw), place:+g("r-place").value, created:new Date().toISOString().slice(0,10)};
      USERS.push(u); store.set("aw-users",USERS);
      if(u.role==="agri"){ ag.place=u.place; const ps=document.getElementById("ag-place"); if(ps) ps.value=String(u.place); renderAgri(); }
      login(u, auto ? "Compte créé. Bienvenue sur ATOM Wind !" : `Compte créé. Votre accès « ${ROLES[role].n} » sera validé par l'équipe ; en attendant, vous avez l'accès Grand public.`); };
  }
}

// ---------------- administration ----------------
function renderAdmin(){
  if(!document.getElementById("adusers")) return;
  const n=s=>USERS.filter(u=>u.status===s).length;
  document.getElementById("adkpis").innerHTML=[["Comptes",USERS.length],["Actifs",n("actif")],["En attente de validation",n("en attente")],["Clés d'API actives",KEYS.filter(k=>!k.revoked).length]].map(([k,v])=>`<div class="card kpi"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  document.getElementById("adusers").innerHTML = `<thead><tr><th>Nom</th><th>E-mail</th><th>Rôle</th><th>Statut</th><th>Actions</th></tr></thead><tbody>${USERS.map(u=>`<tr>
    <td><b>${esc(u.name)}</b><br><span class="muted" style="font-size:13px">créé le ${u.created}</span></td><td>${esc(u.email)}</td>
    <td><select data-role="${u.id}" aria-label="Rôle de ${esc(u.name)}" ${u.id===me?.id?"disabled":""}>${Object.entries(ROLES).map(([k,r])=>`<option value="${k}" ${k===u.role?"selected":""}>${r.n}</option>`).join("")}</select>${u.requested?`<br><span class="pill mid" style="margin-top:4px">demande : ${ROLES[u.requested].n}</span>`:""}</td>
    <td><span class="pill ${u.status==="actif"?"good":u.status==="en attente"?"mid":"bad"}">${u.status}</span></td>
    <td><div class="row" style="gap:6px">${u.requested?`<button class="btn" data-ok="${u.id}">Valider</button>`:""}${u.id!==me?.id?`<button class="btn" data-off="${u.id}">${u.status==="désactivé"?"Réactiver":"Désactiver"}</button>`:`<span class="muted" style="font-size:13px">vous</span>`}</div></td></tr>`).join("")}</tbody>`;
  document.querySelectorAll("[data-role]").forEach(sl=>sl.onchange=()=>{ const u=USERS.find(x=>x.id===sl.dataset.role); u.role=sl.value; u.requested=null; if(u.status==="en attente") u.status="actif"; store.set("aw-users",USERS); renderAdmin(); toast(`Rôle de ${u.name} : ${ROLES[u.role].n}`); });
  document.querySelectorAll("[data-ok]").forEach(b=>b.onclick=()=>{ const u=USERS.find(x=>x.id===b.dataset.ok); u.role=u.requested; u.requested=null; u.status="actif"; store.set("aw-users",USERS); renderAdmin(); toast(`Accès « ${ROLES[u.role].n} » validé pour ${u.name}`); });
  document.querySelectorAll("[data-off]").forEach(b=>b.onclick=()=>{ const u=USERS.find(x=>x.id===b.dataset.off); u.status = u.status==="désactivé" ? "actif" : "désactivé"; store.set("aw-users",USERS); renderAdmin(); toast(`${u.name} : compte ${u.status}`); });
  const tabs=Object.keys(ACCESS);
  document.getElementById("adroles").innerHTML = `<thead><tr><th>Rôle</th>${tabs.map(t=>`<th style="font-size:13px">${TABNAME[t]}</th>`).join("")}</tr></thead><tbody>${[["visiteur","Visiteur (sans compte)"],...Object.entries(ROLES).map(([k,r])=>[k,r.n])].map(([k,nm])=>`<tr><td><b>${nm}</b></td>${tabs.map(t=>{ const ok=ACCESS[t]==="*"||ACCESS[t].includes(k); return `<td style="text-align:center">${ok?'<span class="pill good">oui</span>':'<span class="muted">—</span>'}</td>`; }).join("")}</tr>`).join("")}</tbody>`;
  document.getElementById("k-list").innerHTML = KEYS.length ? KEYS.map((k,i)=>`<div class="it"><span style="min-width:0"><b>${esc(k.name)}</b> <span class="pill ${k.revoked?"bad":"good"}">${k.revoked?"révoquée":"active"}</span><br><span class="muted" style="font-size:13px;font-family:var(--f-data)">${k.prefix}… · ${k.scope} · créée le ${k.created}</span></span>${k.revoked?"":`<button class="btn" data-rev="${i}">Révoquer</button>`}</div>`).join("") : `<span class="muted">Aucune clé pour l'instant.</span>`;
  document.querySelectorAll("[data-rev]").forEach(b=>b.onclick=()=>{ KEYS[+b.dataset.rev].revoked=true; store.set("aw-keys",KEYS); renderAdmin(); toast("Clé révoquée : elle ne fonctionne plus."); });
}
document.getElementById("k-add").onclick=()=>{ const name=document.getElementById("k-name").value.trim()||"Clé sans nom", scope=document.getElementById("k-scope").value;
  const raw="aw_"+[...crypto.getRandomValues(new Uint8Array(20))].map(x=>x.toString(16).padStart(2,"0")).join("");
  KEYS.push({name, scope, prefix:raw.slice(0,10), created:new Date().toISOString().slice(0,10), revoked:false}); store.set("aw-keys",KEYS);
  document.getElementById("k-new").innerHTML=`<div class="answer go" style="padding:12px;margin-top:10px"><b>Copiez cette clé maintenant : elle ne sera plus affichée.</b><div class="keybox" id="k-raw">${raw}</div><button class="btn" id="k-copy">Copier</button></div>`;
  document.getElementById("k-copy").onclick=()=>{ navigator.clipboard?.writeText(raw).then(()=>toast("Clé copiée."),()=>{ const r=document.createRange(); r.selectNodeContents(document.getElementById("k-raw")); getSelection().removeAllRanges(); getSelection().addRange(r); toast("Sélectionnée : copiez-la avec Ctrl+C."); }); };
  document.getElementById("k-name").value=""; renderAdmin(); };

function show(v){ if(!document.getElementById("v-"+v)) v="monvent";
  if(!allowed(v)){ toast(me ? `L'espace « ${TABNAME[v]} » n'est pas inclus dans votre rôle (${ROLES[me.role].n}).` : `Connectez-vous pour accéder à l'espace « ${TABNAME[v]} ».`); if(!me) openAuth("login"); v = me ? ROLES[me.role].home : "monvent"; }
  if(v==="admin") renderAdmin();if(v==="tableau")renderTB();
  // accroche « reseau-aneo » : onglet d'administration Réseau ANEO (engine/reseau-aneo.js, habillage provisoire)
  if(v==="reseau" && window.ATOM?.reseauAneo) ATOM.reseauAneo.afficher(); document.querySelectorAll("section.view").forEach(s=>s.classList.toggle("on",s.id==="v-"+v)); document.querySelectorAll("nav.tabs a").forEach(a=>a.setAttribute("aria-current",a.dataset.v===v?"page":"false")); sheet.hidden=true; if(v==="carte3d")setSceneMode(sceneMode); currentView=v; if(window.ATOM?.workspace) publishWorkspace(); }
function navigate(v,replace=false){
  show(v);
  try{if(location.hash!=="#"+currentView)history[replace?"replaceState":"pushState"](null,"","#"+currentView);}catch(_){}
  window.scrollTo({top:0,behavior:matchMedia("(prefers-reduced-motion: reduce)").matches?"instant":"smooth"});
}
addEventListener("hashchange",()=>{show(location.hash.slice(1));try{if(location.hash!=="#"+currentView)history.replaceState(null,"","#"+currentView);}catch(_){} });
// les liens internes sont gérés par la page elle-même : la visionneuse ne transmet pas toujours le changement d'ancre
document.addEventListener("click", e => { if(e.defaultPrevented||e.button!==0||e.metaKey||e.ctrlKey||e.shiftKey||e.altKey)return;const a=e.target.closest('a[href^="#"]'); if(!a) return; const v=a.getAttribute("href").slice(1); if(!document.getElementById("v-"+v)) return; e.preventDefault();navigate(v); });

// ---------------- « Pourquoi ce vent ? » ----------------
function explainHtml(x,y,z,h){ return explainHtml0(x,y,z,h) + (window.ATOM && ATOM.htmlInfluence ? ATOM.htmlInfluence(x,y) : ""); }
function explainHtml0(x,y,z,h){
  const k=idx(x,y), [bu,bv]=synoptic(h), B=breeze(h), sd=sdist[k], zr=z0[k], zz=Math.max(z,1);
  const low=Math.log(Math.min(zz,150)/zr)/Math.log(10/zr), reach = sd>0 ? Math.exp(-sd/13) : Math.exp(sd/9);
  const relief=Math.max(-.25,Math.min(.45,.0016*(elev[k]-smooth[k])))*Math.exp(-z/400);
  const syn=Math.hypot(bu,bv)*Math.max(.3,low), brz=Math.abs(B)*reach*Math.max(.6,low*.8)*Math.exp(-z/380);
  const L=[];
  L.push(`<b>La météo générale</b> amène un vent ${dirName(bu,bv)} d'environ ${fmtSpeed(syn)} à cette hauteur.`);
  if(B>.5 && brz>.4) L.push(`<b>La brise de mer</b> ajoute environ ${fmtSpeed(brz)} : l'après-midi, la terre chauffe plus vite que la mer et l'air frais de la mer entre dans les terres. Ici, elle vient ${dirName(gx[k],gy[k])} ; ${sd>0?`vous êtes à ${Math.round(sd)} km de la côte`:"vous êtes en mer"}.`);
  else if(B<0 && brz>.2) L.push(`<b>La brise de terre</b> : la nuit, la terre refroidit plus vite que la mer, et un léger vent descend vers la mer.`);
  else if(B>.5 && sd>20) L.push(`<b>Pas de brise de mer ici</b> : vous êtes à ${Math.round(sd)} km de la côte, trop loin pour la sentir.`);
  if(relief>.04) L.push(`<b>Le relief accélère le vent</b> d'environ ${Math.round(relief*100)} % : vous êtes sur une hauteur (sol à ${Math.round(elev[k])} m), le vent s'y engouffre.`);
  else if(relief<-.04) L.push(`<b>Le relief abrite</b> : vous êtes dans un creux, le vent est freiné d'environ ${Math.round(-relief*100)} %.`);
  if(zr>.5 && z<100) L.push(`<b>Bâtiments ou arbres</b> : le sol rugueux freine le vent près du sol.`);
  else if(zr<.01) L.push(`<b>Sur l'eau</b>, rien ne freine le vent : il est plus régulier qu'à terre.`);
  if(z<100){ const s100=Math.hypot(...wind(x,y,100,h)); L.push(`<b>La hauteur compte</b> : à 100 m du sol, le vent souffle à ${fmtSpeed(s100)}. Plus on monte, moins le sol le freine.`); }
  return `<div class="why"><ul>${L.map(t=>`<li>${t}</li>`).join("")}</ul><span class="src">Explication simplifiée, tirée des composantes du calcul.</span></div>`;
}

// ---------------- le vent qui chante ----------------
let audio=null; const sbtn=document.getElementById("b-sound");
function soundToggle(){
  if(audio){ try{ audio.ctx.close(); }catch(e){} audio=null; sbtn.setAttribute("aria-pressed",false); sbtn.innerHTML="Écouter le vent"; return; }
  const AC=window.AudioContext||window.webkitAudioContext; if(!AC){ toast("Le son n'est pas disponible sur cet appareil."); return; }
  const ctx=new AC(), len=ctx.sampleRate*3, buf=ctx.createBuffer(2,len,ctx.sampleRate);
  for(let c=0;c<2;c++){ const d=buf.getChannelData(c); let last=0; for(let i=0;i<len;i++){ last=(last+.02*(Math.random()*2-1))/1.02; d[i]=last*3.2; } }
  const src=ctx.createBufferSource(); src.buffer=buf; src.loop=true;
  const bp=ctx.createBiquadFilter(); bp.type="bandpass"; bp.Q.value=.7;
  const hi=ctx.createBiquadFilter(); hi.type="highpass"; hi.frequency.value=90;
  const gain=ctx.createGain(); gain.gain.value=0; const master=ctx.createGain(); master.gain.value=.8;
  const lfo=ctx.createOscillator(), lfoG=ctx.createGain(); lfo.frequency.value=.13; lfoG.gain.value=0; lfo.connect(lfoG); lfoG.connect(gain.gain);
  const pan=ctx.createStereoPanner ? ctx.createStereoPanner() : null;
  src.connect(hi); hi.connect(bp); bp.connect(gain); if(pan){ gain.connect(pan); pan.connect(master); } else gain.connect(master); master.connect(ctx.destination);
  src.start(); lfo.start(); audio={ctx,bp,gain,lfoG,pan,lfo};
  sbtn.setAttribute("aria-pressed",true); sbtn.innerHTML='<span class="wave" aria-hidden="true"><i></i><i></i><i></i><i></i></span> Couper le son'; updateSound();
  toast("Le son suit le vent : plus fort quand il forcit, à gauche ou à droite selon sa direction.");
}
function updateSound(){ if(!audio) return; const [x,y]=state.pin, [u,v]=wind(x,y,ACTS[state.act].z,state.hour), sp=Math.hypot(u,v), t=audio.ctx.currentTime;
  const g=Math.min(.55,.03+sp*.04); audio.gain.gain.setTargetAtTime(g,t,.5); audio.lfoG.gain.setTargetAtTime(g*.4,t,.5); audio.lfo.frequency.setTargetAtTime(.08+sp*.025,t,.5);
  audio.bp.frequency.setTargetAtTime(220+sp*85,t,.5); if(audio.pan) audio.pan.pan.setTargetAtTime(Math.max(-.9,Math.min(.9,u/7)),t,.5); }
sbtn.onclick=soundToggle;

// ---------------- info-bulle, survol des graphiques, compteurs ----------------
const gtip=document.getElementById("gtip"); let tipOwner=null;
function showTip(html,cx,cy,owner){ tipOwner=owner; gtip.innerHTML=html; gtip.hidden=false; const w=gtip.offsetWidth, h=gtip.offsetHeight; let x=cx+16, y=cy+16; if(x+w>innerWidth-8) x=cx-w-16; if(y+h>innerHeight-8) y=cy-h-16; gtip.style.left=Math.max(8,x)+"px"; gtip.style.top=Math.max(8,y)+"px"; }
function hideTip(owner){ if(!owner || owner===tipOwner){ gtip.hidden=true; tipOwner=null; } }
document.addEventListener("mousemove", e => { const el=e.target.closest && e.target.closest("[data-h]"); if(el) showTip(el.getAttribute("data-h"), e.clientX, e.clientY, "data"); else if(tipOwner==="data") hideTip("data"); });
document.addEventListener("scroll", () => hideTip(), {passive:true});
function hoverChart(id, Wd, pl, pr, html){ const box=document.getElementById(id), svg=box && box.querySelector("svg"); if(!svg) return; const hl=svg.querySelector(".hl");
  const hourAt=e=>{ const r=svg.getBoundingClientRect(), xv=(e.clientX-r.left)/r.width*Wd; return Math.max(0,Math.min(23,Math.round((xv-pl)/(Wd-pl-pr)*23))); };
  svg.style.cursor="pointer";svg.setAttribute("tabindex","0");svg.setAttribute("role","group");
  svg.setAttribute("aria-label",(svg.getAttribute("aria-label")||"Courbe horaire")+". Flèches gauche et droite pour changer d’heure.");
  svg.addEventListener("keydown",e=>{if(e.key==="ArrowLeft"||e.key==="ArrowRight"){e.preventDefault();setHour(Math.max(0,Math.min(23,state.hour+(e.key==="ArrowRight"?1:-1))));document.getElementById(id)?.querySelector("svg")?.focus();}});
  svg.addEventListener("mousemove", e=>{ const h=hourAt(e), x=pl+h*(Wd-pl-pr)/23; if(hl){ hl.setAttribute("x1",x); hl.setAttribute("x2",x); hl.setAttribute("visibility","visible"); } showTip(html(h), e.clientX, e.clientY, "chart"); });
  svg.addEventListener("mouseleave", ()=>{ if(hl) hl.setAttribute("visibility","hidden"); hideTip("chart"); });
  svg.addEventListener("click", e=>setHour(hourAt(e))); }
const KPIPREV={};
function animateKpis(id){ const box=document.getElementById(id); if(!box || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  box.querySelectorAll(".v").forEach((el,i)=>{ const m=el.textContent.match(/^([\d\s ]*\d(?:,\d+)?)(.*)$/); if(!m) return; const txt=m[1].replace(/[\s ]/g,""), dec=(txt.split(",")[1]||"").length, to=parseFloat(txt.replace(",",".")), key=id+i, from=KPIPREV[key]??0; KPIPREV[key]=to; if(from===to) return;
    const t0=performance.now(), suf=m[2]; const step=t=>{ const a=Math.min(1,(t-t0)/550), e=1-(1-a)**3, v=from+(to-from)*e; el.textContent=(dec?v.toFixed(dec).replace(".",","):Math.round(v).toLocaleString("fr-FR"))+suf; if(a<1) requestAnimationFrame(step); }; requestAnimationFrame(step); }); }

// ---------------- thème ----------------
function setTheme(t){ if(t==="auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme=t; ["auto","light","dark"].forEach(k=>document.getElementById("th-"+k).setAttribute("aria-pressed",k===t)); store.set("aw-theme",t); MAPS.forEach(m=>m.redraw()); renderAnswer(); renderTB(); renderEol(); }
["auto","light","dark"].forEach(k=>document.getElementById("th-"+k).onclick=()=>setTheme(k));

// ---------------- recherche rapide (Ctrl K) ----------------
const pal=document.getElementById("palette"); let palSel=0, palItems=[];
const norm=t=>t.normalize("NFD").replace(/[̀-ͯ]/g,"").toLowerCase();
function palAll(){ const L=[];
  PLACES.forEach((p,i)=>L.push({t:p[0], k:"Lieu", go:()=>{ state.pin=xy(p[1],p[2]); state.pinName=p[0]; townSel.value=String(i); navigate("monvent"); renderAnswer(); }}));
  Object.entries(ACTS).forEach(([k,a])=>L.push({t:a.n, k:"Activité", go:()=>{ state.act=k; document.querySelectorAll(".act").forEach(x=>x.setAttribute("aria-pressed",x.dataset.a===k)); navigate("monvent"); renderAnswer(); }}));
  Object.keys(ACCESS).filter(allowed).forEach(v=>L.push({t:TABNAME[v], k:"Espace", go:()=>navigate(v)}));
  L.push({t:"Voir la brise de mer en 3D", k:"Action", go:()=>{ navigate("carte3d"); setTimeout(()=>document.getElementById("breeze").click(),900); }});
  L.push({t:"Rejouer la journée", k:"Action", go:()=>document.getElementById("play").click()});
  L.push({t:"Thème sombre", k:"Action", go:()=>setTheme("dark")}); L.push({t:"Thème clair", k:"Action", go:()=>setTheme("light")});
  if(!me) L.push({t:"Se connecter", k:"Compte", go:()=>openAuth("login")}); else L.push({t:"Se déconnecter", k:"Compte", go:logout});
  return L; }
function palRender(q){ const all=palAll(), nq=norm(q.trim()); palItems = nq ? all.filter(x=>norm(x.t+" "+x.k).includes(nq)) : all; palSel=Math.min(palSel,Math.max(0,palItems.length-1));
  pal.querySelector("ul").innerHTML = palItems.length ? palItems.slice(0,40).map((x,i)=>`<li role="option" aria-selected="${i===palSel}" data-i="${i}"><span>${esc(x.t)}</span><small>${x.k}</small></li>`).join("") : `<li><span class="muted">Aucun résultat</span></li>`;
  pal.querySelectorAll("li[data-i]").forEach(li=>{ li.onclick=()=>palRun(+li.dataset.i); li.onmousemove=()=>{ if(palSel!==+li.dataset.i){ palSel=+li.dataset.i; palRender(pal.querySelector("input").value); } }; }); }
function palRun(i){ const it=palItems[i]; palClose(); if(it) it.go(); }
function palOpen(){ pal.innerHTML=`<div class="card" role="dialog" aria-modal="true" aria-label="Recherche rapide"><input id="pal-in" placeholder="Un lieu, une activité, un espace… (ex. Nabeul, traiter, éolien)" autocomplete="off"><ul role="listbox"></ul><div class="muted" style="font-size:12px">Flèches pour choisir, Entrée pour ouvrir, Échap pour fermer</div></div>`; pal.hidden=false; palSel=0; const inp=document.getElementById("pal-in"); palRender(""); inp.focus();
  inp.oninput=()=>{ palSel=0; palRender(inp.value); };
  inp.onkeydown=e=>{ if(e.key==="ArrowDown"){ palSel=Math.min(palItems.length-1,palSel+1); palRender(inp.value); e.preventDefault(); } else if(e.key==="ArrowUp"){ palSel=Math.max(0,palSel-1); palRender(inp.value); e.preventDefault(); } else if(e.key==="Enter"){ palRun(palSel); } else if(e.key==="Escape"){ palClose(); } };
  pal.onclick=e=>{ if(e.target===pal) palClose(); }; }
function palClose(){ pal.hidden=true; }
document.getElementById("b-search").onclick=palOpen;
document.addEventListener("keydown", e=>{ if((e.ctrlKey||e.metaKey) && e.key.toLowerCase()==="k"){ e.preventDefault(); pal.hidden?palOpen():palClose(); } else if(e.key==="/" && !/input|select|textarea/i.test(document.activeElement.tagName) && pal.hidden){ e.preventDefault(); palOpen(); } else if(e.key==="Escape"){ palClose(); closeAuth(); } });

// ---------------- visite guidée ----------------
const TOUR=[["Le vent partout, même sans capteur",DOM ? "ATOM Wind calcule le vent sur toute la Tunisie, du sol jusqu'à 1 000 m, à partir des stations météo et de la météo de grande échelle." : "ATOM Wind calcule le vent sur tout le Grand Tunis, du sol jusqu'à 1 000 m, à partir des stations météo et de la météo de grande échelle.",'<svg width="200" height="90" viewBox="0 0 200 90"><path d="M10 40c40-25 70 15 110-5s60 5 70 0M10 60c40-20 80 15 120-5s40 5 60 0" stroke="var(--accent)" stroke-width="4" fill="none" stroke-linecap="round"/></svg>'],
  ["Une question, une réponse","Placez votre position, dites ce que vous voulez faire, et ATOM Wind répond tout de suite : vert, vous pouvez ; orange, attention ; rouge, c'est déconseillé.",'<svg width="200" height="90" viewBox="0 0 200 90"><circle cx="50" cy="45" r="22" fill="var(--go)"/><circle cx="100" cy="45" r="22" fill="var(--warn)"/><circle cx="150" cy="45" r="22" fill="var(--stop)"/></svg>'],
  ["Toujours dire si c'est sûr","Chaque valeur indique sa fiabilité. Loin des stations, elle est plus faible : la carte devient plus pâle et l'erreur probable est affichée.",'<svg width="200" height="90" viewBox="0 0 200 90"><rect x="20" y="25" width="50" height="40" rx="8" fill="var(--go)"/><rect x="75" y="25" width="50" height="40" rx="8" fill="var(--warn)" opacity=".7"/><rect x="130" y="25" width="50" height="40" rx="8" fill="var(--stop)" opacity=".35"/></svg>']];
const tourEl=document.getElementById("tour");
function tour(i=0){ if(i>=TOUR.length){ tourEl.hidden=true; try{ localStorage.setItem("aw-tour","1"); }catch(e){} return; } const [t,d,ill]=TOUR[i];
  tourEl.innerHTML=`<div class="card" role="dialog" aria-modal="true" aria-label="Visite guidée"><div class="ill">${ill}</div><h2>${t}</h2><p style="margin:0">${d}</p><div class="row"><div class="dots">${TOUR.map((_,j)=>`<i class="${j===i?"on":""}"></i>`).join("")}</div><span style="margin-left:auto"></span><button class="btn" id="tskip">Passer</button><button class="btn primary" id="tnext">${i===TOUR.length-1?"Commencer":"Suivant"}</button></div></div>`;
  tourEl.hidden=false; document.getElementById("tnext").onclick=()=>tour(i+1); document.getElementById("tskip").onclick=()=>tour(TOUR.length); document.getElementById("tnext").focus(); }
document.getElementById("retour").onclick=()=>tour(0);

// ---------------- vraie carte (Leaflet + OpenStreetMap) ----------------
const toLL = (x,y) => [LAT0 + y/KY, LON0 + x/KX];
const toXY = (lat,lon) => xy(lat,lon);
let lmap=null, lpin=null, lwind=null, tilesFailed=false,leafletPromise=null,leafletStarting=false;
async function loadLeaflet(){
  if(window.L)return;
  if(!leafletPromise)leafletPromise=new Promise((resolve,reject)=>{
    const script=document.createElement("script"),timer=setTimeout(()=>fail(),12000);
    const fail=()=>{clearTimeout(timer);script.remove();leafletPromise=null;reject(new Error("Local map library unavailable"));};
    script.src=import.meta.env.BASE_URL+"vendor/leaflet/leaflet.js";script.async=true;script.onerror=fail;
    script.onload=()=>{clearTimeout(timer);if(!window.L){fail();return;}window.L.Icon.Default.imagePath=import.meta.env.BASE_URL+"vendor/leaflet/images/";resolve();};
    document.head.appendChild(script);
  });
  await leafletPromise;
}
function reliefDataURL(){ const RW=DOM?Math.max(120,Math.round(420*W/Math.max(W,H))):340, RH=DOM?Math.max(120,Math.round(420*H/Math.max(W,H))):280, c=document.createElement("canvas"); c.width=RW; c.height=RH; const g=c.getContext("2d"), img=g.createImageData(RW,RH);
  for(let j=0;j<RH;j++) for(let i=0;i<RW;i++){ const x=(i+.5)/RW*W, y=(1-(j+.5)/RH)*H, k=idx(x,y), o=(j*RW+i)*4; let col, a=150;
    if(wat[k]){ col=[0,0,0]; a=0; } else { const t=Math.min(1,elev[k]/700); col=[150+40*t, 185-55*t, 120-30*t]; const kE=idx(x+.4,y), sh=Math.max(-.3,Math.min(.3,(elev[kE]-elev[k])*.004)); col=col.map(q=>q*(1-sh)); a=Math.round(70+120*Math.min(1,t*1.6)); }
    img.data[o]=col[0]; img.data[o+1]=col[1]; img.data[o+2]=col[2]; img.data[o+3]=a; }
  g.putImageData(img,0,0); return c.toDataURL(); }
function makeWindLayer(){ return L.Layer.extend({
  onAdd(map){ this._map=map; this._c=L.DomUtil.create("canvas","leaflet-zoom-hide"); this._c.style.pointerEvents="none"; map.getPanes().overlayPane.appendChild(this._c); this._parts=[]; map.on("moveend zoomend resize", this._reset, this); map.on("zoomstart movestart", this._clear, this); this._reset(); this._run=true; const f=()=>{ if(!this._run) return; this._frame(); requestAnimationFrame(f); }; requestAnimationFrame(f); },
  onRemove(map){ this._run=false; map.off("moveend zoomend resize", this._reset, this); map.off("zoomstart movestart", this._clear, this); this._c.remove(); },
  _clear(){ const g=this._c.getContext("2d"); g.clearRect(0,0,this._c.width,this._c.height); this._moving=true; },
  _reset(){ const m=this._map, sz=m.getSize(), dpr=Math.min(2,devicePixelRatio||1); this._dpr=dpr; L.DomUtil.setPosition(this._c, m.containerPointToLayerPoint([0,0])); this._c.width=sz.x*dpr; this._c.height=sz.y*dpr; this._c.style.width=sz.x+"px"; this._c.style.height=sz.y+"px"; this._moving=false;
    const b=m.getBounds(), sw=toXY(b.getSouth(),b.getWest()), ne=toXY(b.getNorth(),b.getEast()); this._box=[Math.max(0,sw[0]),Math.max(0,sw[1]),Math.min(W,ne[0]),Math.min(H,ne[1])];
    const n=Math.round(Math.min(2200, sz.x*sz.y/420)); this._parts=Array.from({length:n},()=>this._np()); },
  _np(){ const [x0,y0,x1,y1]=this._box; return {x:x0+Math.random()*Math.max(.1,x1-x0), y:y0+Math.random()*Math.max(.1,y1-y0), a:Math.random()*60|0}; },
  _frame(){ if(this._moving || !this._map || !document.getElementById("lmapbox").offsetParent) return; const g=this._c.getContext("2d"), dpr=this._dpr, m=this._map, z=ACTS[state.act].z, dt=.055*Math.pow(2,10-m.getZoom());
    g.globalCompositeOperation="destination-out"; g.fillStyle="rgba(0,0,0,.1)"; g.fillRect(0,0,this._c.width,this._c.height); g.globalCompositeOperation="source-over"; g.lineWidth=2*dpr; g.lineCap="round";
    for(const p of this._parts){ const [u,v]=wind(p.x,p.y,z,state.hour), nx=p.x+u*dt, ny=p.y+v*dt, a=m.latLngToContainerPoint(toLL(p.x,p.y)), b2=m.latLngToContainerPoint(toLL(nx,ny)), c=speedColor(kmh(Math.hypot(u,v)));
      g.strokeStyle=`rgb(${c[0]},${c[1]},${c[2]})`; g.beginPath(); g.moveTo(a.x*dpr,a.y*dpr); g.lineTo(b2.x*dpr,b2.y*dpr); g.stroke();
      p.x=nx; p.y=ny; const [x0,y0,x1,y1]=this._box; if(++p.a>80 || p.x<x0||p.x>x1||p.y<y0||p.y>y1) Object.assign(p,this._np(),{a:0}); } }
}); }
function pinIcon(){ return L.divIcon({className:"", html:'<div class="pinicon"></div>', iconSize:[26,34], iconAnchor:[13,34]}); }
function syncLeafletPin(pan){ if(!lmap) return; const ll=toLL(...state.pin); if(!lpin) lpin=L.marker(ll,{icon:pinIcon(), keyboard:false}).addTo(lmap); else lpin.setLatLng(ll); if(pan) lmap.panTo(ll); }
async function initLeaflet(){
  if(leafletStarting||lmap)return;leafletStarting=true;
  try{ await loadLeaflet(); }catch(e){leafletStarting=false;setBasemap("relief");toast("Le fond de carte est indisponible. Le relief reste affiché ; vous pouvez réessayer « Vraie carte ».");console.error(e);return;}
  const bounds=L.latLngBounds(toLL(0,0), toLL(W,H));
  lmap=L.map("lmap",{zoomControl:true, minZoom:DOM ? 5 : 8, maxZoom:16, maxBounds:bounds.pad(.4), zoomSnap:.25}).fitBounds(bounds,{padding:[4,4]});leafletStarting=false;
  const tiles=L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",{maxZoom:19, attribution:'© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'}).addTo(lmap);
  tiles.on("tileerror",()=>{ if(tilesFailed) return; tilesFailed=true; const n=document.getElementById("lnotice"); n.hidden=false; n.innerHTML='Les rues OpenStreetMap sont indisponibles. Le vent reste visible ; vous pouvez choisir le fond Relief.<button aria-label="Fermer">×</button>'; n.querySelector("button").onclick=ev=>{ ev.stopPropagation(); n.hidden=true; }; });
  L.imageOverlay(reliefDataURL(), bounds, {opacity:.75, interactive:false}).addTo(lmap);
  L.rectangle(bounds,{color:"#0a6e7d", weight:2, fill:false, dashArray:"6 6", interactive:false}).addTo(lmap);
  for(const st of STATIONS) L.circleMarker(toLL(...st.p),{radius:7, color:"#fff", weight:2, fillColor:"#3a6fd2", fillOpacity:1}).addTo(lmap).bindTooltip(`Station ${st.n} (INM)`);
  for(const p of PLACES){ const ll=[p[1],p[2]]; L.circleMarker(ll,{radius:3, color:"#14232b", weight:1, fillColor:"#fff", fillOpacity:1, interactive:false}).addTo(lmap).bindTooltip(p[0].split(" (")[0],{permanent:true, direction:"right", offset:[4,0], className:"llbl"}); }
  const lblZoom=()=>document.getElementById("lmap").classList.toggle("lmap-nolbl", lmap.getZoom() < (DOM ? 7 : innerWidth<700?11:10)); lmap.on("zoomend", lblZoom); lblZoom();
  lwind=new (makeWindLayer())(); lwind.addTo(lmap); syncLeafletPin();
  lmap.on("click", e=>{ const [x,y]=toXY(e.latlng.lat,e.latlng.lng); if(x<0||x>W||y<0||y>H){ toast(DOM ? `Ce point est en dehors de la zone calculée (${ZONE}).` : "Ce point est en dehors de la zone calculée (Grand Tunis)."); return; } placePin(x,y); });
  lmap.on("mousemove", e=>{ const [x,y]=toXY(e.latlng.lat,e.latlng.lng); if(x<0||x>W||y<0||y>H){ hideTip("map"); return; } const z=ACTS[state.act].z, k=idx(x,y), [u,v]=wind(x,y,z,state.hour), deg=(fromDeg(u,v)+180)%360, [,rl]=relClass(sigma(x,y).s);
    showTip(`<div style="display:flex;gap:8px;align-items:center"><b>${fmtSpeed(Math.hypot(u,v))}</b>${arrowSvg(deg,18,"currentColor")}</div>Vent ${dirName(u,v)}, à ${z} m du sol<br><span class="sub">${wat[k]===1?"En mer":`Sol à ${Math.round(elev[k])} m`} · fiabilité ${rl} · près de ${nearestName([x,y])}</span>`, e.originalEvent.clientX, e.originalEvent.clientY, "map"); });
  lmap.on("mouseout", ()=>hideTip("map"));
  lmap.on("locationfound", e=>{ const [x,y]=toXY(e.latlng.lat,e.latlng.lng); if(x<0||x>W||y<0||y>H){ toast(DOM ? "Vous êtes en dehors de la zone calculée : ATOM Wind ne calcule pas encore votre zone." : "Vous êtes en dehors du Grand Tunis : ATOM Wind ne calcule pas encore votre zone."); return; } placePin(x,y); syncLeafletPin(true); toast("Position trouvée."); });
  lmap.on("locationerror", ()=>toast("Position indisponible : autorisez la localisation ou choisissez votre ville dans la liste."));
}
function setBasemap(b){ const osm=b==="osm"; document.getElementById("bm-relief").setAttribute("aria-pressed",!osm); document.getElementById("bm-osm").setAttribute("aria-pressed",osm);
  document.getElementById("map1").hidden=osm; document.getElementById("lmapbox").hidden=!osm; document.getElementById("locate").hidden=!osm; store.set("aw-basemap",b);
  if(osm){ if(!lmap) initLeaflet(); else setTimeout(()=>lmap.invalidateSize(),50); } }
document.getElementById("bm-relief").onclick=()=>setBasemap("relief"); document.getElementById("bm-osm").onclick=()=>setBasemap("osm");
document.getElementById("locate").onclick=()=>{ if(lmap) lmap.locate({setView:false, maxZoom:14, enableHighAccuracy:true}); };
if(store.get("aw-basemap","relief")==="osm") setTimeout(()=>setBasemap("osm"),0);

// ---------------- en-tête compact au défilement ----------------
const topbar=document.getElementById("topbar"); let compact=false;
addEventListener("scroll", () => { const y=scrollY; if(!compact && y>140){ compact=true; topbar.classList.add("compact"); } else if(compact && y<30){ compact=false; topbar.classList.remove("compact"); } }, {passive:true});

// ---------------- démarrage ----------------
function publishWorkspace(){
  const z=currentView==="carte3d"?z3:ACTS[state.act].z;
  const heightReference=currentView==="carte3d"&&ATOM.sceneHeightMode?.()==="alt"?"asl":"agl";
  const sampleHeight=heightReference==="asl"?ATOM.sceneSampleHeight(...state.pin):z;
  const [u,v]=sampleHeight>=2?wind(...state.pin,sampleHeight,state.hour):[NaN,NaN];
  const snapshot={ready:true,view:currentView,hour:state.hour,location:state.pinName,
    speed:isFinite(u)?fmtSpeed(Math.hypot(u,v)):"—",direction:isFinite(u)?dirName(u,v):"Sous le relief",altitude:z,heightReference,
    source:window.ATOM?.pret()?"modele":"demonstration",sceneMode,terrainSource:window.ATOM?.terrainStats?.().windSource??"national"};
  window.dispatchEvent(new CustomEvent("atom:state",{detail:snapshot}));
  return snapshot;
}
if(window.ATOM) ATOM.workspace={navigate,snapshot:publishWorkspace,
  terrainWind:(lon,lat,height)=>{const [x,y]=xy(lat,lon);if(x<0||x>W||y<0||y>H||!Number.isFinite(height)||height<2||height>1000)return null;const [u,v]=wind(x,y,height,state.hour),speed=Math.hypot(u,v);if(!Number.isFinite(speed))return null;return {u,v,speed,formatted:fmtSpeed(speed),direction:dirName(u,v),color:speedColor(kmh(speed))};},
  terrainPoint:(lon,lat)=>{const [x,y]=xy(lat,lon);if(x<0||x>W||y<0||y>H)return;placePin(x,y);navigate("monvent");},
  inspect:()=>openSheet(...state.pin,currentView==="carte3d"?z3:ACTS[state.act].z)};
applyAccess();
const th0=store.get("aw-theme","light"); if(th0!=="auto") setTheme(th0);
if(window.ATOM) ATOM.rafraichir = () => { for(const k in DSTAT) delete DSTAT[k]; MAPS.forEach(m=>m.redraw()); setHour(state.hour); renderRel(); }; if(window.ATOM) ATOM.setHour = setHour;
renderAlerts(); setHour(15); renderRel();
navigate(location.hash.slice(1)||(me?ROLES[me.role].home:"monvent"),true);
let seen=false; try{ seen=localStorage.getItem("aw-tour")==="1"; }catch(e){}
if(!seen) tour(0);
}
