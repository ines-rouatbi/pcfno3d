/* ATOM Wind — onglet d'administration « Réseau ANEO » : planification nationale et installation par zone.
 *
 * Portage de la maquette atom_wind_reseau_aneo.html de Tukhnanutha (balisage, style et script d'origine ;
 * identifiants préfixés « rp- », style limité à #aneo-plan dans styles/reseau-aneo-plan.css), branché sur
 * l'application :
 *   - plan OS6 mesuré : quand le service répond, GET /api/atom-wind/reseau-aneo?detail=true remplace le plan
 *     d'exemple (capteurs, ordre, phases, statut installé) et la courbe d'exemple par la courbe mesurée sur le test ;
 *     le statut « posé » s'enregistre par POST /api/atom-wind/reseau-aneo/installes, et la commande de relance OS6
 *     et les sorties brutes sont proposées (le service ne lance jamais OS6 lui-même) ;
 *   - sans service, ou si OS6 n'a pas encore écrit ses sorties : plan, relief, erreur et courbes d'EXEMPLE, signalés
 *     comme tels dans l'en-tête (pastille « Données d'exemple ») — rien n'est présenté comme une mesure ;
 *   - la couche d'erreur, le relief et le vent animé restent ceux de la maquette (illustratifs) : le modèle ne
 *     fournit pas encore d'erreur nationale par point ;
 *   - zones ajoutées au plan : gardées dans ce navigateur (localStorage « atom-aneo-zones-v1 ») et exportées ;
 *   - vue 3D de zone : three.js de l'application (ATOM.loadThree), pas de CDN ; polices locales.
 * Point d'entrée : ATOM.reseauAneo.afficher(), appelé par atom-runtime.js à l'ouverture de la vue « reseau ».
 */
const BALISAGE = "<header class=\"top\">\n  <div class=\"path\"><span>Administration</span><span>Réseau ANEO</span><span>Planification nationale</span></div>\n  <div class=\"grow\"></div>\n  <span class=\"flag\"><i style=\"background:var(--bad)\"></i>Modèle C' · <strong>gain capteurs nul</strong></span>\n  <span class=\"flag\"><i style=\"background:var(--warn)\"></i>D2 · <strong>en préparation</strong></span>\n  <span class=\"flag\" id=\"rp-flag-data\" title=\"Relief, vent, erreurs et courbes sont des exemples, pas des mesures\">Données d'exemple</span>\n</header>\n\n<main class=\"app\">\n  <aside class=\"pane rail\" aria-label=\"Commandes du plan national\">\n    <div class=\"blk\">\n      <div class=\"h\">Plan national · placement</div>\n      <div class=\"seg\" role=\"group\" aria-label=\"Stratégie de placement\">\n        <button id=\"rp-s-opt\" aria-pressed=\"true\">Optimisé</button>\n        <button id=\"rp-s-grid\" aria-pressed=\"false\">Grille</button>\n        <button id=\"rp-s-rand\" aria-pressed=\"false\">Aléatoire</button>\n      </div>\n    </div>\n    <div class=\"blk\">\n      <label class=\"h\" for=\"rp-n\">Mâts Porteur au sol <em id=\"rp-n-val\">0</em></label>\n      <input type=\"range\" id=\"rp-n\" min=\"0\" max=\"400\" step=\"5\" value=\"40\">\n      <label class=\"h\" for=\"rp-d\" style=\"margin-top:4px\">Profils drone 25 m à 1 km <em id=\"rp-d-val\">0</em></label>\n      <input type=\"range\" id=\"rp-d\" class=\"dr\" min=\"0\" max=\"24\" step=\"1\" value=\"4\">\n      <label class=\"h\" for=\"rp-mast\" style=\"margin-top:4px\">Hauteur des mâts</label>\n      <select id=\"rp-mast\" class=\"sel\"><option value=\"15\">15 m</option><option value=\"20\">20 m</option><option value=\"25\" selected>25 m</option><option value=\"30\">30 m</option></select>\n    </div>\n    <div class=\"blk\">\n      <label class=\"h\" for=\"rp-layer\">Couche d'analyse</label>\n      <select id=\"rp-layer\" class=\"sel\">\n        <option value=\"res\" selected>Erreur résiduelle (m/s)</option>\n        <option value=\"gain\">Gain apporté par le réseau</option>\n        <option value=\"none\">Aucune</option>\n      </select>\n      <div class=\"cbar\" id=\"rp-cbar\"></div>\n      <div class=\"ticks\" id=\"rp-cticks\"></div>\n    </div>\n    <div class=\"blk\">\n      <div class=\"h\">Couches</div>\n      <label class=\"tg\">Relief ombré <input type=\"checkbox\" id=\"rp-t-relief\" checked></label>\n      <label class=\"tg\">Vent animé <input type=\"checkbox\" id=\"rp-t-wind\" checked></label>\n      <label class=\"tg\">Zones d'influence <input type=\"checkbox\" id=\"rp-t-ring\" checked></label>\n      <label class=\"tg\">Zones agricoles et éoliennes <input type=\"checkbox\" id=\"rp-t-zones\" checked></label>\n      <label class=\"tg\">Stations existantes <input type=\"checkbox\" id=\"rp-t-air\" checked></label>\n    </div>\n    <div class=\"blk\">\n      <div class=\"h\">Symboles</div>\n      <div class=\"sym\">\n        <svg width=\"18\" height=\"14\" aria-hidden=\"true\"><circle cx=\"9\" cy=\"7\" r=\"4.5\" fill=\"#5ec4cf\" stroke=\"#0a1319\" stroke-width=\"1.5\"/></svg><span>Mât ANEO du plan</span>\n        <svg width=\"18\" height=\"18\" aria-hidden=\"true\"><line x1=\"9\" y1=\"2\" x2=\"9\" y2=\"14\" stroke=\"#c4a2ff\" stroke-width=\"2\"/><path d=\"M5 14h8l-4 3z\" fill=\"#c4a2ff\"/><line x1=\"6\" y1=\"5\" x2=\"12\" y2=\"5\" stroke=\"#c4a2ff\"/><line x1=\"6\" y1=\"9\" x2=\"12\" y2=\"9\" stroke=\"#c4a2ff\"/></svg><span>Profil drone, colonne 0 à 1 km</span>\n        <svg width=\"18\" height=\"14\" aria-hidden=\"true\"><circle cx=\"9\" cy=\"7\" r=\"4.5\" fill=\"none\" stroke=\"#f2b84b\" stroke-width=\"2\"/></svg><span>ANEO potentiel dans la zone</span>\n        <svg width=\"18\" height=\"14\" aria-hidden=\"true\"><rect x=\"5\" y=\"3\" width=\"8\" height=\"8\" transform=\"rotate(45 9 7)\" fill=\"none\" stroke=\"#e4ecee\" stroke-width=\"1.4\"/></svg><span>Station METAR existante, 10 m</span>\n        <svg width=\"18\" height=\"10\" aria-hidden=\"true\"><line x1=\"1\" y1=\"5\" x2=\"17\" y2=\"5\" stroke=\"#a9d3df\" stroke-width=\"2\" stroke-linecap=\"round\"/></svg><span>Vent illustratif, à divergence nulle</span>\n      </div>\n    </div>\n    <div class=\"blk\">\n      <div class=\"h\">Ce qui conditionne ce plan</div>\n      <div class=\"state\">\n        <div><i style=\"background:var(--bad)\"></i><p class=\"sub\"><b style=\"color:var(--ink)\">C' ignore les capteurs.</b> Même erreur avec ou sans 50 capteurs : 1,893 m/s (validation).</p></div>\n        <div><i style=\"background:var(--warn)\"></i><p class=\"sub\"><b style=\"color:var(--ink)\">D2 les exploitera.</b> Canal d'innovation, mesures au sol et en altitude.</p></div>\n        <div><i style=\"background:var(--mast)\"></i><p class=\"sub\"><b style=\"color:var(--ink)\">OS4 :</b> au moins 10 % d'erreur en moins que sans réseau. Seuil à confirmer.</p></div>\n      </div>\n    </div>\n  </aside>\n\n  <section class=\"mapwrap\" aria-label=\"Carte de planification\">\n    <canvas id=\"rp-base\" tabindex=\"0\" role=\"img\" aria-label=\"Carte de la Tunisie. Cliquez une zone pour y placer des ANEO potentiels. Glisser pour déplacer, molette ou boutons pour zoomer.\"></canvas>\n    <canvas id=\"rp-flow\" aria-hidden=\"true\"></canvas>\n    <div class=\"ov kpi\" aria-live=\"polite\">\n      <div><small>Erreur attendue</small><b id=\"rp-k-err\">-</b></div>\n      <div><small>Gain vs sans réseau</small><b id=\"rp-k-gain\">-</b></div>\n      <div><small>Objectif OS4</small><b id=\"rp-k-os4\">-</b></div>\n      <div><small>Réseau</small><b id=\"rp-k-net\">-</b></div>\n    </div>\n    <div class=\"ov zoom\">\n      <button id=\"rp-z-in\" aria-label=\"Zoomer\">+</button>\n      <button id=\"rp-z-out\" aria-label=\"Dézoomer\">−</button>\n      <button id=\"rp-z-fit\" aria-label=\"Recadrer sur la Tunisie\" style=\"font-size:12px\">⤢</button>\n    </div>\n    <div class=\"ov read\"><span class=\"scale\"><i id=\"rp-sbar\" style=\"width:60px\"></i><span id=\"rp-slab\">50 km</span></span><span id=\"rp-rpos\"><span class=\"hint\">Cliquez une zone de la Tunisie pour y placer des ANEO</span></span></div>\n\n    <div id=\"rp-v3d\" hidden>\n      <div class=\"ov bar3\">\n        <span class=\"ttl\">Vue 3D de la zone<small id=\"rp-t3s\"></small></span>\n        <span class=\"grow\"></span>\n        <label>Relief ×<span id=\"rp-ex-v\">3</span><input type=\"range\" id=\"rp-ex3\" min=\"1\" max=\"6\" step=\"0.5\" value=\"3\"></label>\n        <label><input type=\"checkbox\" id=\"rp-w3\" checked> Vent à 150 m</label>\n        <label><input type=\"checkbox\" id=\"rp-e3\"> Erreur</label>\n        <label><input type=\"checkbox\" id=\"rp-i3\" checked> Influence</label>\n        <label><input type=\"checkbox\" id=\"rp-l3\" checked> Étiquettes</label>\n        <button class=\"btn\" id=\"rp-b3\">Retour à la carte</button>\n      </div>\n      <div class=\"ov compass\" aria-hidden=\"true\"><svg width=\"34\" height=\"34\" viewBox=\"-17 -17 34 34\"><g id=\"rp-cmpg\"><path d=\"M0-13 L4 0 L0-3 L-4 0Z\" fill=\"#f2b84b\"/><path d=\"M0 13 L4 0 L0 3 L-4 0Z\" fill=\"#5d727a\"/><text y=\"-6\" x=\"7\" font-size=\"8\" fill=\"#e4ecee\" font-family=\"IBM Plex Mono\">N</text></g></svg></div>\n      <div class=\"ov card3\" id=\"rp-card3\"></div>\n      <div class=\"ov legend3\">\n        <span><i style=\"background:#f2b84b\"></i>ANEO potentiel (mât)</span>\n        <span><i style=\"background:#c4a2ff\"></i>Profil drone, anneaux = vent par niveau</span>\n        <span><i style=\"background:#5ec4cf\"></i>Mât déjà au plan</span>\n        <span><i style=\"background:#e4ecee\"></i>Station METAR</span>\n        <span style=\"color:var(--faint)\">Mâts non à l'échelle. Relief d'exemple.</span>\n      </div>\n      <div id=\"rp-lbls\"></div>\n    </div>\n  </section>\n\n  <aside class=\"pane insp\" aria-label=\"Analyse\">\n    <div class=\"tabs\" role=\"tablist\">\n      <button role=\"tab\" id=\"rp-tab-zone\" aria-selected=\"true\" aria-controls=\"rp-p-zone\">Zone</button>\n      <button role=\"tab\" id=\"rp-tab-dim\" aria-selected=\"false\" aria-controls=\"rp-p-dim\">Dimensionnement</button>\n      <button role=\"tab\" id=\"rp-tab-site\" aria-selected=\"false\" aria-controls=\"rp-p-site\">Site</button>\n    </div>\n    <div id=\"rp-p-zone\" role=\"tabpanel\" aria-labelledby=\"rp-tab-zone\"></div>\n    <div id=\"rp-p-dim\" role=\"tabpanel\" aria-labelledby=\"rp-tab-dim\" hidden>\n      <div class=\"blk\">\n        <div class=\"h\">Courbe de dimensionnement <em id=\"rp-c-note\"></em></div>\n        <svg class=\"chart\" id=\"rp-curve\" viewBox=\"0 0 340 236\" role=\"img\" aria-label=\"Erreur attendue selon le nombre de mâts\"></svg>\n        <div class=\"lgd\" id=\"rp-c-lgd\"></div>\n        <p class=\"sub\" id=\"rp-c-sub\">Erreur sur la pire tranche de 0 à 1 km au-dessus du sol. Courbes d'exemple, ancrées sur la référence mesurée sans réseau.</p>\n      </div>\n      <div class=\"blk\">\n        <div class=\"h\">Phases d'installation</div>\n        <div class=\"tbx\"><table>\n          <thead><tr><th>Phase</th><th>Sites</th><th>Erreur</th><th>Gain</th><th>Posés</th></tr></thead>\n          <tbody id=\"rp-phases\"></tbody>\n        </table></div>\n        <div class=\"acts\">\n          <button class=\"btn\" id=\"rp-x-csv\">Copier le CSV</button>\n          <button class=\"btn\" id=\"rp-x-geo\">Copier le GeoJSON</button>\n          <span class=\"toast\" id=\"rp-toast\" aria-live=\"polite\"></span>\n        </div>\n        <textarea id=\"rp-x-out\" readonly hidden aria-label=\"Plan exporté\"></textarea><div id=\"rp-os6\"></div>\n      </div>\n    </div>\n    <div id=\"rp-p-site\" role=\"tabpanel\" aria-labelledby=\"rp-tab-site\" hidden>\n      <div class=\"blk\" id=\"rp-insp-head\"></div>\n      <div class=\"blk\" id=\"rp-insp-body\"></div>\n      <div class=\"blk\">\n        <div class=\"h\">Profil vertical au site <em id=\"rp-prof-note\"></em></div>\n        <svg class=\"chart\" id=\"rp-prof\" viewBox=\"0 0 300 260\" role=\"img\" aria-label=\"Profil vertical du vent au site sélectionné\"></svg>\n        <p class=\"sub\">Profil illustratif. La vraie colonne viendra de la vérité d'entraînement et des premières mesures.</p>\n      </div>\n    </div>\n  </aside>\n</main>";

export function monterPlanAneo(hote){
hote.innerHTML = '<div id="aneo-plan">' + BALISAGE + '</div>';
(function(){
"use strict";
var $=function(id){return document.getElementById(id);};
var fmt=function(x,d){return x.toLocaleString("fr-FR",{minimumFractionDigits:d,maximumFractionDigits:d});};
var RM=window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ================= geographie ================= */
var LON0=7.3,LON1=11.8,LAT0=30.0,LAT1=37.6,K=Math.cos(34*Math.PI/180),KM=111.2;
var TN=[[8.62,36.94],[9.2,37.23],[9.87,37.33],[10.25,37.18],[10.35,36.95],[10.6,36.88],[11.05,37.08],[11.12,36.85],[10.8,36.45],[10.55,36.38],[10.5,36.05],[10.64,35.83],[10.85,35.77],[11.07,35.5],[11.13,35.23],[10.95,34.95],[10.76,34.74],[10.45,34.5],[10.15,34.3],[10.05,34.05],[10.1,33.88],[10.45,33.62],[10.75,33.65],[11.1,33.5],[11.4,33.15],[11.56,33.17],[11.5,32.6],[11.2,32.3],[10.9,32.05],[10.3,31.7],[10.2,31.3],[10.1,30.9],[9.6,30.4],[9.32,30.23],[9.15,30.9],[9.05,31.5],[8.5,32.2],[8.2,32.7],[7.75,33.2],[7.52,33.85],[7.75,34.2],[8.25,34.65],[8.3,35.2],[8.35,35.7],[8.3,36.2],[8.25,36.45],[8.45,36.85]];
var RLON0=5.4,RLON1=13.8,RLAT0=29.4,RLAT1=38.3;
var SEA=TN.slice(0,26).concat([[12.08,32.93],[13.2,32.89],[13.8,32.85],[13.8,38.3],[5.4,38.3],[5.4,36.85],[6.6,37.05],[7.3,36.95],[8.1,36.9]]);
function pip(poly,lon,lat){var c=false;for(var i=0,j=poly.length-1;i<poly.length;j=i++){var a=poly[i],b=poly[j];
  if(((a[1]>lat)!==(b[1]>lat))&&(lon<(b[0]-a[0])*(lat-a[1])/(b[1]-a[1])+a[0]))c=!c;}return c;}
var AIR=[["Tabarka",8.88,36.98],["Bizerte",9.79,37.25],["Tunis-Carthage",10.23,36.85],["Enfidha",10.44,36.08],["Monastir",10.75,35.76],["Kairouan",10.03,35.67],["Sfax",10.69,34.72],["Gafsa",8.82,34.42],["Gabès",10.10,33.88],["Djerba",10.78,33.87],["Tozeur",8.11,33.94],["Remada",10.38,32.31]];
var CITIES=[["Tunis",10.18,36.80],["Bizerte",9.87,37.27],["Sousse",10.64,35.83],["Sfax",10.76,34.74],["Gabès",10.10,33.88],["Gafsa",8.78,34.42],["Kairouan",10.10,35.68],["Kasserine",8.84,35.17],["Tozeur",8.13,33.92],["Tataouine",10.45,32.93],["Jendouba",8.78,36.50],["Béja",9.18,36.73],["Le Kef",8.71,36.17],["Siliana",9.37,36.08],["Zaghouan",10.14,36.40],["Nabeul",10.73,36.45],["Mahdia",11.06,35.50],["Sidi Bouzid",9.48,35.04],["Kébili",8.97,33.70],["Médenine",10.50,33.35]];
var ZONES=[{n:"Cap Bon",t:"agri",lon:10.82,lat:36.75,rx:.3,ry:.2},{n:"Medjerda",t:"agri",lon:9.6,lat:36.72,rx:.45,ry:.18},{n:"Sahel",t:"agri",lon:10.55,lat:35.45,rx:.38,ry:.3},{n:"Metline",t:"eol",lon:9.98,lat:37.2,rx:.2,ry:.08},{n:"Golfe de Gabès",t:"eol",lon:10.25,lat:33.55,rx:.28,ry:.22}];

/* ================= relief d'exemple ================= */
function hash(i,j){var h=(Math.imul(i,374761393)+Math.imul(j,668265263))|0;h=Math.imul(h^(h>>>13),1274126177);return((h^(h>>>16))>>>0)/4294967296;}
function vn(x,y){var xi=Math.floor(x),yi=Math.floor(y),xf=x-xi,yf=y-yi,u=xf*xf*(3-2*xf),v=yf*yf*(3-2*yf);
  var a=hash(xi,yi),b=hash(xi+1,yi),c=hash(xi,yi+1),d=hash(xi+1,yi+1);return a+(b-a)*u+(c-a)*v+(a-b-c+d)*u*v;}
function fbm(x,y){return vn(x,y)*.5+vn(x*2.13,y*2.13)*.27+vn(x*4.41,y*4.41)*.15+vn(x*8.9,y*8.9)*.08;}
function g(lon,lat,a,b,s){var dx=(lon-a)*K,dy=lat-b;return Math.exp(-(dx*dx+dy*dy)/(2*s*s));}
function seg(lon,lat,x0,y0,x1,y1,w){var px=lon*K,ax=x0*K,bx=x1*K,vx=bx-ax,vy=y1-y0,t=((px-ax)*vx+(lat-y0)*vy)/(vx*vx+vy*vy);t=t<0?0:t>1?1:t;
  var dx=px-(ax+t*vx),dy=lat-(y0+t*vy);return Math.exp(-(dx*dx+dy*dy)/(2*w*w));}
function hS(lon,lat){return 110+760*seg(lon,lat,8.35,35.05,10.45,36.42,.19)+620*g(lon,lat,8.75,36.62,.2)+720*g(lon,lat,8.68,35.22,.11)
  +380*seg(lon,lat,8.4,34.42,9.35,34.62,.09)+420*seg(lon,lat,10.05,33.55,10.35,32.3,.15)+240*g(lon,lat,9.3,35.9,.33)
  -90*g(lon,lat,10.65,35.3,.45)-80*g(lon,lat,10.3,34.3,.45)+90*g(lon,lat,9.0,31.2,.9);}
function salt(lon,lat){var a=Math.pow((lon-8.45)/.62,2)+Math.pow((lat-33.72)/.19,2),b=Math.pow((lon-9.25)/.36,2)+Math.pow((lat-33.92)/.09,2),c=Math.pow((lon-8.0)/.26,2)+Math.pow((lat-34.22)/.11,2);return a<1||b<1||c<1;}
function sand(lon,lat){var s=(32.9-lat)/.8;return s<0?0:s>1?1:s;}
function height(lon,lat){if(salt(lon,lat))return 12;var b=hS(lon,lat);return Math.max(0,b+(fbm(lon*7,lat*7)-.5)*260*(.35+Math.min(1,b/550)));}
function hFine(lon,lat){if(salt(lon,lat))return 12;var b=hS(lon,lat),h=height(lon,lat);return Math.min(1544,Math.max(0,h+(fbm(lon*34+3.1,lat*34+7.7)-.5)*150*(.25+Math.min(1,b/500))+(fbm(lon*90,lat*90)-.5)*30));}

var RW=460,RH=Math.round(RW*(RLAT1-RLAT0)/((RLON1-RLON0)*K));
var HGT=new Float32Array(RW*RH),CLS=new Uint8Array(RW*RH);
(function(){for(var j=0;j<RH;j++){var lat=RLAT1-(j+.5)/RH*(RLAT1-RLAT0);for(var i=0;i<RW;i++){var lon=RLON0+(i+.5)/RW*(RLON1-RLON0),k=j*RW+i;
  if(pip(SEA,lon,lat)){CLS[k]=0;HGT[k]=0;}else{CLS[k]=pip(TN,lon,lat)?1:2;HGT[k]=height(lon,lat);}}}})();
function hexrgb(h){return[parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];}
function ramp(stops){var s=stops.map(function(x){return[x[0],hexrgb(x[1])];});return function(v){if(v<=s[0][0])return s[0][1];for(var i=1;i<s.length;i++)if(v<=s[i][0]){var t=(v-s[i-1][0])/(s[i][0]-s[i-1][0]),a=s[i-1][1],b=s[i][1];return[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];}return s[s.length-1][1];};}
var HYPSO=ramp([[0,"#24352b"],[180,"#2f3d2e"],[450,"#4a4b39"],[800,"#625a46"],[1300,"#80745d"]]);
var HYPSO3=ramp([[0,"#34503a"],[150,"#4a5f3e"],[400,"#66624a"],[750,"#776a52"],[1100,"#8a7e69"],[1500,"#a69c8a"]]);
var SANDC=hexrgb("#5d4c31"),SALTC=hexrgb("#8b8877");
function landCol(lon,lat,h,pal){var col=salt(lon,lat)?(pal===HYPSO3?[196,190,170]:SALTC.slice()):pal(h).slice(),sd=sand(lon,lat)*(1-Math.min(1,h/500));
  if(sd>0&&!salt(lon,lat)){var sc=pal===HYPSO3?[176,144,98]:SANDC;col=[col[0]+(sc[0]-col[0])*sd,col[1]+(sc[1]-col[1])*sd,col[2]+(sc[2]-col[2])*sd];}return col;}
function makeRaster(shaded){var c=document.createElement("canvas");c.width=RW;c.height=RH;var x=c.getContext("2d"),im=x.createImageData(RW,RH),d=im.data;
  var cell=(RLON1-RLON0)*K*KM*1000/RW,ex=7;
  for(var j=0;j<RH;j++){var lat=RLAT1-(j+.5)/RH*(RLAT1-RLAT0);for(var i=0;i<RW;i++){var k=j*RW+i,o=k*4,lon=RLON0+(i+.5)/RW*(RLON1-RLON0),col;
    if(CLS[k]===0){var sh=.25*g(lon,lat,10.8,34.2,.55)+.15*g(lon,lat,10.9,36.9,.25),nz=(fbm(lon*3,lat*3)-.5)*.06;col=[11+18*sh,24+30*sh+nz*40,34+34*sh+nz*40];}
    else{var h=HGT[k];col=landCol(lon,lat,h,HYPSO);
      var f=1;if(shaded){var il=k-(i>0?1:0),ir=k+(i<RW-1?1:0),iu=k-(j>0?RW:0),idn=k+(j<RH-1?RW:0);
        var hl=CLS[il]===0?h:HGT[il],hr=CLS[ir]===0?h:HGT[ir],hu=CLS[iu]===0?h:HGT[iu],hd=CLS[idn]===0?h:HGT[idn];
        var dzx=(hr-hl)/(2*cell)*ex,dzy=(hu-hd)/(2*cell)*ex,nl=Math.sqrt(dzx*dzx+dzy*dzy+1),dot=(dzx*.62-dzy*.62+.48)/nl;
        f=.5+.85*Math.max(0,dot)/.78;}
      col=[col[0]*f,col[1]*f,col[2]*f];
      if(CLS[k]===2){var m=(col[0]+col[1]+col[2])/3;col=[(col[0]*.4+m*.6)*.55,(col[1]*.4+m*.6)*.55,(col[2]*.4+m*.6)*.6];}}
    d[o]=col[0];d[o+1]=col[1];d[o+2]=col[2];d[o+3]=255;}}
  x.putImageData(im,0,0);return c;}
var RELIEF=makeRaster(true),FLAT=makeRaster(false);
function hAt(lon,lat){var i=Math.floor((lon-RLON0)/(RLON1-RLON0)*RW),j=Math.floor((RLAT1-lat)/(RLAT1-RLAT0)*RH);if(i<0||j<0||i>=RW||j>=RH)return null;var k=j*RW+i;return{h:HGT[k],c:CLS[k]};}

/* ================= grille d'erreur et reseau ================= */
var R0=1.893,TARGET=R0*.9;
var DLAT=.04,DLON=.04/K,NX=Math.ceil((LON1-LON0)/DLON),NY=Math.ceil((LAT1-LAT0)/DLAT);
var GI=new Int32Array(NX*NY).fill(-1),CLON=[],CLAT=[],CIX=[],CIY=[],E0=[];
function err0(lon,lat){return .42+.5*g(lon,lat,9.0,35.8,.42)+.45*g(lon,lat,8.8,36.6,.28)+.33*g(lon,lat,10.75,36.65,.28)+.3*g(lon,lat,10.2,33.7,.38)+.24*g(lon,lat,9.4,32.2,.75)+.2*g(lon,lat,10.4,35.2,.45)+.16*(fbm(lon*2.6,lat*2.6)-.5);}
for(var iy=0;iy<NY;iy++)for(var ix=0;ix<NX;ix++){var lo=LON0+(ix+.5)*DLON,la=LAT1-(iy+.5)*DLAT;if(pip(TN,lo,la)){GI[iy*NX+ix]=CLON.length;CLON.push(lo);CLAT.push(la);CIX.push(ix);CIY.push(iy);E0.push(err0(lo,la));}}
var NC=CLON.length,SUM0=E0.reduce(function(a,b){return a+b;},0);
var MAST={L:9,A:.3},DRONE={L:32,A:.13};
function apply(res,lon,lat,P){var rx=Math.ceil(3*P.L/(DLON*K*KM)),ry=Math.ceil(3*P.L/(DLAT*KM)),cx=Math.floor((lon-LON0)/DLON),cy=Math.floor((LAT1-lat)/DLAT),dl=0,s2=2*P.L*P.L;
  for(var y=Math.max(0,cy-ry);y<=Math.min(NY-1,cy+ry);y++)for(var x=Math.max(0,cx-rx);x<=Math.min(NX-1,cx+rx);x++){var k=GI[y*NX+x];if(k<0)continue;
    var dx=(CLON[k]-lon)*K*KM,dy=(CLAT[k]-lat)*KM,f=P.A*Math.exp(-(dx*dx+dy*dy)/s2),v=res[k]*f;res[k]-=v;dl+=v;}return dl;}
function cellOf(lon,lat){var x=Math.floor((lon-LON0)/DLON),y=Math.floor((LAT1-lat)/DLAT);if(x<0||y<0||x>=NX||y>=NY)return -1;return GI[y*NX+x];}
function rng(seed){var s=seed>>>0;return function(){s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296;};}
function km2(a,b){var dx=(a.lon-b.lon)*K*KM,dy=(a.lat-b.lat)*KM;return dx*dx+dy*dy;}
function farFrom(list,lon,lat,km){var p={lon:lon,lat:lat};for(var i=0;i<list.length;i++)if(km2(list[i],p)<km*km)return false;return true;}
function farAir(lon,lat){for(var i=0;i<AIR.length;i++){var dx=(AIR[i][1]-lon)*K*KM,dy=(AIR[i][2]-lat)*KM;if(dx*dx+dy*dy<64)return false;}return true;}
function orderOpt(){var res=E0.slice(),used=new Uint8Array(NC),out=[];
  while(out.length<400){var b=-1,bv=-1;for(var k=0;k<NC;k++)if(!used[k]&&res[k]>bv&&farAir(CLON[k],CLAT[k])){bv=res[k];b=k;}if(b<0)break;used[b]=1;out.push({lon:CLON[b],lat:CLAT[b]});apply(res,CLON[b],CLAT[b],MAST);}return out;}
function orderGrid(){var pts=[];for(var k=0;k<NC;k++)if(CIX[k]%5===2&&CIY[k]%5===2&&farAir(CLON[k],CLAT[k]))pts.push({lon:CLON[k],lat:CLAT[k]});
  var ord=[],u={};[16,8,4,2,1].forEach(function(s){for(var i=0;i<pts.length;i+=s)if(!u[i]){u[i]=1;ord.push(pts[i]);}});return ord.slice(0,400);}
function orderRand(){var r=rng(42),idx=[],out=[];for(var k=0;k<NC;k++)idx.push(k);for(var i=idx.length-1;i>0;i--){var j=Math.floor(r()*(i+1)),t=idx[i];idx[i]=idx[j];idx[j]=t;}
  for(var q=0;q<idx.length&&out.length<400;q++){var k2=idx[q];if(farAir(CLON[k2],CLAT[k2])&&farFrom(out,CLON[k2],CLAT[k2],9))out.push({lon:CLON[k2],lat:CLAT[k2]});}return out;}
function orderDrone(){var idx=[];for(var k=0;k<NC;k++)idx.push(k);idx.sort(function(a,b){return E0[b]-E0[a];});var out=[];
  for(var q=0;q<idx.length&&out.length<24;q++){var k2=idx[q];if(farFrom(out,CLON[k2],CLAT[k2],55))out.push({lon:CLON[k2],lat:CLAT[k2]});}return out;}
var ORD={opt:orderOpt(),grid:orderGrid(),rand:orderRand()},DRN=orderDrone();
var PH=[{n:1,a:1,b:40,c:"#8fe3ec"},{n:2,a:41,b:120,c:"#5ec4cf"},{n:3,a:121,b:250,c:"#3c8f9c"},{n:4,a:251,b:400,c:"#2b6570"}];
function phaseOf(r){for(var i=0;i<PH.length;i++)if(r>=PH[i].a&&r<=PH[i].b)return PH[i];return PH[3];}
ORD.opt.forEach(function(o,i){o.id="ANEO-"+String(i+1).padStart(3,"0");o.rank=i+1;o.kind="mast";o.status="prévu";});
["grid","rand"].forEach(function(s){ORD[s].forEach(function(o,i){o.id=(s==="grid"?"GRL-":"ALT-")+String(i+1).padStart(3,"0");o.rank=i+1;o.kind="mast";o.status="référence";});});
DRN.forEach(function(o,i){o.id="ANEO-V"+String(i+1).padStart(2,"0");o.rank=i+1;o.kind="drone";o.status="prévu";});
var CUSTOM=[];try{var cz=JSON.parse(localStorage.getItem("atom-aneo-zones-v1")||"[]");if(Array.isArray(cz))CUSTOM=cz.filter(function(o){return o&&isFinite(o.lon)&&isFinite(o.lat);});}catch(e){}
function saveCustom(){try{localStorage.setItem("atom-aneo-zones-v1",JSON.stringify(CUSTOM));}catch(e){}}
function curveOf(strat,d){var res=E0.slice(),s=SUM0,out=[];for(var i=0;i<d;i++)s-=apply(res,DRN[i].lon,DRN[i].lat,DRONE);out.push(R0*s/SUM0);
  var o=ORD[strat];for(var n=1;n<=400;n++){if(o[n-1])s-=apply(res,o[n-1].lon,o[n-1].lat,MAST);if(n%5===0)out.push(R0*s/SUM0);}return out;}
function fieldOf(strat,n,d,extra){var res=E0.slice();for(var i=0;i<d;i++)apply(res,DRN[i].lon,DRN[i].lat,DRONE);var o=ORD[strat];for(var q=0;q<n&&q<o.length;q++)apply(res,o[q].lon,o[q].lat,MAST);
  CUSTOM.forEach(function(c){apply(res,c.lon,c.lat,c.kind==="drone"?DRONE:MAST);});if(extra)extra.forEach(function(c){apply(res,c.lon,c.lat,c.kind==="drone"?DRONE:MAST);});return res;}
function resAt(arr,lon,lat){var fx=(lon-LON0)/DLON-.5,fy=(LAT1-lat)/DLAT-.5,x0=Math.floor(fx),y0=Math.floor(fy),tx=fx-x0,ty=fy-y0,s=0,w=0;
  [[0,0,(1-tx)*(1-ty)],[1,0,tx*(1-ty)],[0,1,(1-tx)*ty],[1,1,tx*ty]].forEach(function(q){var x=x0+q[0],y=y0+q[1];if(x<0||y<0||x>=NX||y>=NY)return;var k=GI[y*NX+x];if(k<0)return;s+=arr[k]*q[2];w+=q[2];});
  return w>0?s/w:null;}

/* ================= vent d'exemple a divergence nulle ================= */
var WU=5.2,WV=-4.6,WA=95;
function psi(lon,lat){var x=(lon-9.5)*K*KM,y=(lat-34)*KM;return WU*y-WV*x+WA*hS(lon,lat)/1000*(pip(SEA,lon,lat)?.25:1);}
function wind(lon,lat){var e=.02,dy=(psi(lon,lat+e)-psi(lon,lat-e))/(2*e*KM),dx=(psi(lon+e,lat)-psi(lon-e,lat))/(2*e*K*KM);return[dy,-dx];}
var WIND=ramp([[2,"#3d6f8c"],[6,"#7fb6c9"],[9,"#cfe7ee"],[12,"#ffe2a1"],[15,"#ffb46b"]]);

/* ================= etat ================= */
var st={strat:"opt",n:40,d:4,mast:25,layer:"res",relief:true,wind:!RM,ring:true,zones:true,air:true,sel:null,tab:"zone"};
try{var sv=JSON.parse(localStorage.getItem("atom-aneo-plan-v3")||"null");if(sv){["strat","n","d","layer"].forEach(function(k){if(sv[k]!=null)st[k]=sv[k];});}}catch(e){}
function save(){try{localStorage.setItem("atom-aneo-plan-v3",JSON.stringify({strat:st.strat,n:st.n,d:st.d,layer:st.layer}));}catch(e){}}
var RES=null,CURVES={},CUR_D=-1;
function recompute(){RES=fieldOf(st.strat,st.n,st.d);if(CUR_D!==st.d){CURVES={opt:curveOf("opt",0),grid:curveOf("grid",0),rand:curveOf("rand",0),optd:curveOf("opt",st.d)};CUR_D=st.d;}buildErr();}
function metric(arr){arr=arr||RES;var s=0;for(var k=0;k<NC;k++)s+=arr[k];return R0*s/SUM0;}
function visible(){return{m:ORD[st.strat].slice(0,st.n),d:DRN.slice(0,st.d)};}

/* ================= zone : placement local ================= */
var ZONE=null,ZL={mast:{L:4.5,A:.38},drone:{L:11,A:.24}};
function nearestCity(lon,lat){var b=null,bd=1e9;CITIES.forEach(function(c){var d=km2({lon:c[1],lat:c[2]},{lon:lon,lat:lat});if(d<bd){bd=d;b=c[0];}});return{name:b,km:Math.sqrt(bd)};}
function slopeDeg(lon,lat){var e=.006,hx=(hFine(lon+e/K,lat)-hFine(lon-e/K,lat))/(2*e*KM*1000),hy=(hFine(lon,lat+e)-hFine(lon,lat-e))/(2*e*KM*1000);return Math.atan(Math.sqrt(hx*hx+hy*hy))*180/Math.PI;}
function rough(lon,lat){return Math.min(1,slopeDeg(lon,lat)/12);}
function terrainOf(lon,lat){if(pip(SEA,lon,lat))return"mer";if(salt(lon,lat))return"chott";var h=hFine(lon,lat);if(h>550)return"relief";if(sand(lon,lat)>.5)return"désert";
  for(var a=0;a<8;a++){var an=a*Math.PI/4;if(pip(SEA,lon+.15*Math.cos(an)/K,lat+.15*Math.sin(an)))return"côte";}return h>300?"collines":"plaine";}
function planSites(){var v=visible();return v.m.concat(v.d).concat(CUSTOM);}
function setZone(lon,lat){var c=nearestCity(lon,lat);ZONE={lon:lon,lat:lat,R:ZONE?ZONE.R:25,nm:ZONE?ZONE.nm:10,nd:ZONE?ZONE.nd:1,name:c.km<35?c.name:"Secteur "+fmt(lat,2)+"° N",pots:[],sel:null};computeZone();ZONE.sel=ZONE.pots[0]||null;}
function computeZone(){var Z=ZONE;if(!Z)return;var R=Z.R,step=Math.max(1,R/25),cands=[],plan=planSites();
  for(var y=-R;y<=R;y+=step)for(var x=-R;x<=R;x+=step){if(x*x+y*y>R*R)continue;var lon=Z.lon+x/(K*KM),lat=Z.lat+y/KM;if(!pip(TN,lon,lat)||salt(lon,lat))continue;
    var b=resAt(RES,lon,lat);if(b==null)continue;cands.push({lon:lon,lat:lat,h:hFine(lon,lat),e:b*(.78+.55*rough(lon,lat))});}
  Z.cells=cands.map(function(c){return{lon:c.lon,lat:c.lat,e:c.e,r:c.e};});
  function okc(c,limH){return c.h<=limH&&slopeDeg(c.lon,c.lat)<=20&&farAir(c.lon,c.lat)&&farFrom(plan,c.lon,c.lat,2);}
  var lim=650;if(!cands.some(function(c){return okc(c,lim);}))lim=1100;
  var pool=[];cands.forEach(function(c,i){if(okc(c,lim)){c.ci=i;pool.push(c);}});var pots=[];
  function red(P,pl){var s2=2*P.L*P.L,tot=0;Z.cells.forEach(function(c){var f=P.A*Math.exp(-km2(c,pl)/s2),v=c.r*f;c.r-=v;tot+=v;});return tot;}
  for(var q=0;q<Z.nd;q++){var b1=null,bv=-1;pool.forEach(function(c){if(!farFrom(pots,c.lon,c.lat,Math.max(10,R*.5)))return;var s=0;Z.cells.forEach(function(e){if(km2(e,c)<64)s+=e.r;});if(s>bv){bv=s;b1=c;}});
    if(!b1)break;var p={lon:b1.lon,lat:b1.lat,h:b1.h,kind:"drone"};p.gain=red(ZL.drone,p);pots.push(p);}
  for(var m=0;m<Z.nm;m++){var b2=null,bv2=-1;pool.forEach(function(c){if(!farFrom(pots,c.lon,c.lat,2.5))return;var r=Z.cells[c.ci].r;if(r>bv2){bv2=r;b2=c;}});
    if(!b2)break;var p2={lon:b2.lon,lat:b2.lat,h:b2.h,kind:"mast"};p2.gain=red(ZL.mast,p2);pots.push(p2);}
  var nmI=0,ndI=0;pots.forEach(function(p){if(p.kind==="drone"){ndI++;p.id="Z-V"+ndI;}else{nmI++;p.id="Z-"+String(nmI).padStart(2,"0");}p.terrain=terrainOf(p.lon,p.lat);p.slope=slopeDeg(p.lon,p.lat);p.zone=true;});
  var e0=0,e1=0,hmin=1e9,hmax=-1;Z.cells.forEach(function(c){e0+=c.e;e1+=c.r;});cands.forEach(function(c){if(c.h<hmin)hmin=c.h;if(c.h>hmax)hmax=c.h;});
  var tg=Z.cells.reduce(function(a,c){return a+(c.e-c.r);},0)||1;pots.forEach(function(p){p.share=p.gain/tg;});
  Z.pots=pots;Z.limH=lim;Z.stats={area:cands.length*step*step,e0:e0/Math.max(1,cands.length),e1:e1/Math.max(1,cands.length),hmin:hmin<1e9?hmin:0,hmax:hmax>0?hmax:0,n:cands.length};
  if(Z.sel&&pots.indexOf(Z.sel)<0)Z.sel=pots[0]||null;}
function acceptZone(){if(!ZONE||!ZONE.pots.length)return;var base=CUSTOM.length,nadd=ZONE.pots.length;
  ZONE.pots.forEach(function(p,i){CUSTOM.push({lon:p.lon,lat:p.lat,kind:p.kind,id:"ANEO-Z"+String(base+i+1).padStart(3,"0"),rank:base+i+1,status:"prévu",custom:true});});
  saveCustom();zMsg=nadd+" sites ajoutés au plan national (gardés dans ce navigateur). La zone propose maintenant les compléments suivants.";renderAll();}
var zMsg="";

/* ================= calque d'erreur ================= */
var ERRC=document.createElement("canvas");ERRC.width=NX;ERRC.height=NY;
var CRES=ramp([[.25,"#16405a"],[.45,"#2a7b8a"],[.65,"#d9b04a"],[.9,"#e2693d"],[1.2,"#b3293a"]]);
var CGAIN=ramp([[0,"#0e2a33"],[.08,"#1f5f6b"],[.2,"#3fb0bf"],[.35,"#a8f0e0"],[.5,"#ffffff"]]);
function buildErr(){var x=ERRC.getContext("2d"),im=x.createImageData(NX,NY),d=im.data;
  for(var k=0;k<NC;k++){var o=(CIY[k]*NX+CIX[k])*4,gn=(E0[k]-RES[k])/E0[k],c=st.layer==="gain"?CGAIN(gn):CRES(RES[k]);d[o]=c[0];d[o+1]=c[1];d[o+2]=c[2];d[o+3]=st.layer==="gain"?Math.min(255,60+900*gn):215;}
  x.putImageData(im,0,0);}
function legendBar(){var cb=$("rp-cbar"),tk=$("rp-cticks");
  if(st.layer==="none"){cb.style.background="var(--line)";tk.innerHTML="<span>Aucune couche</span>";return;}
  var f=st.layer==="gain"?CGAIN:CRES,a=st.layer==="gain"?0:.25,b=st.layer==="gain"?.5:1.2,stops=[];
  for(var i=0;i<=10;i++){var c=f(a+(b-a)*i/10);stops.push("rgb("+c.map(Math.round).join(",")+") "+(i*10)+"%");}
  cb.style.background="linear-gradient(90deg,"+stops.join(",")+")";
  tk.innerHTML=st.layer==="gain"?"<span>0 %</span><span>25 %</span><span>50 %</span>":"<span>0,25</span><span>0,70</span><span>1,20 m/s</span>";}

/* ================= carte 2D ================= */
var base=$("rp-base"),flow=$("rp-flow"),bx=base.getContext("2d"),fx=flow.getContext("2d"),wrap=base.parentNode;
var VW=0,VH=0,DPR=1,view={cx:9.55,cy:33.8,z:80},FITZ=80;
function w2s(lon,lat){return[VW/2+(lon-view.cx)*K*view.z,VH/2-(lat-view.cy)*view.z];}
function s2w(x,y){return[view.cx+(x-VW/2)/(K*view.z),view.cy-(y-VH/2)/view.z];}
function fitZ(){var w=(11.75-7.45)*K,h=37.5-30.15;return Math.max(5,Math.min((VW-24)/w,(VH-100)/h));}
function resize(){var r=wrap.getBoundingClientRect();DPR=Math.min(2,window.devicePixelRatio||1);VW=Math.max(10,r.width);VH=Math.max(10,r.height);
  [base,flow].forEach(function(c){c.width=Math.round(VW*DPR);c.height=Math.round(VH*DPR);});FITZ=fitZ();
  if(!resize.done){view={cx:9.6,cy:33.82,z:FITZ};resize.done=true;}clamp();draw();resetFlow();if(T3.on)T3.resize();}
function tnPath(ctx){ctx.beginPath();TN.forEach(function(p,i){var s=w2s(p[0],p[1]);if(i)ctx.lineTo(s[0],s[1]);else ctx.moveTo(s[0],s[1]);});ctx.closePath();}
function drawMast(c,s,r,col){c.beginPath();c.arc(s[0],s[1],r,0,Math.PI*2);c.fillStyle=col;c.fill();c.lineWidth=1.3;c.strokeStyle="#0a1319";c.stroke();}
function drawDrone(c,s,col){c.strokeStyle="#0a1319";c.lineWidth=4;c.beginPath();c.moveTo(s[0],s[1]);c.lineTo(s[0],s[1]-18);c.stroke();
  c.strokeStyle=col;c.lineWidth=2;c.beginPath();c.moveTo(s[0],s[1]);c.lineTo(s[0],s[1]-18);c.stroke();c.lineWidth=1.2;[5,10,15].forEach(function(t){c.beginPath();c.moveTo(s[0]-3,s[1]-t);c.lineTo(s[0]+3,s[1]-t);c.stroke();});
  c.fillStyle=col;c.beginPath();c.moveTo(s[0]-4.5,s[1]);c.lineTo(s[0]+4.5,s[1]);c.lineTo(s[0],s[1]+4.5);c.closePath();c.fill();}
function draw(){var c=bx;c.setTransform(DPR,0,0,DPR,0,0);c.fillStyle="#0b1822";c.fillRect(0,0,VW,VH);
  var a=w2s(RLON0,RLAT1),b=w2s(RLON1,RLAT0);c.imageSmoothingEnabled=true;c.drawImage(st.relief?RELIEF:FLAT,a[0],a[1],b[0]-a[0],b[1]-a[1]);
  if(st.layer!=="none"){var e0=w2s(LON0,LAT1),e1=w2s(LON0+NX*DLON,LAT1-NY*DLAT);c.save();tnPath(c);c.clip();c.globalAlpha=st.layer==="gain"?.85:.62;c.drawImage(ERRC,e0[0],e0[1],e1[0]-e0[0],e1[1]-e0[1]);c.restore();}
  c.strokeStyle="rgba(228,236,238,.08)";c.lineWidth=1;c.font="10px 'IBM Plex Mono', monospace";c.fillStyle="rgba(228,236,238,.42)";
  for(var la=30;la<=38;la++){var p=w2s(RLON0,la),q=w2s(RLON1,la);c.beginPath();c.moveTo(p[0],p[1]);c.lineTo(q[0],q[1]);c.stroke();if(p[1]>60&&p[1]<VH-40)c.fillText(la+"°N",6,p[1]-3);}
  for(var lo=6;lo<=13;lo++){var p2=w2s(lo,RLAT0),q2=w2s(lo,RLAT1);c.beginPath();c.moveTo(p2[0],p2[1]);c.lineTo(q2[0],q2[1]);c.stroke();if(p2[0]>30&&p2[0]<VW-40)c.fillText(lo+"°E",p2[0]+3,VH-44);}
  tnPath(c);c.strokeStyle="rgba(214,226,224,.7)";c.lineWidth=1.3;c.stroke();
  if(st.zones){c.setLineDash([4,4]);c.lineWidth=1.2;ZONES.forEach(function(z){var s=w2s(z.lon,z.lat);c.strokeStyle=z.t==="agri"?"rgba(150,214,120,.8)":"rgba(140,180,255,.8)";
    c.beginPath();c.ellipse(s[0],s[1],z.rx*K*view.z,z.ry*view.z,0,0,Math.PI*2);c.stroke();});c.setLineDash([]);
    if(view.z>FITZ*1.35){c.font="600 10.5px 'IBM Plex Sans Condensed', sans-serif";ZONES.forEach(function(z){var s=w2s(z.lon,z.lat);c.fillStyle=z.t==="agri"?"rgba(170,224,140,.95)":"rgba(165,195,255,.95)";c.fillText(z.n.toUpperCase(),s[0]+z.rx*K*view.z*.72,s[1]-z.ry*view.z*.75);});}}
  var v=visible(),pxkm=view.z/KM,cm=CUSTOM.filter(function(o){return o.kind==="mast";}),cd=CUSTOM.filter(function(o){return o.kind==="drone";});
  if(st.ring){c.lineWidth=1;v.d.concat(cd).forEach(function(o){var s=w2s(o.lon,o.lat);c.strokeStyle="rgba(196,162,255,.45)";c.fillStyle="rgba(196,162,255,.06)";c.setLineDash([3,3]);c.beginPath();c.arc(s[0],s[1],DRONE.L*pxkm,0,Math.PI*2);c.fill();c.stroke();});c.setLineDash([]);
    if(view.z>FITZ*1.6)v.m.concat(cm).forEach(function(o){var s=w2s(o.lon,o.lat);c.strokeStyle="rgba(94,196,207,.35)";c.beginPath();c.arc(s[0],s[1],MAST.L*pxkm,0,Math.PI*2);c.stroke();});}
  c.font=(view.z>FITZ*2?12:11)+"px 'IBM Plex Sans Condensed', sans-serif";
  CITIES.forEach(function(t,i){if(i>10&&view.z<FITZ*1.8)return;var s=w2s(t[1],t[2]);c.fillStyle="rgba(228,236,238,.85)";c.beginPath();c.arc(s[0],s[1],1.8,0,Math.PI*2);c.fill();c.fillStyle="rgba(228,236,238,.62)";c.fillText(t[0],s[0]+5,s[1]+3.5);});
  if(st.air){c.lineWidth=1.4;AIR.forEach(function(t){var s=w2s(t[1],t[2]);c.save();c.translate(s[0],s[1]);c.rotate(Math.PI/4);c.fillStyle="#0a1319";c.strokeStyle="#e4ecee";c.fillRect(-4,-4,8,8);c.strokeRect(-4,-4,8,8);c.restore();});}
  var r=view.z>FITZ*2?4.6:3.6;
  v.m.forEach(function(o){drawMast(c,w2s(o.lon,o.lat),r,o.status==="installé"?"#6fd39a":st.strat==="opt"?phaseOf(o.rank).c:st.strat==="grid"?"#d9b04a":"#9aa9b0");});
  cm.forEach(function(o){drawMast(c,w2s(o.lon,o.lat),r,o.status==="installé"?"#6fd39a":"#5ec4cf");});
  v.d.concat(cd).forEach(function(o){drawDrone(c,w2s(o.lon,o.lat),o.status==="installé"?"#6fd39a":"#c4a2ff");});
  if(ZONE){var zc=w2s(ZONE.lon,ZONE.lat),zr=ZONE.R*pxkm;c.beginPath();c.arc(zc[0],zc[1],zr,0,Math.PI*2);c.fillStyle="rgba(242,184,75,.07)";c.fill();
    c.setLineDash([6,4]);c.strokeStyle="#f2b84b";c.lineWidth=1.6;c.stroke();c.setLineDash([]);
    c.strokeStyle="rgba(242,184,75,.75)";c.lineWidth=1;[[0,-1],[0,1],[-1,0],[1,0]].forEach(function(dv){c.beginPath();c.moveTo(zc[0]+dv[0]*zr*.92,zc[1]+dv[1]*zr*.92);c.lineTo(zc[0]+dv[0]*(zr+8),zc[1]+dv[1]*(zr+8));c.stroke();});
    c.font="600 11px 'IBM Plex Sans Condensed', sans-serif";c.fillStyle="#f2b84b";c.fillText("ZONE · "+ZONE.name.toUpperCase()+" · "+ZONE.R+" KM",zc[0]-zr,zc[1]-zr-10);
    var showN=zr>70;ZONE.pots.forEach(function(p){var s=w2s(p.lon,p.lat),on=ZONE.sel===p;
      if(p.kind==="drone"){drawDrone(c,s,"#f2b84b");if(on){c.strokeStyle="#fff";c.lineWidth=1.5;c.beginPath();c.arc(s[0],s[1]-8,13,0,Math.PI*2);c.stroke();}}
      else{c.beginPath();c.arc(s[0],s[1],on?7:5,0,Math.PI*2);c.fillStyle="rgba(10,19,25,.75)";c.fill();c.lineWidth=2;c.strokeStyle=on?"#ffffff":"#f2b84b";c.stroke();c.beginPath();c.arc(s[0],s[1],1.8,0,Math.PI*2);c.fillStyle="#f2b84b";c.fill();}
      if(showN){c.font="10px 'IBM Plex Mono', monospace";c.fillStyle="#f2b84b";c.fillText(p.id.replace("Z-",""),s[0]+7,s[1]-6);}});}
  if(st.sel){var s=w2s(st.sel.lon,st.sel.lat),yy=st.sel.kind==="drone"?s[1]-8:s[1];c.strokeStyle="#ffffff";c.lineWidth=2;c.beginPath();c.arc(s[0],yy,st.sel.kind==="drone"?14:9,0,Math.PI*2);c.stroke();}
  scaleBar();}
function scaleBar(){var pxkm=view.z/KM,opts=[2,5,10,20,25,50,100,200],L=opts[0];for(var i=0;i<opts.length;i++)if(opts[i]*pxkm<=120)L=opts[i];$("rp-sbar").style.width=Math.round(L*pxkm)+"px";$("rp-slab").textContent=L+" km";}

/* particules 2D */
var PARTS=[],NP=1300;
function seedP(p){var w0=s2w(0,0),w1=s2w(VW||800,VH||800),lo0=Math.max(RLON0,w0[0]),lo1=Math.min(RLON1,w1[0]),la0=Math.max(RLAT0,w1[1]),la1=Math.min(RLAT1,w0[1]);
  p.lon=lo0+Math.random()*(lo1-lo0);p.lat=la0+Math.random()*(la1-la0);p.age=Math.floor(Math.random()*90);p.max=60+Math.random()*60;}
for(var i=0;i<NP;i++){var pp={};PARTS.push(pp);}
function resetFlow(){fx.setTransform(1,0,0,1,0,0);fx.clearRect(0,0,flow.width,flow.height);PARTS.forEach(seedP);if(!st.wind||T3.on)return;if(RM)staticFlow();}
function staticFlow(){fx.setTransform(DPR,0,0,DPR,0,0);fx.lineWidth=1.1;for(var y=10;y<VH;y+=22)for(var x=10;x<VW;x+=22){var w=s2w(x,y),u=wind(w[0],w[1]),sp=Math.hypot(u[0],u[1]),cc=WIND(sp),l=6+sp*.6;
  fx.strokeStyle="rgba("+cc.map(Math.round).join(",")+",.5)";fx.beginPath();fx.moveTo(x,y);fx.lineTo(x+u[0]/sp*l,y-u[1]/sp*l);fx.stroke();}}
function step2D(){if(!st.wind||RM||T3.on||!wrap.offsetParent)return;fx.setTransform(1,0,0,1,0,0);fx.globalCompositeOperation="destination-in";fx.fillStyle="rgba(0,0,0,.88)";fx.fillRect(0,0,flow.width,flow.height);
  fx.globalCompositeOperation="source-over";fx.setTransform(DPR,0,0,DPR,0,0);fx.lineWidth=1;var kk=.0065*Math.min(1,FITZ*1.5/view.z);
  for(var i=0;i<PARTS.length;i++){var p=PARTS[i],u=wind(p.lon,p.lat),a=w2s(p.lon,p.lat);p.lon+=u[0]*kk/K;p.lat+=u[1]*kk;p.age++;
    var b=w2s(p.lon,p.lat);if(p.age>p.max||b[0]<-5||b[0]>VW+5||b[1]<-5||b[1]>VH+5){seedP(p);p.age=0;continue;}var cc=WIND(Math.hypot(u[0],u[1]));
    fx.strokeStyle="rgba("+Math.round(cc[0])+","+Math.round(cc[1])+","+Math.round(cc[2])+",.5)";fx.beginPath();fx.moveTo(a[0],a[1]);fx.lineTo(b[0],b[1]);fx.stroke();}}

/* interactions carte */
var drag=null,anim=null;
base.addEventListener("pointerdown",function(e){base.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,cx:view.cx,cy:view.cy,moved:false};});
base.addEventListener("pointermove",function(e){var r=base.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
  if(drag){var dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>4){drag.moved=true;base.classList.add("drag");}
    if(drag.moved){view.cx=drag.cx-dx/(K*view.z);view.cy=drag.cy+dy/view.z;clamp();draw();resetFlow();}}
  readout(x,y);});
base.addEventListener("pointerup",function(e){var r=base.getBoundingClientRect();if(drag&&!drag.moved)pick(e.clientX-r.left,e.clientY-r.top);drag=null;base.classList.remove("drag");});
base.addEventListener("pointerleave",function(){$("rp-rpos").innerHTML='<span class="hint">Cliquez une zone de la Tunisie pour y placer des ANEO</span>';});
base.addEventListener("wheel",function(e){e.preventDefault();var r=base.getBoundingClientRect();zoomAt(e.clientX-r.left,e.clientY-r.top,Math.pow(1.0018,-e.deltaY));},{passive:false});
function clamp(){view.z=Math.max(FITZ*.8,Math.min(FITZ*14,view.z));view.cx=Math.max(RLON0+1,Math.min(RLON1-1,view.cx));view.cy=Math.max(RLAT0+1,Math.min(RLAT1-1,view.cy));}
function zoomAt(x,y,f){var w=s2w(x,y);view.z*=f;clamp();var w2=s2w(x,y);view.cx+=w[0]-w2[0];view.cy+=w[1]-w2[1];clamp();draw();resetFlow();}
$("rp-z-in").addEventListener("click",function(){zoomAt(VW/2,VH/2,1.5);});
$("rp-z-out").addEventListener("click",function(){zoomAt(VW/2,VH/2,1/1.5);});
$("rp-z-fit").addEventListener("click",function(){fly({cx:9.6,cy:33.82,z:FITZ});});
base.addEventListener("keydown",function(e){var m={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,1],ArrowDown:[0,-1]}[e.key];
  if(m){e.preventDefault();view.cx+=m[0]*60/(K*view.z);view.cy+=m[1]*60/view.z;clamp();draw();resetFlow();}
  else if(e.key==="+"||e.key==="="){zoomAt(VW/2,VH/2,1.4);}else if(e.key==="-"){zoomAt(VW/2,VH/2,1/1.4);}
  else if(e.key==="Enter"&&pip(TN,view.cx,view.cy)){setZone(view.cx,view.cy);afterZone();}});
function fly(to,instant){if(anim)cancelAnimationFrame(anim);var fr={cx:view.cx,cy:view.cy,z:view.z},t0=performance.now(),D=RM||instant?0:600;
  function f(t){var k=D?Math.min(1,(t-t0)/D):1,e=k<.5?2*k*k:1-Math.pow(-2*k+2,2)/2;view.cx=fr.cx+(to.cx-fr.cx)*e;view.cy=fr.cy+(to.cy-fr.cy)*e;view.z=Math.exp(Math.log(fr.z)+(Math.log(to.z)-Math.log(fr.z))*e);
    clamp();draw();if(k<1)anim=requestAnimationFrame(f);else{anim=null;resetFlow();}}anim=requestAnimationFrame(f);}
function focusZone(instant){if(!ZONE)return;var z=Math.min(VW,VH-130)*.42/(ZONE.R/KM);fly({cx:ZONE.lon,cy:ZONE.lat,z:Math.max(FITZ,Math.min(FITZ*14,z))},instant);}
function readout(x,y){var w=s2w(x,y),h=hAt(w[0],w[1]);if(!h){$("rp-rpos").textContent="Hors domaine";return;}
  var k=cellOf(w[0],w[1]),u=wind(w[0],w[1]),sp=Math.hypot(u[0],u[1]),dir=(Math.atan2(-u[0],-u[1])*180/Math.PI+360)%360;
  var t="<b>"+fmt(w[1],3)+"° N  "+fmt(w[0],3)+"° E</b>  ·  "+(h.c===0?"mer":"alt "+Math.round(h.h)+" m")+"  ·  vent "+fmt(sp,1)+" m/s du "+Math.round(dir)+"°";
  if(k>=0)t+="  ·  erreur <b>"+fmt(E0[k],2)+" → "+fmt(RES[k],2)+" m/s</b>  ·  <span class=\"hint\">clic : placer des ANEO ici</span>";$("rp-rpos").innerHTML=t;}
function pick(x,y){var v=visible(),best=null,bd=14*14,bz=null;
  if(ZONE)ZONE.pots.forEach(function(p){var s=w2s(p.lon,p.lat),dx=s[0]-x,dy=s[1]-(p.kind==="drone"?8:0)-y,dd=dx*dx+dy*dy;if(dd<bd){bd=dd;bz=p;}});
  if(bz){ZONE.sel=bz;tab("zone");draw();zonePanel();return;}
  bd=14*14;v.d.concat(CUSTOM.filter(function(o){return o.kind==="drone";})).forEach(function(o){var s=w2s(o.lon,o.lat),dx=s[0]-x,dy=s[1]-8-y,dd=dx*dx+dy*dy;if(dd<bd*1.6){bd=dd;best=o;}});
  if(!best)v.m.concat(CUSTOM.filter(function(o){return o.kind==="mast";})).forEach(function(o){var s=w2s(o.lon,o.lat),dx=s[0]-x,dy=s[1]-y,dd=dx*dx+dy*dy;if(dd<bd){bd=dd;best=o;}});
  if(best){select(best);return;}
  var w=s2w(x,y);if(pip(TN,w[0],w[1])){setZone(w[0],w[1]);afterZone();}}
function afterZone(){zMsg="";tab("zone");draw();zonePanel();focusZone();}

/* ================= onglets ================= */
function tab(t){st.tab=t;["zone","dim","site"].forEach(function(k){$("rp-tab-"+k).setAttribute("aria-selected",String(k===t));$("rp-p-"+k).hidden=k!==t;});}
["zone","dim","site"].forEach(function(k){$("rp-tab-"+k).addEventListener("click",function(){tab(k);});});

/* ================= panneau zone ================= */
var zFocus=null;
function zonePanel(){var P=$("rp-p-zone"),Z=ZONE;
  if(!Z){P.innerHTML='<div class="blk"><div class="h">Placer dans une zone</div><p class="empty">Cliquez une zone de la Tunisie sur la carte. ATOM Wind y place les ANEO potentiels selon l\'erreur locale, le relief et les contraintes de pose.</p></div>';return;}
  var S=Z.stats,g=S.e0>0?100*(1-S.e1/S.e0):0,nm=Z.pots.filter(function(p){return p.kind==="mast";}).length,nd=Z.pots.length-nm;
  var h='<div class="blk"><div class="idrow"><div><div class="h" style="margin-bottom:2px">Zone d\'étude</div><div class="id">'+Z.name+'</div></div><button class="btn" id="rp-zfoc">Recentrer</button></div>'
   +'<p class="sub">'+fmt(Z.lat,3)+'° N, '+fmt(Z.lon,3)+'° E · rayon '+Z.R+' km · '+terrainOf(Z.lon,Z.lat)+'</p>'
   +'<div class="zk"><div><small>Surface terrestre</small><b>'+fmt(S.area,0)+' km²</b></div><div><small>Relief</small><b>'+Math.round(S.hmin)+'–'+Math.round(S.hmax)+' m</b></div>'
   +'<div><small>Erreur locale</small><b>'+fmt(S.e0,2)+' → <span class="delta">'+fmt(S.e1,2)+'</span></b></div><div><small>Gain dans la zone</small><b class="delta">-'+fmt(g,1)+' %</b></div></div>'
   +'<div class="acts"><button class="btn pri" id="rp-z3d">Vue 3D de la zone</button><button class="btn sec" id="rp-zadd"'+(Z.pots.length?'':' disabled')+'>Ajouter '+Z.pots.length+' sites au plan</button><button class="btn" id="rp-zclr">Effacer</button></div>'
   +(zMsg?'<p class="toast">'+zMsg+'</p>':'')+'</div>'
   +'<div class="blk"><label class="h" for="rp-zr">Rayon de la zone <em>'+Z.R+' km</em></label><input type="range" class="am" id="rp-zr" min="10" max="60" step="5" value="'+Z.R+'">'
   +'<label class="h" for="rp-zm">Mâts potentiels <em>'+Z.nm+'</em></label><input type="range" id="rp-zm" min="0" max="30" step="1" value="'+Z.nm+'">'
   +'<label class="h" for="rp-zd">Profils drone potentiels <em>'+Z.nd+'</em></label><input type="range" class="dr" id="rp-zd" min="0" max="4" step="1" value="'+Z.nd+'">'
   +'<p class="sub">Contraintes de pose : terre ferme hors chott, altitude ≤ '+Z.limH+' m, pente ≤ 20°, au moins 2 km d\'un site du plan et 8 km d\'une station existante.</p></div>'
   +'<div class="blk"><div class="h">Sites potentiels <em>'+nm+' mâts · '+nd+' drone'+(nd>1?'s':'')+'</em></div><div class="tbx"><table><thead><tr><th>Site</th><th>Alt.</th><th>Pente</th><th>Terrain</th><th>Part du gain</th></tr></thead><tbody>'
   +Z.pots.map(function(p,i){return'<tr class="pick'+(Z.sel===p?' on':'')+'" data-i="'+i+'" tabindex="0"><td><span class="sw" style="background:'+(p.kind==="drone"?"#c4a2ff":"#f2b84b")+'"></span>'+p.id+'</td><td class="n">'+Math.round(p.h)+' m</td><td class="n">'+fmt(p.slope,1)+'°</td><td>'+p.terrain+'</td><td class="n">'+fmt(100*p.share,1)+' %</td></tr>';}).join("")
   +'</tbody></table></div>'+(Z.sel?zSel(Z.sel):'<p class="sub">Sélectionnez un site sur la carte ou dans la liste.</p>')+'</div>';
  P.innerHTML=h;
  $("rp-zfoc").onclick=function(){focusZone();};$("rp-z3d").onclick=function(){T3.open();};$("rp-zadd").onclick=acceptZone;
  $("rp-zclr").onclick=function(){ZONE=null;zMsg="";draw();zonePanel();if(T3.on)T3.close();};
  function sl(id,key){$(id).oninput=function(e){Z[key]=+e.target.value;zFocus=id;computeZone();draw();zonePanel();if(key==="R")focusZone(true);if(T3.on)T3.build(true);};}
  sl("rp-zr","R");sl("rp-zm","nm");sl("rp-zd","nd");
  Array.prototype.forEach.call(P.querySelectorAll("tr.pick"),function(tr){var f=function(){Z.sel=Z.pots[+tr.dataset.i];draw();zonePanel();if(T3.on)T3.highlight();};tr.onclick=f;tr.onkeydown=function(e){if(e.key==="Enter"||e.key===" "){e.preventDefault();f();}};});
  if(zFocus){var el=$(zFocus);if(el)el.focus();zFocus=null;}}
function zSel(p){var P=profData(p,st.mast);return'<dl class="kv" style="margin-top:6px"><dt>Site</dt><dd>'+p.id+(p.kind==="drone"?" · profil drone":" · mât "+st.mast+" m")+'</dd><dt>Position</dt><dd>'+fmt(p.lat,4)+'° N, '+fmt(p.lon,4)+'° E</dd><dt>Vent à 100 m / 1 km</dt><dd>'+fmt(P.truth(100),1)+' / '+fmt(P.truth(1000),1)+' m/s</dd><dt>Gain apporté</dt><dd class="delta">'+fmt(100*p.share,1)+' % du gain de la zone</dd></dl>';}

/* ================= fiche site ================= */
function allSites(){var v=visible();return v.d.concat(v.m).concat(CUSTOM);}
function select(o){st.sel=o||null;draw();inspector();if(o)tab("site");}
function inspector(){var hd=$("rp-insp-head"),bd=$("rp-insp-body"),o=st.sel;
  if(!o){hd.innerHTML='<div class="h">Fiche du site</div><p class="empty">Sélectionnez un mât ou un profil drone du plan sur la carte.</p>';bd.innerHTML='<p class="sub">La fiche donne la position, le terrain, l\'erreur locale avant et après le réseau, et le profil vertical attendu.</p>';profile(null);return;}
  var isD=o.kind==="drone",list=allSites(),i=list.indexOf(o),k=cellOf(o.lon,o.lat),h=hAt(o.lon,o.lat);
  hd.innerHTML='<div class="idrow"><div><div class="h" style="margin-bottom:2px">'+(isD?"Profil drone":"Mât Porteur")+(o.custom?" · ajouté depuis une zone":(!isD&&st.strat==="opt"?" · phase "+phaseOf(o.rank).n:""))+'</div><div class="id">'+o.id+'</div></div>'
    +'<div class="nav"><button id="rp-pv" aria-label="Site précédent">‹</button><button id="rp-nx" aria-label="Site suivant">›</button></div></div>'
    +'<span class="tag" style="color:'+(o.status==="installé"?"var(--ok)":isD?"var(--drone)":"var(--mast)")+';justify-self:start">'+o.status+'</span>';
  var rows=[["Position",fmt(o.lat,4)+"° N, "+fmt(o.lon,4)+"° E"],["Altitude du sol",h?Math.round(h.h)+" m":"-"],["Terrain",terrainOf(o.lon,o.lat)],
    [isD?"Mesure":"Hauteur de mesure",isD?"colonne 25 m à 1 km, 1 vol par jour":st.mast+" m, en continu"],
    ["Erreur locale sans réseau",k>=0?fmt(E0[k],2)+" m/s":"-"],["Erreur locale avec le réseau",k>=0?'<span class="delta">'+fmt(RES[k],2)+" m/s (-"+fmt(100*(E0[k]-RES[k])/E0[k],0)+" %)</span>":"-"]];
  if(isD)rows.push(["Autorisation de vol","à obtenir avant toute campagne"]);
  var canT=isD||o.custom||st.strat==="opt";
  bd.innerHTML='<dl class="kv">'+rows.map(function(r){return"<dt>"+r[0]+"</dt><dd>"+r[1]+"</dd>";}).join("")+'</dl>'
    +(canT?'<div class="acts"><button class="btn sec" id="rp-tgl">'+(o.status==="installé"?"Remettre en prévu":isD?"Marquer comme actif":"Marquer comme posé")+'</button></div>':'<p class="sub">Stratégie de comparaison : sélectionnez « Optimisé » pour planifier.</p>');
  $("rp-pv").onclick=function(){if(list.length)select(list[(i-1+list.length)%list.length]);};
  $("rp-nx").onclick=function(){if(list.length)select(list[(i+1)%list.length]);};
  var t=$("rp-tgl");if(t)t.onclick=function(){o.status=o.status==="installé"?"prévu":"installé";if(o.custom)saveCustom();if(o.os6)OS6.modifie=true;draw();inspector();phases();};
  profile(o);}
function profData(o,hm){var seed=hash(Math.round(o.lon*100),Math.round(o.lat*100)),G=9+4*seed,tr=terrainOf(o.lon,o.lat),z0=tr==="relief"?.5:tr==="côte"?.02:.1,jet=1.5+3*hash(Math.round(o.lat*90),7),hj=220+120*seed;
  function lg(z,p){return Math.pow(Math.log((z+z0)/z0)/Math.log((1000+z0)/z0),p);}
  return{truth:function(z){return G*lg(z,.55)+jet*Math.exp(-Math.pow((z-hj)/90,2));},era:function(z){return G*.93*lg(z,.42)+.25;},veer:function(z){return 25*z/1000;},hm:hm};}
function profile(o){var sv=$("rp-prof"),X0=40,X1=290,Y0=228,Y1=12,VM=18;
  function x(v){return X0+(X1-X0)*v/VM;}function y(z){return Y0-(Y0-Y1)*z/1000;}
  var h='';for(var z=0;z<=1000;z+=200)h+='<line x1="'+X0+'" x2="'+X1+'" y1="'+y(z)+'" y2="'+y(z)+'" stroke="#22343e"/><text x="'+(X0-6)+'" y="'+(y(z)+4)+'" text-anchor="end" font-size="10" fill="#8fa3aa" font-family="IBM Plex Mono, monospace">'+z+'</text>';
  for(var v=0;v<=VM;v+=6)h+='<text x="'+x(v)+'" y="'+(Y0+15)+'" text-anchor="middle" font-size="10" fill="#8fa3aa" font-family="IBM Plex Mono, monospace">'+v+'</text>';
  h+='<text x="'+X1+'" y="'+(Y0+27)+'" text-anchor="end" font-size="10" fill="#8fa3aa">vitesse (m/s)</text><text x="8" y="'+(Y1+2)+'" font-size="10" fill="#8fa3aa" transform="rotate(-90 8 '+(Y1+2)+')" text-anchor="end">hauteur au-dessus du sol (m)</text>';
  if(!o){sv.innerHTML=h;$("rp-prof-note").textContent="";return;}
  var isD=o.kind==="drone",hm=st.mast,P=profData(o,hm),T=[],E=[],A=[],B=[];
  for(var zz=5;zz<=1000;zz+=10){var tr=P.truth(zz),er=P.era(zz),a=er+.5*(tr-er),wv=isD?.85:.75*Math.exp(-Math.max(0,zz-hm)/140);T.push([zz,tr]);E.push([zz,er]);A.push([zz,a]);B.push([zz,a+wv*(tr-a)]);}
  function pl(arr,col,w,dash){return'<polyline fill="none" stroke="'+col+'" stroke-width="'+w+'"'+(dash?' stroke-dasharray="'+dash+'"':'')+' points="'+arr.map(function(p){return x(p[1]).toFixed(1)+","+y(p[0]).toFixed(1);}).join(" ")+'"/>';}
  h+=pl(E,"#7d8f96",1.4,"")+pl(T,"#e4ecee",1.2,"4 3")+pl(A,"#f2b84b",1.4,"")+pl(B,isD?"#c4a2ff":"#5ec4cf",2.4,"");
  if(isD){for(var q=50;q<=1000;q+=50)h+='<circle cx="'+x(P.truth(q)+(hash(q,Math.round(o.lon*100))-.5)*.5)+'" cy="'+y(q)+'" r="2.4" fill="#c4a2ff"/>';}
  else h+='<circle cx="'+x(P.truth(hm))+'" cy="'+y(hm)+'" r="4" fill="#5ec4cf" stroke="#0a1319" stroke-width="1.5"/>';
  function rm(arr){var s=0;for(var i=0;i<arr.length;i++)s+=Math.pow(arr[i][1]-T[i][1],2);return Math.sqrt(s/arr.length);}
  [["#7d8f96","ERA5 seule",rm(E)],["#f2b84b","Reconstruction sans réseau",rm(A)],[isD?"#c4a2ff":"#5ec4cf","Avec ce "+(isD?"profil":"mât"),rm(B)],["#e4ecee","Vérité d'entraînement",null]].forEach(function(l,i){var yy=Y1+8+i*14,xx=X0+8;
    h+='<rect x="'+(xx-4)+'" y="'+(yy-9)+'" width="150" height="13" fill="#0f1b22" opacity=".85"/><line x1="'+xx+'" x2="'+(xx+14)+'" y1="'+(yy-3)+'" y2="'+(yy-3)+'" stroke="'+l[0]+'" stroke-width="2"'+(l[2]==null?' stroke-dasharray="3 2"':'')+'/><text x="'+(xx+19)+'" y="'+yy+'" font-size="10" fill="#c9d6d9">'+l[1]+(l[2]!=null?" · "+fmt(l[2],2):"")+'</text>';});
  sv.innerHTML=h;$("rp-prof-note").textContent=isD?"colonne mesurée":"un point mesuré à "+hm+" m";}

/* ================= courbe ================= */
function curve(){if(OS6&&OS6.V.courbe){curveOS6();return;}var sv=$("rp-curve"),X0=36,X1=334,Y0=210,Y1=8,all=[CURVES.opt,CURVES.grid,CURVES.rand,CURVES.optd],mn=Infinity,cv=metric();
  all.forEach(function(c){c.forEach(function(v){if(v<mn)mn=v;});});mn=Math.min(mn,cv);var ymin=Math.floor((mn-.02)*10)/10,ymax=1.95;
  function x(n){return X0+(X1-X0)*Math.min(400,n)/400;}function y(v){return Y0-(Y0-Y1)*(v-ymin)/(ymax-ymin);}
  var h='';for(var v=Math.ceil(ymin*10)/10;v<=ymax+1e-9;v+=.1)h+='<line x1="'+X0+'" x2="'+X1+'" y1="'+y(v)+'" y2="'+y(v)+'" stroke="#22343e"/><text x="'+(X0-5)+'" y="'+(y(v)+3.5)+'" text-anchor="end" font-size="9.5" fill="#8fa3aa" font-family="IBM Plex Mono, monospace">'+fmt(v,1)+'</text>';
  [0,100,200,300,400].forEach(function(n){h+='<text x="'+x(n)+'" y="'+(Y0+13)+'" text-anchor="middle" font-size="9.5" fill="#8fa3aa" font-family="IBM Plex Mono, monospace">'+n+'</text>';});
  h+='<text x="'+X1+'" y="'+(Y0+25)+'" text-anchor="end" font-size="9.5" fill="#8fa3aa">mâts au sol</text><text x="'+X0+'" y="'+(Y0+25)+'" font-size="9.5" fill="#8fa3aa">m/s</text>';
  h+='<line x1="'+X0+'" x2="'+X1+'" y1="'+y(R0)+'" y2="'+y(R0)+'" stroke="#8fa3aa" stroke-dasharray="2 3"/><text x="'+X1+'" y="'+(y(R0)-4)+'" text-anchor="end" font-size="9.5" fill="#8fa3aa">sans réseau 1,893 (mesuré)</text>';
  h+='<line x1="'+X0+'" x2="'+X1+'" y1="'+y(TARGET)+'" y2="'+y(TARGET)+'" stroke="#ef6b5f" stroke-width="1.3" stroke-dasharray="6 4"/><text x="'+X1+'" y="'+(y(TARGET)+12)+'" text-anchor="end" font-size="9.5" fill="#ef6b5f">OS4 -10 % · '+fmt(TARGET,3)+'</text>';
  function pl(arr,col,w){return'<polyline fill="none" stroke="'+col+'" stroke-width="'+w+'" stroke-linejoin="round" points="'+arr.map(function(v,i){return x(i*5).toFixed(1)+","+y(v).toFixed(1);}).join(" ")+'"/>';}
  h+=pl(CURVES.rand,st.strat==="rand"?"#c9d6d9":"#5d727a",st.strat==="rand"?2.2:1.3)+pl(CURVES.grid,"#d9b04a",st.strat==="grid"?2.4:1.3)+pl(CURVES.opt,"#5ec4cf",st.strat==="opt"?2.4:1.3);
  if(st.d>0)h+=pl(CURVES.optd,"#c4a2ff",2.2);
  function cross(arr){for(var i=0;i<arr.length;i++)if(arr[i]<=TARGET)return i*5;return null;}
  var c0=cross(CURVES.opt),cd=cross(CURVES.optd);
  if(c0!=null)h+='<circle cx="'+x(c0)+'" cy="'+y(TARGET)+'" r="4" fill="#0f1b22" stroke="#5ec4cf" stroke-width="2"/>';
  if(st.d>0&&cd!=null)h+='<circle cx="'+x(cd)+'" cy="'+y(TARGET)+'" r="4" fill="#0f1b22" stroke="#c4a2ff" stroke-width="2"/>';
  var nTot=st.n+CUSTOM.filter(function(o){return o.kind==="mast";}).length;
  h+='<line x1="'+x(nTot)+'" x2="'+x(nTot)+'" y1="'+Y1+'" y2="'+Y0+'" stroke="#f2b84b" stroke-width="1" opacity=".7"/><circle cx="'+x(nTot)+'" cy="'+y(cv)+'" r="4.5" fill="#f2b84b" stroke="#0f1b22" stroke-width="2"/>';
  sv.innerHTML=h;
  var lg=[["#5ec4cf","Optimisé"],["#d9b04a","Grille"],["#8fa3aa","Aléatoire"]];if(st.d>0)lg.push(["#c4a2ff","Optimisé + "+st.d+" profils drone"]);lg.push(["#f2b84b","Plan actuel"+(CUSTOM.length?" (avec zones)":"")]);
  $("rp-c-lgd").innerHTML=lg.map(function(l){return'<span><i style="background:'+l[0]+'"></i>'+l[1]+'</span>';}).join("");
  $("rp-c-note").textContent=c0!=null?"OS4 à "+c0+" mâts"+(st.d>0&&cd!=null?" · "+cd+" avec drones":""):"OS4 non atteint";}

/* ================= phases, kpi ================= */
function phases(){panneauOS6();if(OS6){phasesOS6();return;}var tb=$("rp-phases"),h="",res=E0.slice(),s=SUM0,last=0;for(var i=0;i<st.d;i++)s-=apply(res,DRN[i].lon,DRN[i].lat,DRONE);
  var dg=100*(1-s/SUM0),di=DRN.slice(0,st.d).filter(function(o){return o.status==="installé";}).length;
  h+="<tr><td><span class='sw' style='background:#c4a2ff'></span>Drones</td><td class='n'>"+st.d+"</td><td class='n'>"+fmt(R0*s/SUM0,3)+"</td><td class='n'>-"+fmt(dg,1)+" %</td><td class='n'>"+di+"/"+st.d+"</td></tr>";
  PH.forEach(function(p){for(var q=last;q<p.b;q++)s-=apply(res,ORD.opt[q].lon,ORD.opt[q].lat,MAST);last=p.b;var posed=ORD.opt.slice(p.a-1,p.b).filter(function(o){return o.status==="installé";}).length,m=R0*s/SUM0;
    h+="<tr><td><span class='sw' style='background:"+p.c+"'></span>Phase "+p.n+"</td><td class='n'>"+p.a+"-"+p.b+"</td><td class='n'"+(m<=TARGET?" style='color:var(--ok)'":"")+">"+fmt(m,3)+"</td><td class='n'>-"+fmt(100*(1-s/SUM0),1)+" %</td><td class='n'>"+posed+"/"+(p.b-p.a+1)+"</td></tr>";});
  if(CUSTOM.length){var mm=metric();h+="<tr><td><span class='sw' style='background:#f2b84b'></span>Zones</td><td class='n'>"+CUSTOM.length+"</td><td class='n'>"+fmt(mm,3)+"</td><td class='n'>-"+fmt(100*(1-mm/R0),1)+" %</td><td class='n'>"+CUSTOM.filter(function(o){return o.status==="installé";}).length+"/"+CUSTOM.length+"</td></tr>";}
  tb.innerHTML=h;}
function kpi(){var lb=$("rp-k-os4").previousElementSibling;
  if(OS6&&OS6.V.courbe){var mo=mesureOS6(st.n),ac=mesureOS6(0),seuil=OS6.V.seuil_dso1_m_s;lb.textContent="Seuil DSO1";
    $("rp-k-err").textContent=mo?fmt(mo.pire_rmse_niveau_m_s,3)+" m/s":"—";
    $("rp-k-gain").textContent=(mo&&ac)?"-"+fmt(100*(1-mo.pire_rmse_niveau_m_s/ac.pire_rmse_niveau_m_s),1)+" %":"—";
    var okd=mo&&seuil!=null&&mo.pire_rmse_niveau_m_s<=seuil;$("rp-k-os4").innerHTML='<span style="color:'+(okd?"var(--ok)":"var(--warn)")+'">'+(mo&&seuil!=null?(okd?"respecté":"non respecté"):"—")+"</span>";
    $("rp-k-net").textContent=st.n+" mâts · "+st.d+" drones";$("rp-n-val").textContent=st.n;$("rp-d-val").textContent=st.d;return;}
  lb.textContent="Objectif OS4";var m=metric(),gn=100*(1-m/R0);$("rp-k-err").textContent=fmt(m,3)+" m/s";$("rp-k-gain").textContent="-"+fmt(gn,1)+" %";
  var ok=m<=TARGET;$("rp-k-os4").innerHTML='<span style="color:'+(ok?"var(--ok)":"var(--warn)")+'">'+(ok?"atteint":"non atteint")+"</span>";
  var cm=CUSTOM.filter(function(o){return o.kind==="mast";}).length,cdn=CUSTOM.length-cm;
  $("rp-k-net").textContent=(st.n+cm)+" mâts · "+(st.d+cdn)+" drones";$("rp-n-val").textContent=st.n;$("rp-d-val").textContent=st.d;}

/* ================= vue 3D de zone ================= */
var T3={on:false};
(function(){var host=$("rp-v3d"),R=null,scene,cam,ctl,root=null,parts=null,pGeo,pPos,lblBox=$("rp-lbls"),card=$("rp-card3"),pick3=[],cur=null,ray,mouse,clock=0,started=false,B=10,N=170,Hm=null;
  function ok3(){return typeof THREE!=="undefined"&&THREE.OrbitControls;}
  // three.js >= r152 gère les couleurs en espace linéaire : les couleurs de sommets de la maquette (sRGB) sont converties
  function LIN(v){return(typeof THREE!=="undefined"&&+THREE.REVISION>=152)?Math.pow(Math.max(0,v),2.2):v;}
  function kmXY(lon,lat){return[(lon-ZONE.lon)*K*KM,-(lat-ZONE.lat)*KM];}
  function llOf(x,z){return[ZONE.lon+x/(K*KM),ZONE.lat-z/KM];}
  function yOf(hm){return hm/1000*(+$("rp-ex3").value);}
  function hAt3(x,z){var gx=Math.round((x+B)/(2*B)*N),gz=Math.round((z+B)/(2*B)*N);gx=Math.max(0,Math.min(N,gx));gz=Math.max(0,Math.min(N,gz));return Math.max(0,Hm[gz*(N+1)+gx]);}
  T3.open=function(){if(!ZONE)return;
    if(!ok3()&&window.ATOM&&ATOM.loadThree&&!T3.charge){T3.charge=true;ATOM.loadThree().then(function(){T3.open();},function(){T3.open();});return;}
    host.hidden=false;
    if(!ok3()){if(!host.querySelector(".msg3"))host.insertAdjacentHTML("beforeend",'<div class="msg3"><div>La vue 3D demande WebGL et la bibliothèque three.js, qui n\'a pas pu être chargée.<br><br><button class="btn" id="rp-b3x">Retour à la carte</button></div></div>');var bb=$("rp-b3x");if(bb)bb.onclick=function(){host.hidden=true;};return;}
    T3.on=true;
    if(!R){try{R=new THREE.WebGLRenderer({antialias:true});}catch(e){R=null;}
      if(!R){host.insertAdjacentHTML("beforeend",'<div class="msg3">WebGL n\'est pas disponible dans ce navigateur.</div>');return;}
      R.setPixelRatio(Math.min(2,window.devicePixelRatio||1));host.insertBefore(R.domElement,host.firstChild);
      scene=new THREE.Scene();scene.background=new THREE.Color(0x08121a);cam=new THREE.PerspectiveCamera(42,1,.05,3000);
      ctl=new THREE.OrbitControls(cam,R.domElement);ctl.enableDamping=true;ctl.dampingFactor=.08;ctl.maxPolarAngle=Math.PI*.47;ctl.autoRotateSpeed=.35;
      ctl.addEventListener("start",function(){ctl.autoRotate=false;});
      var LI=(+THREE.REVISION>=155)?Math.PI:1;scene.add(new THREE.HemisphereLight(0xb9d3dc,0x1c1a12,.5*LI));var sun=new THREE.DirectionalLight(0xffecd0,.7*LI);sun.position.set(-60,80,-40);scene.add(sun);
      ray=new THREE.Raycaster();mouse=new THREE.Vector2();
      R.domElement.addEventListener("pointerdown",function(e){R.domElement._d=[e.clientX,e.clientY];});
      R.domElement.addEventListener("pointerup",function(e){var d=R.domElement._d;if(!d||Math.abs(e.clientX-d[0])+Math.abs(e.clientY-d[1])>4)return;var r=R.domElement.getBoundingClientRect();
        mouse.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(mouse,cam);var hit=ray.intersectObjects(pick3,true)[0];
        if(hit){var o=hit.object;while(o&&!o.userData.site)o=o.parent;if(o){var s=o.userData.site;if(s.zone){ZONE.sel=s;zonePanel();draw();}cur=s;T3.highlight();}}});
      $("rp-ex3").addEventListener("input",function(){$("rp-ex-v").textContent=$("rp-ex3").value;T3.build(true);});
      ["rp-w3","rp-e3","rp-i3"].forEach(function(id){$(id).addEventListener("change",function(){T3.build(true);});});
      $("rp-l3").addEventListener("change",function(){lblBox.hidden=!$("rp-l3").checked;});
      $("rp-b3").addEventListener("click",T3.close);
      document.addEventListener("keydown",function(e){if(e.key==="Escape"&&T3.on)T3.close();});}
    T3.resize();T3.build(false);cur=ZONE.sel||null;T3.highlight();fx.setTransform(1,0,0,1,0,0);fx.clearRect(0,0,flow.width,flow.height);if(!started){started=true;loop();}};
  T3.close=function(){host.hidden=true;T3.on=false;resetFlow();};
  T3.resize=function(){if(!R)return;var r=host.getBoundingClientRect();R.setSize(Math.max(1,r.width),Math.max(1,r.height),false);R.domElement.style.width="100%";R.domElement.style.height="100%";cam.aspect=r.width/Math.max(1,r.height);cam.updateProjectionMatrix();};
  function disposeAll(){if(!root)return;scene.remove(root);root.traverse(function(o){if(o.geometry)o.geometry.dispose();if(o.material){(Array.isArray(o.material)?o.material:[o.material]).forEach(function(m){m.dispose();});}});root=null;}
  T3.build=function(keepCam){if(!R||!ZONE)return;disposeAll();root=new THREE.Group();scene.add(root);pick3=[];lblBox.innerHTML="";T3.labels=[];
    var Z=ZONE,showErr=$("rp-e3").checked;B=Z.R*1.3;
    $("rp-t3s").textContent=Z.name+" · rayon "+Z.R+" km · "+fmt(Z.lat,3)+"° N "+fmt(Z.lon,3)+"° E";
    var geo=new THREE.PlaneGeometry(2*B,2*B,N,N);geo.rotateX(-Math.PI/2);var pos=geo.attributes.position,cols=new Float32Array(pos.count*3),fld=showErr?fieldOf(st.strat,st.n,st.d,Z.pots):null;Hm=new Float32Array(pos.count);
    for(var i=0;i<pos.count;i++){var x=pos.getX(i),z=pos.getZ(i),ll=llOf(x,z),sea=pip(SEA,ll[0],ll[1]),h=sea?-25:hFine(ll[0],ll[1]),inT=pip(TN,ll[0],ll[1]);Hm[i]=h;pos.setY(i,yOf(h));
      var c;if(sea)c=[22,52,70];else{c=landCol(ll[0],ll[1],h,HYPSO3);if(showErr&&inT){var ev=resAt(fld,ll[0],ll[1]);if(ev!=null){var ec=CRES(ev*(.78+.55*rough(ll[0],ll[1])));c=[c[0]*.35+ec[0]*.65,c[1]*.35+ec[1]*.65,c[2]*.35+ec[2]*.65];}}
        if(!inT){var mm=(c[0]+c[1]+c[2])/3;c=[(c[0]+mm)/2*.55,(c[1]+mm)/2*.55,(c[2]+mm)/2*.6];}}
      var dd=Math.sqrt(x*x+z*z),fade=dd>Z.R?Math.max(.38,1-(dd-Z.R)/(B-Z.R)*.62):1;cols[i*3]=LIN(c[0]/255*fade);cols[i*3+1]=LIN(c[1]/255*fade);cols[i*3+2]=LIN(c[2]/255*fade);}
    geo.setAttribute("color",new THREE.BufferAttribute(cols,3));geo.computeVertexNormals();
    root.add(new THREE.Mesh(geo,new THREE.MeshLambertMaterial({vertexColors:true})));
    var sea2=new THREE.Mesh(new THREE.PlaneGeometry(2*B,2*B),new THREE.MeshBasicMaterial({color:0x1b4a63,transparent:true,opacity:.55}));sea2.rotation.x=-Math.PI/2;sea2.position.y=.004;root.add(sea2);
    var skirt=new THREE.Mesh(new THREE.BoxGeometry(2*B,.6,2*B),new THREE.MeshLambertMaterial({color:0x0c171e}));skirt.position.y=yOf(-25)-.31;root.add(skirt);
    var ring=[];for(var a=0;a<=128;a++){var an=a/128*Math.PI*2,rx=Math.cos(an)*Z.R,rz=Math.sin(an)*Z.R;ring.push(new THREE.Vector3(rx,yOf(hAt3(rx,rz))+.15,rz));}
    var rl=new THREE.Line(new THREE.BufferGeometry().setFromPoints(ring),new THREE.LineDashedMaterial({color:0xf2b84b,dashSize:1.2,gapSize:.8}));rl.computeLineDistances();root.add(rl);
    var MH=Math.max(1.6,B*.06);T3.MH=MH;
    function disc(g2,L,col,op){var dsk=new THREE.Mesh(new THREE.CircleGeometry(L,48),new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:op,depthWrite:false}));dsk.rotation.x=-Math.PI/2;dsk.position.y=.08;g2.add(dsk);
      var pts=[];for(var a2=0;a2<=64;a2++){var an2=a2/64*Math.PI*2;pts.push(new THREE.Vector3(Math.cos(an2)*L,.1,Math.sin(an2)*L));}g2.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:col,transparent:true,opacity:.45})));}
    function mast(s,col,zone){var p=kmXY(s.lon,s.lat),g2=new THREE.Group();g2.position.set(p[0],yOf(hAt3(p[0],p[1])),p[1]);
      var pole=new THREE.Mesh(new THREE.CylinderGeometry(.07,.11,MH,8),new THREE.MeshLambertMaterial({color:col}));pole.position.y=MH/2;g2.add(pole);
      var head=new THREE.Mesh(new THREE.BoxGeometry(.42,.22,.42),new THREE.MeshLambertMaterial({color:col,emissive:col,emissiveIntensity:.35}));head.position.y=MH;g2.add(head);
      var blade=new THREE.Mesh(new THREE.BoxGeometry(.9,.03,.08),new THREE.MeshLambertMaterial({color:0xe4ecee}));blade.position.y=MH+.2;blade.userData.spin=true;g2.add(blade);
      if($("rp-i3").checked)disc(g2,zone?ZL.mast.L:MAST.L,col,.11);g2.userData.site=s;root.add(g2);pick3.push(g2);}
    function drone(s,col,zone){var p=kmXY(s.lon,s.lat),g2=new THREE.Group(),top=yOf(1000);g2.position.set(p[0],yOf(hAt3(p[0],p[1])),p[1]);
      var beam=new THREE.Mesh(new THREE.CylinderGeometry(.5,.5,top,24,1,true),new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:.13,side:THREE.DoubleSide,depthWrite:false}));beam.position.y=top/2;g2.add(beam);
      var axis=new THREE.Mesh(new THREE.CylinderGeometry(.04,.04,top,6),new THREE.MeshBasicMaterial({color:col}));axis.position.y=top/2;g2.add(axis);
      var P=profData(s,st.mast),u=wind(s.lon,s.lat),b0=Math.atan2(u[1],u[0]);
      for(var zz=100;zz<=1000;zz+=100){var sp=P.truth(zz),c=WIND(sp),cc=new THREE.Color(c[0]/255,c[1]/255,c[2]/255),yy=yOf(zz);
        var tor=new THREE.Mesh(new THREE.TorusGeometry(.5,.045,6,32),new THREE.MeshBasicMaterial({color:cc}));tor.rotation.x=Math.PI/2;tor.position.y=yy;g2.add(tor);
        var ang=b0+P.veer(zz)*Math.PI/180;g2.add(new THREE.ArrowHelper(new THREE.Vector3(Math.cos(ang),0,-Math.sin(ang)),new THREE.Vector3(0,yy,0),.35+sp*.14,cc.getHex(),.28,.18));}
      var cone=new THREE.Mesh(new THREE.ConeGeometry(.28,.5,4),new THREE.MeshLambertMaterial({color:col,emissive:col,emissiveIntensity:.4}));cone.position.y=top+.4;cone.rotation.x=Math.PI;cone.userData.bob=top+.4;g2.add(cone);
      if($("rp-i3").checked)disc(g2,zone?ZL.drone.L:DRONE.L,col,.06);g2.userData.site=s;root.add(g2);pick3.push(g2);}
    function inBox(s){var p=kmXY(s.lon,s.lat);return Math.abs(p[0])<B&&Math.abs(p[1])<B;}
    var pv=visible();pv.m.concat(CUSTOM.filter(function(o){return o.kind==="mast";})).filter(inBox).forEach(function(s){mast(s,0x5ec4cf,false);});
    pv.d.concat(CUSTOM.filter(function(o){return o.kind==="drone";})).filter(inBox).forEach(function(s){drone(s,0xc4a2ff,false);});
    Z.pots.forEach(function(s){if(s.kind==="drone")drone(s,0xc4a2ff,true);else mast(s,0xf2b84b,true);});
    AIR.forEach(function(a3){var s={lon:a3[1],lat:a3[2],id:a3[0],kind:"station"};if(!inBox(s))return;var p=kmXY(s.lon,s.lat),m=new THREE.Mesh(new THREE.OctahedronGeometry(.45),new THREE.MeshLambertMaterial({color:0xe4ecee,emissive:0x555555}));m.position.set(p[0],yOf(hAt3(p[0],p[1]))+.6,p[1]);m.userData.site=s;root.add(m);pick3.push(m);});
    function label(txt,cls,v){var d=document.createElement("div");d.className="lbl3 "+cls;d.textContent=txt;lblBox.appendChild(d);T3.labels.push({el:d,v:v});}
    Z.pots.forEach(function(s){var p=kmXY(s.lon,s.lat);label(s.id,s.kind==="drone"?"d":"",new THREE.Vector3(p[0],yOf(hAt3(p[0],p[1]))+(s.kind==="drone"?yOf(1000)+.9:MH+.5),p[1]));});
    CITIES.forEach(function(c){var s={lon:c[1],lat:c[2]};if(!inBox(s))return;var p=kmXY(s.lon,s.lat);label(c[0],"c",new THREE.Vector3(p[0],yOf(hAt3(p[0],p[1]))+.4,p[1]));});
    lblBox.hidden=!$("rp-l3").checked;
    parts=null;T3.P=null;if($("rp-w3").checked){if(RM){for(var gx=-B;gx<=B;gx+=B/7)for(var gz=-B;gz<=B;gz+=B/7){var ll2=llOf(gx,gz),u2=wind(ll2[0],ll2[1]),sp2=Math.hypot(u2[0],u2[1]),c2=WIND(sp2);
          root.add(new THREE.ArrowHelper(new THREE.Vector3(u2[0]/sp2,0,-u2[1]/sp2),new THREE.Vector3(gx,yOf(hAt3(gx,gz)+150),gz),.6+sp2*.18,new THREE.Color(c2[0]/255,c2[1]/255,c2[2]/255).getHex(),.3,.2));}}
      else{var NPp=1600;pGeo=new THREE.BufferGeometry();pPos=new Float32Array(NPp*3);T3.P=[];for(var q=0;q<NPp;q++)T3.P.push({x:(Math.random()*2-1)*B,z:(Math.random()*2-1)*B,a:Math.floor(Math.random()*120)});
        pGeo.setAttribute("position",new THREE.BufferAttribute(pPos,3));pGeo.setAttribute("color",new THREE.BufferAttribute(new Float32Array(NPp*3),3));
        parts=new THREE.Points(pGeo,new THREE.PointsMaterial({size:Math.max(.18,B*.007),vertexColors:true,transparent:true,opacity:.9,depthWrite:false}));root.add(parts);}}
    if(!keepCam){var d0=Z.R*2.7,hmx=yOf(Z.stats.hmax||0);cam.position.set(d0*.75,d0*.8+hmx,d0*.95);ctl.target.set(0,yOf(((Z.stats.hmin||0)+(Z.stats.hmax||0))/2),0);ctl.update();ctl.autoRotate=!RM;}
    scene.fog=new THREE.Fog(0x08121a,B*2.4,B*6);T3.highlight();};
  T3.highlight=function(){if(!root)return;if(ZONE&&ZONE.sel)cur=ZONE.sel;var s=cur;
    root.children.forEach(function(o){if(o.userData&&o.userData.site)o.scale.setScalar(o.userData.site===s?1.35:1);});
    if(!s){card.innerHTML='<div class="h">Vue 3D</div><p class="sub">Glissez pour tourner, molette pour zoomer, clic droit pour déplacer. Cliquez un site pour sa fiche.</p>';return;}
    if(s.kind==="station"){card.innerHTML='<div class="h">Station METAR existante</div><div class="id">'+s.id+'</div><p class="sub">Vent mesuré à 10 m, sert d\'entrée au modèle.</p>';return;}
    var p=kmXY(s.lon,s.lat),P=profData(s,st.mast);
    card.innerHTML='<div class="h">'+(s.zone?"ANEO potentiel":"Site du plan")+' · '+(s.kind==="drone"?"profil drone":"mât "+st.mast+" m")+'</div><div class="id">'+s.id+'</div>'
      +'<dl class="kv"><dt>Altitude</dt><dd>'+Math.round(hAt3(p[0],p[1]))+' m</dd><dt>Terrain</dt><dd>'+terrainOf(s.lon,s.lat)+'</dd><dt>Vent à 100 m</dt><dd>'+fmt(P.truth(100),1)+' m/s</dd><dt>Vent à 1 km</dt><dd>'+fmt(P.truth(1000),1)+' m/s</dd>'
      +(s.zone?'<dt>Part du gain de la zone</dt><dd class="delta">'+fmt(100*s.share,1)+' %</dd>':'')+'</dl>';};
  function loop(){requestAnimationFrame(loop);if(!T3.on||!R||!root||!host.offsetParent)return;clock+=.016;ctl.update();
    if(parts&&T3.P){var pc=pGeo.attributes.color.array,kk=.045;for(var i=0;i<T3.P.length;i++){var p=T3.P[i],ll=llOf(p.x,p.z),u=wind(ll[0],ll[1]);p.x+=u[0]*kk;p.z-=u[1]*kk;p.a++;
        if(p.a>140||Math.abs(p.x)>B||Math.abs(p.z)>B){p.x=(Math.random()*2-1)*B;p.z=(Math.random()*2-1)*B;p.a=0;}
        pPos[i*3]=p.x;pPos[i*3+1]=yOf(hAt3(p.x,p.z)+150);pPos[i*3+2]=p.z;var c=WIND(Math.hypot(u[0],u[1])),f=Math.min(1,p.a/20)*Math.min(1,(140-p.a)/20);pc[i*3]=LIN(c[0]/255*f);pc[i*3+1]=LIN(c[1]/255*f);pc[i*3+2]=LIN(c[2]/255*f);}
      pGeo.attributes.position.needsUpdate=true;pGeo.attributes.color.needsUpdate=true;}
    if(!RM)root.traverse(function(o){if(o.userData.spin)o.rotation.y+=.12;if(o.userData.bob!=null)o.position.y=o.userData.bob+Math.sin(clock*2)*.12;});
    R.render(scene,cam);
    var r=host.getBoundingClientRect();if(T3.labels&&!lblBox.hidden)T3.labels.forEach(function(L){var v=L.v.clone().project(cam),vis=v.z<1&&v.x>-1.1&&v.x<1.1&&v.y>-1.1&&v.y<1.1;L.el.style.display=vis?"":"none";if(vis){L.el.style.left=((v.x+1)/2*r.width)+"px";L.el.style.top=((1-v.y)/2*r.height)+"px";}});
    var az=Math.atan2(cam.position.x-ctl.target.x,cam.position.z-ctl.target.z);$("rp-cmpg").setAttribute("transform","rotate("+(az*180/Math.PI)+")");}
})();

/* ================= service PC-FNO3D : plan OS6 mesuré ================= */
var OS6=null,STRAT_OS6={opt:"optimise",grid:"grille",rand:"hasard"};
function brancherOS6(V){
  if(!V||V.demonstration||!Array.isArray(V.capteurs))return false;
  var pts=V.capteurs.filter(function(c){return c.lat!=null&&c.lon!=null&&isFinite(c.lat)&&isFinite(c.lon);}).sort(function(a,b){return(a.ordre||0)-(b.ordre||0);});
  if(!pts.length)return false;
  OS6={V:V,modifie:false,message:OS6?OS6.message:""};
  ORD.opt=pts.map(function(c,i){return{lon:c.lon,lat:c.lat,id:c.id,rank:c.ordre||i+1,kind:"mast",status:c.statut==="installé"?"installé":"prévu",os6:c};});
  var cols=["#8fe3ec","#5ec4cf","#3c8f9c","#2b6570"];
  if(Array.isArray(V.phases)&&V.phases.length)PH=V.phases.map(function(q,i){return{n:q.phase,a:q.de,b:q.a,c:cols[Math.min(i,3)],os6:q};});
  $("rp-n").max=String(ORD.opt.length);$("rp-n").step="1";st.n=Math.min(Math.max(st.n,1),ORD.opt.length);st.strat="opt";
  var fl=$("rp-flag-data");if(fl){fl.title="Plan et courbe : sorties OS6 du modèle servi. Relief, vent et couche d'erreur : illustratifs.";
    fl.innerHTML='<i style="background:var(--ok)"></i>Plan OS6 · <strong>'+ORD.opt.length+' capteurs</strong> · relief et erreur d\'exemple';}
  return true;}
// pas de condition sur ATOM.pret() : le plan OS6 ne dépend pas de la journée chargée ; sans service, la requête échoue et la maquette garde ses données d'exemple
function chargerOS6(){var A=window.ATOM;if(!A||!A.api)return;
  A.api.get("/atom-wind/reseau-aneo?detail=true",60000).then(function(V){if(brancherOS6(V))renderAll();},function(){});}
function mesureOS6(n){var c=OS6&&OS6.V.courbe;if(!c)return null;if(n<=0)return c.actuel||OS6.V.reseau_actuel||null;
  var L=(c.strategies[STRAT_OS6[st.strat]]||[]).filter(function(q){return q.n<=n;});return L.length?L[L.length-1]:(c.actuel||OS6.V.reseau_actuel||null);}
function curveOS6(){var sv=$("rp-curve"),c=OS6.V.courbe,X0=36,X1=334,Y0=210,Y1=8,a=c.actuel||OS6.V.reseau_actuel,seuil=OS6.V.seuil_dso1_m_s;
  var all=[];Object.keys(c.strategies).forEach(function(k){c.strategies[k].forEach(function(q){all.push(q.pire_rmse_niveau_m_s);});});if(a)all.push(a.pire_rmse_niveau_m_s);if(seuil!=null)all.push(seuil);
  var mn=Math.min.apply(null,all)*.97,mx=Math.max.apply(null,all)*1.03,nMax=Math.max.apply(null,[1].concat(c.tailles||[]));
  function x(n){return X0+(X1-X0)*n/nMax;}function y(v){return Y0-(Y0-Y1)*(v-mn)/Math.max(1e-9,mx-mn);}
  var h='';[mn,(mn+mx)/2,mx].forEach(function(v){h+='<line x1="'+X0+'" x2="'+X1+'" y1="'+y(v)+'" y2="'+y(v)+'" stroke="#22343e"/><text x="'+(X0-5)+'" y="'+(y(v)+3.5)+'" text-anchor="end" font-size="9.5" fill="#8fa3aa" font-family="IBM Plex Mono, monospace">'+fmt(v,2)+'</text>';});
  [0].concat(c.tailles||[]).forEach(function(n){h+='<text x="'+x(n)+'" y="'+(Y0+13)+'" text-anchor="middle" font-size="9.5" fill="#8fa3aa" font-family="IBM Plex Mono, monospace">'+n+'</text>';});
  h+='<text x="'+X1+'" y="'+(Y0+25)+'" text-anchor="end" font-size="9.5" fill="#8fa3aa">capteurs ajoutés</text><text x="'+X0+'" y="'+(Y0+25)+'" font-size="9.5" fill="#8fa3aa">m/s</text>';
  if(a)h+='<line x1="'+X0+'" x2="'+X1+'" y1="'+y(a.pire_rmse_niveau_m_s)+'" y2="'+y(a.pire_rmse_niveau_m_s)+'" stroke="#8fa3aa" stroke-dasharray="2 3"/><text x="'+X1+'" y="'+(y(a.pire_rmse_niveau_m_s)-4)+'" text-anchor="end" font-size="9.5" fill="#8fa3aa">réseau actuel '+fmt(a.pire_rmse_niveau_m_s,3)+'</text>';
  if(seuil!=null)h+='<line x1="'+X0+'" x2="'+X1+'" y1="'+y(seuil)+'" y2="'+y(seuil)+'" stroke="#ef6b5f" stroke-width="1.3" stroke-dasharray="6 4"/><text x="'+X1+'" y="'+(y(seuil)+12)+'" text-anchor="end" font-size="9.5" fill="#ef6b5f">seuil DSO1 '+fmt(seuil,2)+'</text>';
  var coul={optimise:"#5ec4cf",grille:"#d9b04a",hasard:"#8fa3aa"};
  Object.keys(coul).forEach(function(k){var L=(a?[{n:0,pire_rmse_niveau_m_s:a.pire_rmse_niveau_m_s}]:[]).concat(c.strategies[k]||[]);if(!L.length)return;var on=STRAT_OS6[st.strat]===k;
    h+='<polyline fill="none" stroke="'+coul[k]+'" stroke-width="'+(on?2.4:1.3)+'" stroke-linejoin="round" points="'+L.map(function(q){return x(q.n).toFixed(1)+","+y(q.pire_rmse_niveau_m_s).toFixed(1);}).join(" ")+'"/>';
    L.forEach(function(q){h+='<circle cx="'+x(q.n).toFixed(1)+'" cy="'+y(q.pire_rmse_niveau_m_s).toFixed(1)+'" r="'+(on?3:2)+'" fill="'+coul[k]+'"/>';});});
  var m=mesureOS6(st.n);if(m)h+='<line x1="'+x(Math.min(st.n,nMax))+'" x2="'+x(Math.min(st.n,nMax))+'" y1="'+Y1+'" y2="'+Y0+'" stroke="#f2b84b" opacity=".7"/>';
  sv.innerHTML=h;
  $("rp-c-lgd").innerHTML=[["#5ec4cf","Optimisé"],["#d9b04a","Grille"],["#8fa3aa","Aléatoire (moyenne de trois tirages)"]].map(function(l){return'<span><i style="background:'+l[0]+'"></i>'+l[1]+'</span>';}).join("");
  $("rp-c-note").textContent=OS6.V.nombre_recommande!=null?"DSO1 à "+OS6.V.nombre_recommande+" capteurs":"DSO1 non atteint";
  $("rp-c-sub").textContent="Pire RMSE de niveau sur le test ("+(c.jeu||"test")+"), mesurée par OS6 aux seuls paliers de la courbe : rien n'est interpolé.";}
function phasesOS6(){var tb=$("rp-phases"),h="";
  PH.forEach(function(p){var q=p.os6||{},posed=ORD.opt.slice(p.a-1,p.b).filter(function(o){return o.status==="installé";}).length;
    h+="<tr><td><span class='sw' style='background:"+p.c+"'></span>Phase "+p.n+"</td><td class='n'>"+p.a+"-"+p.b+"</td><td class='n'>"+(q.pire_rmse_niveau_m_s!=null?fmt(q.pire_rmse_niveau_m_s,3):"—")+"</td><td class='n'>"+(q.gain_pire_niveau_m_s!=null?"-"+fmt(q.gain_pire_niveau_m_s,3)+" m/s":"—")+"</td><td class='n'>"+posed+"/"+(p.b-p.a+1)+"</td></tr>";});
  if(CUSTOM.length)h+="<tr><td><span class='sw' style='background:#f2b84b'></span>Zones</td><td class='n'>"+CUSTOM.length+"</td><td class='n'>—</td><td class='n'>—</td><td class='n'>"+CUSTOM.filter(function(o){return o.status==="installé";}).length+"/"+CUSTOM.length+"</td></tr>";
  tb.innerHTML=h;}
function panneauOS6(){var B=$("rp-os6");if(!B)return;if(!OS6){B.innerHTML="";return;}var V=OS6.V,r=V.relance||{};var esc=function(t){return String(t==null?"":t).replace(/[&<>"']/g,function(c){return{"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c];});};
  B.innerHTML='<div class="acts" style="margin-top:8px"><button class="btn sec" id="rp-os6-save"'+(OS6.modifie?"":" disabled")+'>Enregistrer le statut posé</button>'
    +(OS6.message?'<span class="toast">'+esc(OS6.message)+'</span>':'')+'</div>'
    +(r.commande?'<p class="sub" style="margin-top:8px">'+esc(r.note||"Relance du plan OS6 (carnet GPU)")+'</p><textarea readonly aria-label="Commande de relance OS6">'+esc(r.commande)+'</textarea>':'')
    +((V.fichiers||[]).length?'<div class="acts">'+V.fichiers.filter(function(f){return /\.(csv|geojson|json|png)$/.test(f);}).map(function(f){return'<button class="btn" data-rp-brut="'+esc(f)+'">'+esc(f)+'</button>';}).join("")+'</div>':'');
  var b=$("rp-os6-save");if(b)b.onclick=function(){b.disabled=true;var ids=ORD.opt.filter(function(o){return o.status==="installé";}).map(function(o){return o.id;});
    window.ATOM.api.post("/atom-wind/reseau-aneo/installes",{installes:ids}).then(function(r){
      if(r.erreur){OS6.message="Enregistrement refusé : "+r.erreur;panneauOS6();return;}
      OS6.message=r.installes+" capteur(s) posé(s) enregistré(s) dans "+r.fichier+" ; relancez le plan avec la commande ci-dessous.";OS6.modifie=false;chargerOS6();panneauOS6();});};
  Array.prototype.forEach.call(B.querySelectorAll("[data-rp-brut]"),function(bt){bt.onclick=function(){window.ATOM.api.blob("/atom-wind/reseau-aneo/fichier/"+encodeURIComponent(bt.dataset.rpBrut)).then(function(u){
    var a=document.createElement("a");a.href=u;a.download=bt.dataset.rpBrut;document.body.appendChild(a);a.click();a.remove();},function(e){OS6.message="Export impossible : "+e.message;panneauOS6();});};});}

/* ================= rendu ================= */
function renderAll(){recompute();if(ZONE)computeZone();kpi();legendBar();draw();curve();phases();
  if(st.sel&&allSites().indexOf(st.sel)<0)st.sel=null;inspector();zonePanel();
  ["opt","grid","rand"].forEach(function(s){$("rp-s-"+s).setAttribute("aria-pressed",String(st.strat===s));});
  $("rp-n").value=st.n;$("rp-d").value=st.d;$("rp-layer").value=st.layer;if(T3.on)T3.build(true);}
$("rp-n").addEventListener("input",function(e){st.n=+e.target.value;save();renderAll();});
$("rp-d").addEventListener("input",function(e){st.d=+e.target.value;save();renderAll();});
$("rp-layer").addEventListener("change",function(e){st.layer=e.target.value;save();renderAll();});
$("rp-mast").addEventListener("change",function(e){st.mast=+e.target.value;inspector();zonePanel();if(T3.on)T3.highlight();});
["opt","grid","rand"].forEach(function(s){$("rp-s-"+s).addEventListener("click",function(){st.strat=s;save();renderAll();});});
[["rp-t-relief","relief"],["rp-t-ring","ring"],["rp-t-zones","zones"],["rp-t-air","air"]].forEach(function(t){$(t[0]).addEventListener("change",function(e){st[t[1]]=e.target.checked;draw();});});
$("rp-t-wind").checked=st.wind;$("rp-t-wind").addEventListener("change",function(e){st.wind=e.target.checked;resetFlow();});

/* export */
function rowsAll(){return ORD.opt.slice(0,st.strat==="opt"?st.n:0).map(function(o){return{id:o.id,type:"mat_sol",prio:o.rank,phase:phaseOf(o.rank).n,lat:o.lat,lon:o.lon,haut:st.mast+" m",statut:o.status};})
  .concat(DRN.slice(0,st.d).map(function(o){return{id:o.id,type:"profil_drone",prio:o.rank,phase:"",lat:o.lat,lon:o.lon,haut:"25-1000 m",statut:o.status};}))
  .concat(CUSTOM.map(function(o){return{id:o.id,type:o.kind==="drone"?"profil_drone":"mat_sol",prio:"zone",phase:"",lat:o.lat,lon:o.lon,haut:o.kind==="drone"?"25-1000 m":st.mast+" m",statut:o.status};}));}
function copy(text,label){var out=$("rp-x-out");out.hidden=false;out.value=text;var fail=function(){out.select();$("rp-toast").textContent="Copie bloquée : sélectionnez le texte.";};
  try{navigator.clipboard.writeText(text).then(function(){$("rp-toast").textContent=label+" copié.";},fail);}catch(e){fail();}}
$("rp-x-csv").addEventListener("click",function(){copy("id,type,priorite,phase,latitude,longitude,hauteur,statut\n"+rowsAll().map(function(r){return[r.id,r.type,r.prio,r.phase,r.lat.toFixed(5),r.lon.toFixed(5),r.haut,r.statut].join(",");}).join("\n"),"CSV");});
$("rp-x-geo").addEventListener("click",function(){copy(JSON.stringify({type:"FeatureCollection",features:rowsAll().map(function(r){return{type:"Feature",geometry:{type:"Point",coordinates:[+r.lon.toFixed(5),+r.lat.toFixed(5)]},properties:{id:r.id,type:r.type,priorite:r.prio,phase:r.phase,hauteur:r.haut,statut:r.statut,source:"exemple"}};})},null,1),"GeoJSON");});

/* demarrage : zone d'exemple sur la Dorsale, autour de Kasserine */
recompute();ZONE={lon:8.95,lat:35.3,R:30,nm:10,nd:1,name:"Kasserine",pots:[],sel:null};computeZone();ZONE.sel=ZONE.pots[0]||null;
renderAll();tab("zone");
new ResizeObserver(function(){resize();}).observe(wrap);resize();focusZone(true);
(function tick(){step2D();requestAnimationFrame(tick);})();
chargerOS6();
hote._rp={recharger:chargerOS6,redimensionner:resize};
})();

}

// point d'entrée : atom-runtime.js appelle ATOM.reseauAneo.afficher() à l'ouverture de la vue « reseau » (admin)
const A = window.ATOM = window.ATOM || {};
A.reseauAneo = {
  afficher(){
    const h = document.getElementById("aneo-reseau"); if(!h) return;
    // la vue n'est visible qu'après ce rappel (atom-runtime.js bascule la classe « on » ensuite) : on monte au cadre suivant
    if(!h._rp){ if(!h._montage){ h._montage = true; requestAnimationFrame(() => { monterPlanAneo(h); }); } return; }
    requestAnimationFrame(() => { h._rp.redimensionner(); h._rp.recharger(); });
  },
  recharger(){ const h = document.getElementById("aneo-reseau"); if(h && h._rp) h._rp.recharger(); },
};
