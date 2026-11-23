// Les 24 gouvernorats de Tunisie : chef-lieu de chacun, et leurs CONTOURS officiels (geoBoundaries, d'après
// OpenStreetMap, ODbL ; fichier public/atom-wind/gouvernorats-tn.json, ~160 Ko) pour nommer un point exactement.
// Tant que les contours ne sont pas chargés, le gouvernorat est celui du chef-lieu le plus proche (la page le dit).
export const GOUVERNORATS: [string, number, number][] = [
  ['Ariana', 36.8625, 10.1956], ['Béja', 36.7256, 9.1817], ['Ben Arous', 36.7531, 10.2189], ['Bizerte', 37.2744, 9.8739],
  ['Gabès', 33.8815, 10.0982], ['Gafsa', 34.425, 8.7842], ['Jendouba', 36.5011, 8.7802], ['Kairouan', 35.6781, 10.0963],
  ['Kasserine', 35.1676, 8.8365], ['Kébili', 33.7044, 8.969], ['Le Kef', 36.1742, 8.7049], ['Mahdia', 35.5047, 11.0622],
  ['Manouba', 36.8101, 10.0863], ['Médenine', 33.3549, 10.5055], ['Monastir', 35.7643, 10.8113], ['Nabeul', 36.4561, 10.7376],
  ['Sfax', 34.7406, 10.7603], ['Sidi Bouzid', 35.0382, 9.4849], ['Siliana', 36.0849, 9.3708], ['Sousse', 35.8256, 10.6084],
  ['Tataouine', 32.9297, 10.4518], ['Tozeur', 33.9197, 8.1335], ['Tunis', 36.8065, 10.1815], ['Zaghouan', 36.4029, 10.1429],
];

export type Gouvernorat = { nom: string; ar: string; chef_lieu: [number, number]; contours: [number, number][][] };
export type Contours = { source: string; licence: string; url: string; gouvernorats: Gouvernorat[] };

let contours: Contours | null = null;
let chargement: Promise<Contours | null> | null = null;
/** Les contours des 24 gouvernorats (chargés une fois) ; null si le fichier manque. */
export function chargerContours(): Promise<Contours | null> {
  return chargement ??= fetch(`${import.meta.env.BASE_URL}atom-wind/gouvernorats-tn.json`)
    .then(r => (r.ok ? r.json() : null)).then((d: Contours | null) => (contours = d && d.gouvernorats?.length === 24 ? d : null))
    .catch(() => null);
}

function dansAnneau(lon: number, lat: number, r: [number, number][]) {
  let dedans = false;
  for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
    const [xi, yi] = r[i], [xj, yj] = r[j];
    if ((yi > lat) !== (yj > lat) && lon < (xj - xi) * (lat - yi) / (yj - yi) + xi) dedans = !dedans;
  }
  return dedans;
}

const km = (la1: number, lo1: number, la2: number, lo2: number) => {
  const r = Math.PI / 180, a = Math.sin((la2 - la1) * r / 2) ** 2
    + Math.cos(la1 * r) * Math.cos(la2 * r) * Math.sin((lo2 - lo1) * r / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(a));
};

/** Le gouvernorat d'un point : par ses contours (`exact`) s'ils sont chargés, sinon le chef-lieu le plus proche ;
 *  null hors de Tunisie (en mer ou à l'étranger avec les contours ; au-delà de 200 km sans eux). */
export function gouvernoratDe(lon: number, lat: number): { nom: string; distanceKm: number; exact: boolean } | null {
  if (contours) {
    const g = contours.gouvernorats.find(x => x.contours.some(r => dansAnneau(lon, lat, r)));
    return g ? { nom: g.nom, distanceKm: km(lat, lon, g.chef_lieu[0], g.chef_lieu[1]), exact: true } : null;
  }
  let best: { nom: string; distanceKm: number; exact: boolean } | null = null;
  for (const [nom, la, lo] of GOUVERNORATS) {
    const d = km(lat, lon, la, lo);
    if (!best || d < best.distanceKm) best = { nom, distanceKm: d, exact: false };
  }
  return best && best.distanceKm <= 200 ? best : null;
}
void chargerContours();
