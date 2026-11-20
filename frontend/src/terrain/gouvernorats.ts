// Les 24 gouvernorats de Tunisie (chef-lieu de chacun), pour nommer le point choisi sur la carte 3D.
// Sans les contours officiels, le gouvernorat affiché est celui du CHEF-LIEU LE PLUS PROCHE : juste à l'intérieur
// d'un gouvernorat, approximatif près d'une limite entre deux gouvernorats (la page le dit).
export const GOUVERNORATS: [string, number, number][] = [
  ['Ariana', 36.8625, 10.1956], ['Béja', 36.7256, 9.1817], ['Ben Arous', 36.7531, 10.2189], ['Bizerte', 37.2744, 9.8739],
  ['Gabès', 33.8815, 10.0982], ['Gafsa', 34.425, 8.7842], ['Jendouba', 36.5011, 8.7802], ['Kairouan', 35.6781, 10.0963],
  ['Kasserine', 35.1676, 8.8365], ['Kébili', 33.7044, 8.969], ['Le Kef', 36.1742, 8.7049], ['Mahdia', 35.5047, 11.0622],
  ['Manouba', 36.8101, 10.0863], ['Médenine', 33.3549, 10.5055], ['Monastir', 35.7643, 10.8113], ['Nabeul', 36.4561, 10.7376],
  ['Sfax', 34.7406, 10.7603], ['Sidi Bouzid', 35.0382, 9.4849], ['Siliana', 36.0849, 9.3708], ['Sousse', 35.8256, 10.6084],
  ['Tataouine', 32.9297, 10.4518], ['Tozeur', 33.9197, 8.1335], ['Tunis', 36.8065, 10.1815], ['Zaghouan', 36.4029, 10.1429],
];

const km = (la1: number, lo1: number, la2: number, lo2: number) => {
  const r = Math.PI / 180, a = Math.sin((la2 - la1) * r / 2) ** 2
    + Math.cos(la1 * r) * Math.cos(la2 * r) * Math.sin((lo2 - lo1) * r / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(a));
};

/** Le gouvernorat du chef-lieu le plus proche et la distance (km) ; null au-delà de 200 km (hors de Tunisie). */
export function gouvernoratDe(lon: number, lat: number): { nom: string; distanceKm: number } | null {
  let best: { nom: string; distanceKm: number } | null = null;
  for (const [nom, la, lo] of GOUVERNORATS) {
    const d = km(lat, lon, la, lo);
    if (!best || d < best.distanceKm) best = { nom, distanceKm: d };
  }
  return best && best.distanceKm <= 200 ? best : null;
}
