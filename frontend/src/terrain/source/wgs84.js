// Geodetic / projected coordinate math for the geospatial importer (docs/SPEC-import.md § geo).
//
// Pure module — no three.js, no DOM — so it runs in Node tests and in the DEM worker unchanged.
//
// Tukai world convention (CONVENTION-unites.md, SPEC-import.md): 1 world unit = 1 metre.
// Scene axes are three.js Y-up. The mapping from a local East-North-Up tangent plane is:
//
//     world X = East   (metres)
//     world Y = Up      (metres, elevation above the reference height)
//     world Z = -North  (metres)   ← north points towards -Z, the usual "into the screen" forward
//
// Everything geographic is projected to a LOCAL tangent plane anchored at a chosen reference
// (`geo_set_reference`). A tangent plane is exact at the anchor and good to well under a metre over
// tens of kilometres; for a whole country we also record the distortion so the assumption is explicit
// rather than hidden (a flat projection cannot preserve every distance — SPEC-import.md § precision).

// ---- WGS84 (EPSG:4326) ellipsoid --------------------------------------------------------------------

export const WGS84 = Object.freeze( {
	a: 6378137.0,                 // semi-major axis (m)
	f: 1 / 298.257223563,         // flattening
	get b() { return this.a * ( 1 - this.f ); },
	get e2() { return this.f * ( 2 - this.f ); },        // first eccentricity squared
	get ep2() { return this.e2 / ( 1 - this.e2 ); },      // second eccentricity squared
} );

const DEG = Math.PI / 180;

function toRad( deg ) { return deg * DEG; }
function toDeg( rad ) { return rad / DEG; }

/** Finite number in [min, max]. Throws a readable error otherwise — bad coordinates must not pass silently. */
export function assertLatLon( lat, lon ) {

	if ( ! Number.isFinite( lat ) || lat < -90 || lat > 90 ) throw new Error( `latitude invalide : ${lat} (attendu -90…90)` );
	if ( ! Number.isFinite( lon ) || lon < -180 || lon > 180 ) throw new Error( `longitude invalide : ${lon} (attendu -180…180)` );

}

// ---- geodetic ↔ ECEF (Earth-Centred Earth-Fixed) ----------------------------------------------------

/** Geodetic (deg, deg, m) → ECEF metres {x, y, z}. */
export function geodeticToEcef( lat, lon, h = 0 ) {

	assertLatLon( lat, lon );
	const { a, e2 } = WGS84;
	const phi = toRad( lat );
	const lam = toRad( lon );
	const sinPhi = Math.sin( phi );
	const cosPhi = Math.cos( phi );
	const N = a / Math.sqrt( 1 - e2 * sinPhi * sinPhi );   // prime vertical radius of curvature

	return {
		x: ( N + h ) * cosPhi * Math.cos( lam ),
		y: ( N + h ) * cosPhi * Math.sin( lam ),
		z: ( N * ( 1 - e2 ) + h ) * sinPhi,
	};

}

/** ECEF metres → geodetic {lat, lon, h} (deg, deg, m). Bowring's closed-form approximation (mm-level). */
export function ecefToGeodetic( x, y, z ) {

	const { a, b, e2, ep2 } = WGS84;
	const lon = Math.atan2( y, x );
	const p = Math.hypot( x, y );
	const theta = Math.atan2( z * a, p * b );
	const sinT = Math.sin( theta );
	const cosT = Math.cos( theta );
	const lat = Math.atan2( z + ep2 * b * sinT * sinT * sinT, p - e2 * a * cosT * cosT * cosT );
	const sinLat = Math.sin( lat );
	const N = a / Math.sqrt( 1 - e2 * sinLat * sinLat );
	const h = p / Math.cos( lat ) - N;

	return { lat: toDeg( lat ), lon: toDeg( lon ), h };

}

// ---- geodetic ↔ local East-North-Up tangent plane ---------------------------------------------------

/**
 * Geodetic point → local ENU metres {e, n, u} relative to a reference (anchor) geodetic point.
 * Exact at the anchor; this is the projection used to place terrain and footprints in the scene.
 */
export function geodeticToEnu( lat, lon, h, refLat, refLon, refH = 0 ) {

	const p = geodeticToEcef( lat, lon, h );
	const r = geodeticToEcef( refLat, refLon, refH );
	return ecefVectorToEnu( p.x - r.x, p.y - r.y, p.z - r.z, refLat, refLon );

}

/** ENU metres → geodetic {lat, lon, h}. Inverse of geodeticToEnu. */
export function enuToGeodetic( e, n, u, refLat, refLon, refH = 0 ) {

	const phi = toRad( refLat );
	const lam = toRad( refLon );
	const sinPhi = Math.sin( phi ), cosPhi = Math.cos( phi );
	const sinLam = Math.sin( lam ), cosLam = Math.cos( lam );

	// Rotate ENU back into ECEF deltas (transpose of the ENU rotation matrix).
	const dx = -sinLam * e - sinPhi * cosLam * n + cosPhi * cosLam * u;
	const dy = cosLam * e - sinPhi * sinLam * n + cosPhi * sinLam * u;
	const dz = cosPhi * n + sinPhi * u;

	const r = geodeticToEcef( refLat, refLon, refH );
	return ecefToGeodetic( r.x + dx, r.y + dy, r.z + dz );

}

function ecefVectorToEnu( dx, dy, dz, refLat, refLon ) {

	const phi = toRad( refLat );
	const lam = toRad( refLon );
	const sinPhi = Math.sin( phi ), cosPhi = Math.cos( phi );
	const sinLam = Math.sin( lam ), cosLam = Math.cos( lam );

	return {
		e: -sinLam * dx + cosLam * dy,
		n: -sinPhi * cosLam * dx - sinPhi * sinLam * dy + cosPhi * dz,
		u: cosPhi * cosLam * dx + cosPhi * sinLam * dy + sinPhi * dz,
	};

}

/** Local ENU (metres) → Tukai world position {x, y, z} (East, Up, -North). */
export function enuToWorld( e, n, u ) {

	return { x: e, y: u, z: -n };

}

// ---- geodesic distance (reference for tests & distortion reporting) ---------------------------------

/** Great-circle distance on the WGS84 mean sphere (m). Fast; ~0.3 % vs the ellipsoid. */
export function haversine( lat1, lon1, lat2, lon2 ) {

	const R = 6371008.8;   // WGS84 mean radius
	const dPhi = toRad( lat2 - lat1 );
	const dLam = toRad( lon2 - lon1 );
	const s = Math.sin( dPhi / 2 ) ** 2 + Math.cos( toRad( lat1 ) ) * Math.cos( toRad( lat2 ) ) * Math.sin( dLam / 2 ) ** 2;
	return 2 * R * Math.asin( Math.min( 1, Math.sqrt( s ) ) );

}

/** Ellipsoidal geodesic distance (m), Vincenty inverse. Returns the reference used to measure projection distortion. */
export function geodesicDistance( lat1, lon1, lat2, lon2 ) {

	const { a, b, f } = WGS84;
	const L = toRad( lon2 - lon1 );
	const U1 = Math.atan( ( 1 - f ) * Math.tan( toRad( lat1 ) ) );
	const U2 = Math.atan( ( 1 - f ) * Math.tan( toRad( lat2 ) ) );
	const sinU1 = Math.sin( U1 ), cosU1 = Math.cos( U1 );
	const sinU2 = Math.sin( U2 ), cosU2 = Math.cos( U2 );

	let lambda = L, lambdaP, iter = 0;
	let cosSqAlpha, sinSigma, cos2SigmaM, cosSigma, sigma;

	do {

		const sinLambda = Math.sin( lambda ), cosLambda = Math.cos( lambda );
		sinSigma = Math.sqrt( ( cosU2 * sinLambda ) ** 2 + ( cosU1 * sinU2 - sinU1 * cosU2 * cosLambda ) ** 2 );
		if ( sinSigma === 0 ) return 0;   // coincident points
		cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda;
		sigma = Math.atan2( sinSigma, cosSigma );
		const sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma;
		cosSqAlpha = 1 - sinAlpha * sinAlpha;
		cos2SigmaM = cosSqAlpha !== 0 ? cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha : 0;
		const C = f / 16 * cosSqAlpha * ( 4 + f * ( 4 - 3 * cosSqAlpha ) );
		lambdaP = lambda;
		lambda = L + ( 1 - C ) * f * sinAlpha * ( sigma + C * sinSigma * ( cos2SigmaM + C * cosSigma * ( -1 + 2 * cos2SigmaM ** 2 ) ) );

	} while ( Math.abs( lambda - lambdaP ) > 1e-12 && ++iter < 200 );

	const uSq = cosSqAlpha * ( a * a - b * b ) / ( b * b );
	const A = 1 + uSq / 16384 * ( 4096 + uSq * ( -768 + uSq * ( 320 - 175 * uSq ) ) );
	const B = uSq / 1024 * ( 256 + uSq * ( -128 + uSq * ( 74 - 47 * uSq ) ) );
	const deltaSigma = B * sinSigma * ( cos2SigmaM + B / 4 * ( cosSigma * ( -1 + 2 * cos2SigmaM ** 2 )
		- B / 6 * cos2SigmaM * ( -3 + 4 * sinSigma ** 2 ) * ( -3 + 4 * cos2SigmaM ** 2 ) ) );

	return b * A * ( sigma - deltaSigma );

}

// ---- UTM (transverse Mercator, Karney / Snyder series) ----------------------------------------------

/** UTM zone number (1…60) for a longitude. */
export function utmZone( lon ) {

	return Math.floor( ( ( ( lon + 180 ) % 360 + 360 ) % 360 ) / 6 ) + 1;

}

/** EPSG code of the WGS84 UTM CRS for a lat/lon (326xx north, 327xx south). */
export function utmEpsg( lat, lon ) {

	const zone = utmZone( lon );
	return ( lat >= 0 ? 32600 : 32700 ) + zone;

}

/**
 * Geodetic → UTM {zone, hemisphere, easting, northing, epsg}. Snyder series, sub-millimetre inside a zone.
 * A projected CRS, in metres — the usual target when a user wants flat Cartesian coordinates for a region.
 */
export function geodeticToUtm( lat, lon, zone = utmZone( lon ) ) {

	assertLatLon( lat, lon );
	const { a, f } = WGS84;
	const k0 = 0.9996;
	const e2 = f * ( 2 - f );
	const ep2 = e2 / ( 1 - e2 );

	const lon0 = toRad( ( zone - 1 ) * 6 - 180 + 3 );   // central meridian
	const phi = toRad( lat );
	const lam = toRad( lon );

	const N = a / Math.sqrt( 1 - e2 * Math.sin( phi ) ** 2 );
	const T = Math.tan( phi ) ** 2;
	const C = ep2 * Math.cos( phi ) ** 2;
	const A = Math.cos( phi ) * ( lam - lon0 );

	const M = a * ( ( 1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256 ) * phi
		- ( 3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024 ) * Math.sin( 2 * phi )
		+ ( 15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024 ) * Math.sin( 4 * phi )
		- ( 35 * e2 ** 3 / 3072 ) * Math.sin( 6 * phi ) );

	const easting = k0 * N * ( A + ( 1 - T + C ) * A ** 3 / 6
		+ ( 5 - 18 * T + T ** 2 + 72 * C - 58 * ep2 ) * A ** 5 / 120 ) + 500000;

	let northing = k0 * ( M + N * Math.tan( phi ) * ( A ** 2 / 2 + ( 5 - T + 9 * C + 4 * C ** 2 ) * A ** 4 / 24
		+ ( 61 - 58 * T + T ** 2 + 600 * C - 330 * ep2 ) * A ** 6 / 720 ) );

	if ( lat < 0 ) northing += 10000000;   // false northing in the southern hemisphere

	return { zone, hemisphere: lat >= 0 ? 'N' : 'S', easting, northing, epsg: utmEpsg( lat, lon ) };

}

/** UTM {easting, northing, zone, hemisphere} → geodetic {lat, lon}. Inverse of geodeticToUtm. */
export function utmToGeodetic( easting, northing, zone, hemisphere = 'N' ) {

	const { a } = WGS84;
	const k0 = 0.9996;
	const e2 = WGS84.e2;
	const ep2 = e2 / ( 1 - e2 );
	const e1 = ( 1 - Math.sqrt( 1 - e2 ) ) / ( 1 + Math.sqrt( 1 - e2 ) );

	const x = easting - 500000;
	let y = northing;
	if ( String( hemisphere ).toUpperCase() === 'S' ) y -= 10000000;

	const M = y / k0;
	const mu = M / ( a * ( 1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256 ) );
	const phi1 = mu + ( 3 * e1 / 2 - 27 * e1 ** 3 / 32 ) * Math.sin( 2 * mu )
		+ ( 21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32 ) * Math.sin( 4 * mu )
		+ ( 151 * e1 ** 3 / 96 ) * Math.sin( 6 * mu );

	const sinP = Math.sin( phi1 ), cosP = Math.cos( phi1 ), tanP = Math.tan( phi1 );
	const N1 = a / Math.sqrt( 1 - e2 * sinP ** 2 );
	const T1 = tanP ** 2;
	const C1 = ep2 * cosP ** 2;
	const R1 = a * ( 1 - e2 ) / ( 1 - e2 * sinP ** 2 ) ** 1.5;
	const D = x / ( N1 * k0 );

	const lat = phi1 - ( N1 * tanP / R1 ) * ( D ** 2 / 2
		- ( 5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * ep2 ) * D ** 4 / 24
		+ ( 61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * ep2 - 3 * C1 ** 2 ) * D ** 6 / 720 );

	const lon0 = toRad( ( zone - 1 ) * 6 - 180 + 3 );
	const lon = lon0 + ( D - ( 1 + 2 * T1 + C1 ) * D ** 3 / 6
		+ ( 5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * ep2 + 24 * T1 ** 2 ) * D ** 5 / 120 ) / cosP;

	return { lat: toDeg( lat ), lon: toDeg( lon ) };

}
