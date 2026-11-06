import {geodeticToUtm,utmToGeodetic} from './wgs84.js';
import {elevationBytes} from '../elevation-codec';
export const SCENE_ORIGIN={easting:500000,northing:3800000,orthometricHeight:0};
/** heightEGM2008 is orthometric metres, NOT browser GPS ellipsoidal altitude. */
export function gpsToScene(longitude,latitude,heightEGM2008=0){
  if(!Number.isFinite(heightEGM2008))throw new TypeError('A finite EGM2008 height is required.');
  const p=geodeticToUtm(latitude,longitude,32);
  return [p.easting-500000,heightEGM2008,3800000-p.northing];
}
export function sceneToGps([x,y,z]){
  const p=utmToGeodetic(x+500000,3800000-z,32,'N');
  return {longitude:p.lon,latitude:p.lat,heightEGM2008:y};
}
export function heightAboveGround(altitudeEGM2008,terrainHeightEGM2008){return altitudeEGM2008-terrainHeightEGM2008;}
export function altitudeFromAGL(heightAGL,terrainHeightEGM2008){return terrainHeightEGM2008+heightAGL;}

/** Input is the byte stream AFTER gzip decompression. Restores signed values in-place. */
export function decodeElevationGrid(buffer,metadata){
  if(buffer.byteLength!==metadata.width*metadata.height*2)throw new Error('Elevation byte count does not match its metadata.');
  const values=new DataView(buffer);
  if(!metadata.predictor)return values;
  if(metadata.predictor!=='horizontal-delta-int16-modulo')throw new Error(`Unsupported elevation predictor: ${metadata.predictor}`);
  for(let row=0;row<metadata.height;row++){
    const start=row*metadata.width*2;let previous=values.getInt16(start,true);
    for(let col=1;col<metadata.width;col++){
      const at=start+col*2;
      previous=(previous+values.getInt16(at,true))<<16>>16;
      values.setInt16(at,previous,true);
    }
  }
  return values;
}

export class ElevationQueries{
  constructor(manifest,baseUrl){this.tiles=new Map(manifest.tiles.filter(t=>t.lod===2).map(t=>[t.id,t]));this.baseUrl=baseUrl;this.cache=new Map();}
  async sample(longitude,latitude){
    const [x,,z]=gpsToScene(longitude,latitude);const e=x+500000,n=3800000-z;
    const tx=Math.floor(e/8192),ty=Math.floor((n-3276800)/8192),t=this.tiles.get(`7-${tx}-${ty}`);
    if(!t)return null;
    let values=this.cache.get(t.id);
    if(!values){
      const r=await fetch(new URL(t.elevation.uri,this.baseUrl));if(!r.ok)throw new Error(`Elevation HTTP ${r.status}`);
      const buffer=await elevationBytes(r);
      values=decodeElevationGrid(buffer,t.elevation);this.cache.set(t.id,values);if(this.cache.size>32)this.cache.delete(this.cache.keys().next().value);
    }
    const [w,s,ee,nn]=t.projectedBounds,meta=t.elevation;
    const c=(e-w)/32,r=(nn-n)/32,ci=Math.min(meta.width-2,Math.max(0,Math.floor(c))),ri=Math.min(meta.height-2,Math.max(0,Math.floor(r)));
    const get=(row,col)=>values.getInt16((row*meta.width+col)*2,true);
    const a=get(ri,ci),b=get(ri,ci+1),d=get(ri+1,ci),f=get(ri+1,ci+1);
    if([a,b,d,f].includes(meta.noData))return null;
    const u=c-ci,v=r-ri;return ((a*(1-u)+b*u)*(1-v)+(d*(1-u)+f*u)*v)*meta.scale;
  }
}
