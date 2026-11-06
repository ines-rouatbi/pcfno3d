export type NationalSnapshot = {
  version:number;source:string;sourceUrl:string;attribution:string;retrievedAt:number;timezone:string;
  grid:{west:number;south:number;step:number;nx:number;ny:number};heights:number[];times:number[];
  points:Array<{latitude:number;longitude:number;modelLatitude:number;modelLongitude:number;[key:string]:number|Array<number|null>}>;
};
const CACHE='atom-national-gfs-v1',HOUR=3600000;
const VARIABLES=['wind_speed_10m','wind_direction_10m','wind_speed_80m','wind_direction_80m'];
const COLORS:[[number,number[]],...[number,number[]][]]=[[0,[191,227,242]],[8,[89,195,160]],[16,[182,216,74]],[24,[244,196,48]],[34,[240,138,44]],[46,[224,65,75]],[62,[167,58,158]]];
export function windColor(speed:number){
  const value=speed*3.6;
  for(let i=1;i<COLORS.length;i++)if(value<=COLORS[i][0]){const [a,ca]=COLORS[i-1],[b,cb]=COLORS[i],t=(value-a)/(b-a);return ca.map((x,j)=>Math.round(x+(cb[j]-x)*t));}
  return COLORS.at(-1)![1];
}
function validate(value:any):NationalSnapshot {
  if(value?.version!==1||value.grid?.west!==7.5||value.grid?.south!==30||value.grid?.nx!==10||value.grid?.ny!==17||value.grid?.step!==.5||value.timezone!=='Africa/Tunis'||JSON.stringify(value.heights)!=='[10,80]'||value.points?.length!==170||!Array.isArray(value.times)||value.times.length<24||!Number.isFinite(value.retrievedAt))throw new Error('Invalid national forecast snapshot');
  for(let i=0;i<value.times.length;i++)if(!Number.isFinite(value.times[i])||(i&&value.times[i]-value.times[i-1]!==3600))throw new Error('Invalid forecast timestamps');
  value.points.forEach((point:any,index:number)=>{
    if(point.latitude!==30+Math.floor(index/10)*.5||point.longitude!==7.5+(index%10)*.5)throw new Error('Invalid forecast grid ordering');
    for(const variable of VARIABLES)if(!Array.isArray(point[variable])||point[variable].length!==value.times.length||point[variable].some((v:any)=>v!==null&&(!Number.isFinite(v)||(variable.includes('speed')&&v<0))))throw new Error('Invalid forecast values');
  });
  return value;
}
function usable(snapshot:NationalSnapshot,age:number){return Date.now()-snapshot.retrievedAt<=age&&snapshot.retrievedAt<=Date.now()+300000&&snapshot.times[0]*1000<=Date.now()&&snapshot.times.at(-1)!*1000>=Date.now();}
export class NationalForecast {
  constructor(public data:NationalSnapshot,public stale=false){}
  get nearestHour(){return this.data.times.reduce((best,t,i)=>Math.abs(t*1000-Date.now())<Math.abs(this.data.times[best]*1000-Date.now())?i:best,0);}
  sample(lon:number,lat:number,height:number,index:number){
    if(!this.data.heights.includes(height)||index<0||index>=this.data.times.length)return null;
    const g=this.data.grid,x=(lon-g.west)/g.step,y=(lat-g.south)/g.step;
    if(x<0||y<0||x>g.nx-1||y>g.ny-1)return null;
    const i=Math.min(g.nx-2,Math.floor(x)),j=Math.min(g.ny-2,Math.floor(y)),dx=x-i,dy=y-j;
    const corners=[[i,j,(1-dx)*(1-dy)],[i+1,j,dx*(1-dy)],[i,j+1,(1-dx)*dy],[i+1,j+1,dx*dy]];
    let u=0,v=0;
    for(const [cx,cy,weight] of corners){
      const point=this.data.points[cy*g.nx+cx],speed=(point[`wind_speed_${height}m`] as Array<number|null>)[index],from=(point[`wind_direction_${height}m`] as Array<number|null>)[index];
      if(speed===null||from===null)return null;
      u-=speed*Math.sin(from*Math.PI/180)*weight;v-=speed*Math.cos(from*Math.PI/180)*weight;
    }
    const speed=Math.hypot(u,v),bearing=(Math.atan2(-u,-v)*180/Math.PI+360)%360;
    const directions=['du nord','du nord-est','de l’est','du sud-est','du sud','du sud-ouest','de l’ouest','du nord-ouest'];
    return {u,v,speed,color:windColor(speed),formatted:`${Math.round(speed*3.6)} km/h`,direction:directions[Math.round(bearing/45)%8]};
  }
}
async function directForecast(signal:AbortSignal):Promise<NationalSnapshot>{
  const points:NationalSnapshot['points']=[];let times:number[]=[];
  for(let offset=0;offset<170;offset+=32){
    const coordinates=Array.from({length:Math.min(32,170-offset)},(_,n)=>({latitude:30+Math.floor((offset+n)/10)*.5,longitude:7.5+(offset+n)%10*.5}));
    const params=new URLSearchParams({latitude:coordinates.map(p=>p.latitude).join(','),longitude:coordinates.map(p=>p.longitude).join(','),hourly:VARIABLES.join(','),wind_speed_unit:'ms',timeformat:'unixtime',timezone:'Africa/Tunis',forecast_days:'2',cell_selection:'nearest'});
    const response=await fetch(`https://api.open-meteo.com/v1/gfs?${params}`,{signal:AbortSignal.any([signal,AbortSignal.timeout(25000)])});
    if(!response.ok)throw new Error(`Forecast HTTP ${response.status}`);
    const records=await response.json();if(!Array.isArray(records)||records.length!==coordinates.length)throw new Error('Invalid forecast locations');
    records.forEach((record:any,i:number)=>{
      if(record.hourly_units?.wind_speed_80m!=='m/s'||(times.length&&JSON.stringify(record.hourly.time)!==JSON.stringify(times)))throw new Error('Inconsistent forecast times or units');
      times=record.hourly.time;
      points.push({...coordinates[i],modelLatitude:record.latitude,modelLongitude:record.longitude,...Object.fromEntries(VARIABLES.map(key=>[key,record.hourly[key]]))});
    });
  }
  return validate({version:1,source:'NOAA GFS via Open-Meteo',sourceUrl:'https://open-meteo.com/en/docs/gfs-api',attribution:'Weather data by Open-Meteo.com, NOAA GFS; CC BY 4.0',retrievedAt:Date.now(),timezone:'Africa/Tunis',grid:{west:7.5,south:30,step:.5,nx:10,ny:17},heights:[10,80],times,points});
}
export async function loadNationalForecast(signal:AbortSignal){
  let cached:NationalSnapshot|undefined;
  try{cached=validate(JSON.parse(localStorage.getItem(CACHE)||'null'));if(usable(cached,HOUR))return new NationalForecast(cached);}catch{}
  try{
    const response=await fetch(`${import.meta.env.BASE_URL}weather/tunisia-gfs.json`,{signal,cache:'no-cache'});
    const snapshot=validate(await response.json());if(!cached||snapshot.retrievedAt>cached.retrievedAt)cached=snapshot;
    if(usable(snapshot,HOUR)){try{localStorage.setItem(CACHE,JSON.stringify(snapshot));}catch{}return new NationalForecast(snapshot);}
  }catch(error){if(signal.aborted)throw error;}
  try{const fresh=await directForecast(signal);try{localStorage.setItem(CACHE,JSON.stringify(fresh));}catch{}return new NationalForecast(fresh);}
  catch(error){if(!signal.aborted&&cached&&usable(cached,6*HOUR))return new NationalForecast(cached,true);throw error;}
}
