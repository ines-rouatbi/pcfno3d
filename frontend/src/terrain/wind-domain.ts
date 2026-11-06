export type WindDomain={id:'local'|'national';west:number;south:number;east:number;north:number;width:number;height:number};
export const LOCAL_DOMAIN:WindDomain={id:'local',west:10.04,south:36.395,east:10.04+68/89.42,north:36.395+56/110.97,width:68,height:56};
export const NATIONAL_DOMAIN:WindDomain={id:'national',west:7.5,south:30,east:12,north:38,width:4.5*111.32*Math.cos(34*Math.PI/180),height:8*110.97};
export const toLongitude=(x:number,domain:WindDomain)=>domain.west+x/domain.width*(domain.east-domain.west);
export const toLatitude=(y:number,domain:WindDomain)=>domain.south+y/domain.height*(domain.north-domain.south);
