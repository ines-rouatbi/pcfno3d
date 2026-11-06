import * as THREE from 'three';
import { decodeElevationGrid, gpsToScene } from './source/geo.js';
import { elevationBytes } from './elevation-codec';
import { NationalSurface } from './national-surface';
import { LOCAL_DOMAIN,toLongitude,toLatitude,type WindDomain } from './wind-domain';
import type { loadTerrain, Tile } from './terrain-stream';

type Terrain = Awaited<ReturnType<typeof loadTerrain>>;
type Wind = {u:number;v:number;speed:number;color:number[]};
type Grid = { tile:Tile; values:DataView };
type WaterTriangle = number[];
const BASE = `${import.meta.env.BASE_URL}terrain/tunisia/`;
const longitude = (x:number)=>10.04+x/89.42;
const latitude = (y:number)=>36.395+y/110.97;

/** A bounded, synchronous height atlas for the wind domain, using the supplied DEM codec. */
class WindSurface {
  grids = new Map<string,Grid>();
  water = new Map<string,WaterTriangle[]>();
  waterSamples = 0;
  async load(terrain:Terrain,signal:AbortSignal) {
    const corners=[[0,0],[68,0],[0,56],[68,56]].map(([x,y])=>gpsToScene(longitude(x),latitude(y)));
    const west=Math.min(...corners.map(p=>p[0]))+500000,east=Math.max(...corners.map(p=>p[0]))+500000;
    const south=3800000-Math.max(...corners.map(p=>p[2])),north=3800000-Math.min(...corners.map(p=>p[2]));
    const tiles=[...terrain.tiles.values()].filter(tile=>{const [w,s,e,n]=tile.projectedBounds;return tile.lod===2&&e>west&&w<east&&n>south&&s<north;});
    for(let offset=0;offset<tiles.length;offset+=4){
      signal.throwIfAborted();
      await Promise.all(tiles.slice(offset,offset+4).map(async tile=>{
        const response=await fetch(new URL(tile.elevation.uri,new URL(BASE,location.href)),{signal});
        if(!response.ok)throw new Error(`DEM ${response.status}`);
        const decoded=await elevationBytes(response);
        this.grids.set(tile.id,{tile,values:decodeElevationGrid(decoded,tile.elevation)});
      }));
    }
    // Index the actual water triangles, rather than treating missing DEM as sea.
    terrain.root.updateMatrixWorld(true);
    const a=new THREE.Vector3(),b=new THREE.Vector3(),c=new THREE.Vector3();
    terrain.water.traverse(object=>{
      if(!(object instanceof THREE.Mesh))return;
      const position=object.geometry.attributes.position,index=object.geometry.index,count=index?index.count:position.count;
      for(let offset=0;offset<count;offset+=3){
        a.fromBufferAttribute(position,index?index.getX(offset):offset).applyMatrix4(object.matrixWorld);
        b.fromBufferAttribute(position,index?index.getX(offset+1):offset+1).applyMatrix4(object.matrixWorld);
        c.fromBufferAttribute(position,index?index.getX(offset+2):offset+2).applyMatrix4(object.matrixWorld);
        const minX=Math.max(west-500000,Math.min(a.x,b.x,c.x)),maxX=Math.min(east-500000,Math.max(a.x,b.x,c.x));
        const minZ=Math.max(3800000-north,Math.min(a.z,b.z,c.z)),maxZ=Math.min(3800000-south,Math.max(a.z,b.z,c.z));
        if(minX>maxX||minZ>maxZ)continue;
        const triangle=[a.x,a.z,a.y,b.x,b.z,b.y,c.x,c.z,c.y];
        for(let x=Math.floor(minX/8192);x<=Math.floor(maxX/8192);x++)for(let z=Math.floor(minZ/8192);z<=Math.floor(maxZ/8192);z++){
          const key=`${x}/${z}`,bucket=this.water.get(key)||[];bucket.push(triangle);this.water.set(key,bucket);
        }
      }
    });
  }
  sample(lon:number,lat:number):THREE.Vector3|null {
    const p=gpsToScene(lon,lat),px=p[0],pz=p[2];
    for(const t of this.water.get(`${Math.floor(px/8192)}/${Math.floor(pz/8192)}`)||[]){
      const denominator=(t[4]-t[7])*(t[0]-t[6])+(t[6]-t[3])*(t[1]-t[7]);
      if(Math.abs(denominator)<1e-8)continue;
      const u=((t[4]-t[7])*(px-t[6])+(t[6]-t[3])*(pz-t[7]))/denominator;
      const v=((t[7]-t[1])*(px-t[6])+(t[0]-t[6])*(pz-t[7]))/denominator,w=1-u-v;
      if(u>=-1e-7&&v>=-1e-7&&w>=-1e-7){this.waterSamples++;return new THREE.Vector3(px,u*t[2]+v*t[5]+w*t[8],pz);}
    }
    const e=px+500000,n=3800000-pz,grid=this.grids.get(`7-${Math.floor(e/8192)}-${Math.floor((n-3276800)/8192)}`);
    if(!grid)return null;
    const meta=grid.tile.elevation,[west,,,north]=grid.tile.projectedBounds;
    const column=(e-west)/32,row=(north-n)/32,i=Math.min(meta.width-2,Math.max(0,Math.floor(column))),j=Math.min(meta.height-2,Math.max(0,Math.floor(row)));
    const get=(r:number,c:number)=>grid.values.getInt16((r*meta.width+c)*2,true);
    const a=get(j,i),b=get(j,i+1),c=get(j+1,i),d=get(j+1,i+1);
    if([a,b,c,d].includes(meta.noData))return null;
    const u=column-i,v=row-j,h=((a*(1-u)+b*u)*(1-v)+(c*(1-u)+d*u)*v)*meta.scale;
    return new THREE.Vector3(px,h,pz);
  }
}

const vertexShader=`
  attribute vec3 aNext;
  attribute float aForward;
  attribute float aSide;
  attribute float aAlong;
  attribute float aSeed;
  attribute float aRate;
  attribute vec3 aColor;
  uniform vec2 uResolution;
  uniform float uWidth;
  uniform float uTime;
  varying float vAlong;
  varying float vSeed;
  varying float vRate;
  varying float vSide;
  varying vec3 vColor;
  void main(){
    vec4 p=projectionMatrix*modelViewMatrix*vec4(position,1.0);
    vec4 q=projectionMatrix*modelViewMatrix*vec4(aNext,1.0);
    if(p.w<=0.1||q.w<=0.1){gl_Position=vec4(2.0,2.0,2.0,1.0);return;}
    vec2 direction=(q.xy/q.w-p.xy/p.w)*uResolution*aForward;
    direction=normalize(direction+vec2(0.000001));
    vec2 normal=vec2(-direction.y,direction.x);
    float age=fract(aSeed+uTime*aRate-aAlong);
    float taper=0.45+0.55*max(0.0,1.0-age/0.23);
    p.xy+=normal*aSide*uWidth/uResolution*p.w*taper;
    gl_Position=p;vAlong=aAlong;vSeed=aSeed;vRate=aRate;vSide=aSide;vColor=aColor;
  }
`;
const fragmentShader=`
  uniform float uTime;
  uniform float uShadow;
  varying float vAlong;
  varying float vSeed;
  varying float vRate;
  varying float vSide;
  varying vec3 vColor;
  void main(){
    float age=fract(vSeed+uTime*vRate-vAlong);
    if(age>0.23)discard;
    float tail=pow(max(0.0,1.0-age/0.23),1.25);
    float edge=1.0-smoothstep(0.5,1.0,abs(vSide));
    vec3 color=mix(vColor,vec3(1.0),0.35+0.5*tail);
    color=mix(color,vec3(0.015,0.065,0.085),uShadow);
    gl_FragColor=vec4(color,tail*edge*mix(0.96,0.32,uShadow));
    #include <colorspace_fragment>
  }
`;

/** Streamlines follow the sampled field; GPU phase animation is deliberately accelerated. */
export class WindFlow {
  root=new THREE.Group();
  surface=new WindSurface();
  private localSurface=this.surface;
  private nationalSurface=new NationalSurface();
  private prepared=new Set<string>();
  private preparing=new Map<string,Promise<void>>();
  domain:WindDomain=LOCAL_DOMAIN;
  ready=false;
  time=0;
  paths=0;
  samples=0;
  builds=0;
  segments=0;
  private geometry=new THREE.BufferGeometry();
  private resolution=new THREE.Vector2(1,1);
  private uniforms={uTime:{value:0},uResolution:{value:this.resolution},uWidth:{value:2.2},uShadow:{value:0}};
  private core=new THREE.ShaderMaterial({vertexShader,fragmentShader,uniforms:this.uniforms,transparent:true,depthWrite:false,depthTest:false,toneMapped:false});
  private shadowUniforms={...this.uniforms,uWidth:{value:3.6},uShadow:{value:1}};
  private shadow=new THREE.ShaderMaterial({vertexShader,fragmentShader,uniforms:this.shadowUniforms,transparent:true,depthWrite:false,depthTest:false,toneMapped:false});
  constructor(){
    this.root.name='Animated wind on Tunisia terrain';
    const halo=new THREE.Mesh(this.geometry,this.shadow),traces=new THREE.Mesh(this.geometry,this.core);
    halo.renderOrder=10;traces.renderOrder=11;halo.frustumCulled=false;traces.frustumCulled=false;this.root.add(halo,traces);
  }
  async prepare(terrain:Terrain,signal:AbortSignal,domain:WindDomain=LOCAL_DOMAIN){
    if(this.prepared.has(domain.id))return;
    let task=this.preparing.get(domain.id);
    if(!task){task=(domain.id==='national'?this.nationalSurface:this.localSurface).load(terrain,signal).then(()=>{this.prepared.add(domain.id);}).finally(()=>this.preparing.delete(domain.id));this.preparing.set(domain.id,task);}
    await task;
  }
  activate(domain:WindDomain){this.domain=domain;this.ready=this.prepared.has(domain.id);}
  resize(width:number,height:number,dpr:number){this.resolution.set(width,height);this.uniforms.uWidth.value=2.2*dpr;this.shadowUniforms.uWidth.value=3.6*dpr;}
  tick(seconds:number){this.time=seconds;this.uniforms.uTime.value=seconds;}
  rebuild(height:number,mobile:boolean,sample:(lon:number,lat:number,height:number)=>Wind|null,view?:{longitude:number;latitude:number;distanceKm:number;aspect:number}){
    if(!this.ready)return;
    const positions:number[]=[],next:number[]=[],forward:number[]=[],sides:number[]=[],along:number[]=[],seeds:number[]=[],rates:number[]=[],colors:number[]=[],indices:number[]=[];
    let seed=92731;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
    this.paths=0;this.samples=0;this.surface.waterSamples=0;
    const domain=this.domain,national=domain.id==='national',surface=national?this.nationalSurface:this.localSurface;
    const count=national?(mobile?430:760):(mobile?270:460),step=national?THREE.MathUtils.clamp((view?.distanceKm??1000)/200,.3,5):.38,steps=40;
    let minX=.2,maxX=domain.width-.2,minY=.2,maxY=domain.height-.2;
    if(national&&view&&view.distanceKm<500){
      const centerX=(view.longitude-domain.west)/(domain.east-domain.west)*domain.width,centerY=(view.latitude-domain.south)/(domain.north-domain.south)*domain.height;
      const radius=view.distanceKm*.65*Math.max(1,view.aspect);
      minX=Math.max(minX,centerX-radius);maxX=Math.min(maxX,centerX+radius);minY=Math.max(minY,centerY-radius);maxY=Math.min(maxY,centerY+radius);
    }
    if(minX>=maxX||minY>=maxY)return;
    for(let path=0;path<count;path++){
      let x=minX+random()*(maxX-minX),y=minY+random()*(maxY-minY);
      const points:Array<{position:THREE.Vector3;speed:number;color:THREE.Color}>=[];
      for(let i=0;i<steps;i++){
        if(x<0||x>domain.width||y<0||y>domain.height)break;
        const lon=toLongitude(x,domain),lat=toLatitude(y,domain);
        const wind=sample(lon,lat,height);this.samples++;
        if(!wind||wind.speed<.15)break;
        const position=surface.sample(lon,lat);if(!position)break;
        position.y+=height;
        points.push({position,speed:wind.speed,color:new THREE.Color(`rgb(${wind.color.join(',')})`)});
        x+=wind.u/wind.speed*step;y+=wind.v/wind.speed*step;
      }
      if(points.length<5)continue;
      const phase=random(),mean=points.reduce((sum,p)=>sum+p.speed,0)/points.length;
      // Around 500× advection for legibility; weather time itself is never advanced here.
      const rate=THREE.MathUtils.clamp(mean*(national?4000:500)/((points.length-1)*step*1000),national ? .08 : .12,national ? .45 : .65);
      for(let i=0;i<points.length-1;i++){
        const a=points[i],b=points[i+1],base=positions.length/3;
        for(const [p,q,sign,side,t] of [[a,b,1,-1,i],[a,b,1,1,i],[b,a,-1,-1,i+1],[b,a,-1,1,i+1]] as const){
          positions.push(p.position.x,p.position.y,p.position.z);next.push(q.position.x,q.position.y,q.position.z);
          forward.push(sign);sides.push(side);along.push(t/(points.length-1));seeds.push(phase);rates.push(rate);colors.push(p.color.r,p.color.g,p.color.b);
        }
        // The ribbon is expanded toward the camera in the vertex shader; keep its face CCW.
        indices.push(base,base+2,base+1,base+2,base+3,base+1);
      }
      this.paths++;
    }
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('aNext',new THREE.Float32BufferAttribute(next,3));
    for(const [name,array] of [['aForward',forward],['aSide',sides],['aAlong',along],['aSeed',seeds],['aRate',rates]] as const)geometry.setAttribute(name,new THREE.Float32BufferAttribute(array,1));
    geometry.setAttribute('aColor',new THREE.Float32BufferAttribute(colors,3));geometry.setIndex(indices);
    this.geometry.dispose();this.geometry=geometry;
    this.root.children.forEach(object=>{(object as THREE.Mesh).geometry=geometry;});
    this.segments=indices.length/6;this.builds++;
  }
  dispose(){this.geometry.dispose();this.core.dispose();this.shadow.dispose();this.localSurface.grids.clear();this.localSurface.water.clear();this.nationalSurface.clear();this.root.removeFromParent();}
}
