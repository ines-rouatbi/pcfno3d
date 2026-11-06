import * as THREE from 'three';
import { gpsToScene } from './source/geo.js';
import type { loadTerrain } from './terrain-stream';
import { NATIONAL_DOMAIN as DOMAIN } from './wind-domain';

/** National display-height raster, derived from the supplied overview and water meshes. */
export class NationalSurface {
  grids=new Map();waterSamples=0;
  private step=1000;private width=0;private rows=0;private originX=0;private originZ=0;
  private values=new Float32Array();
  async load(terrain:Awaited<ReturnType<typeof loadTerrain>>,signal:AbortSignal){
    const corners=[[DOMAIN.west,DOMAIN.south],[DOMAIN.east,DOMAIN.south],[DOMAIN.west,DOMAIN.north],[DOMAIN.east,DOMAIN.north]].map(([lon,lat])=>gpsToScene(lon,lat));
    this.originX=Math.floor(Math.min(...corners.map(p=>p[0]))/this.step)*this.step-this.step;
    this.originZ=Math.floor(Math.min(...corners.map(p=>p[2]))/this.step)*this.step-this.step;
    this.width=Math.ceil((Math.max(...corners.map(p=>p[0]))-this.originX)/this.step)+2;
    this.rows=Math.ceil((Math.max(...corners.map(p=>p[2]))-this.originZ)/this.step)+2;
    this.values=new Float32Array(this.width*this.rows).fill(NaN);
    terrain.root.updateMatrixWorld(true);
    const a=new THREE.Vector3(),b=new THREE.Vector3(),c=new THREE.Vector3();
    const draw=(object:THREE.Object3D)=>object.traverse(node=>{
      if(!(node instanceof THREE.Mesh))return;
      const position=node.geometry.attributes.position,index=node.geometry.index,count=index?index.count:position.count;
      for(let offset=0;offset<count;offset+=3){
        a.fromBufferAttribute(position,index?index.getX(offset):offset).applyMatrix4(node.matrixWorld);
        b.fromBufferAttribute(position,index?index.getX(offset+1):offset+1).applyMatrix4(node.matrixWorld);
        c.fromBufferAttribute(position,index?index.getX(offset+2):offset+2).applyMatrix4(node.matrixWorld);
        const minX=Math.max(0,Math.ceil((Math.min(a.x,b.x,c.x)-this.originX)/this.step)),maxX=Math.min(this.width-1,Math.floor((Math.max(a.x,b.x,c.x)-this.originX)/this.step));
        const minZ=Math.max(0,Math.ceil((Math.min(a.z,b.z,c.z)-this.originZ)/this.step)),maxZ=Math.min(this.rows-1,Math.floor((Math.max(a.z,b.z,c.z)-this.originZ)/this.step));
        const den=(b.z-c.z)*(a.x-c.x)+(c.x-b.x)*(a.z-c.z);if(Math.abs(den)<1e-8)continue;
        for(let row=minZ;row<=maxZ;row++)for(let column=minX;column<=maxX;column++){
          const x=this.originX+column*this.step,z=this.originZ+row*this.step;
          const u=((b.z-c.z)*(x-c.x)+(c.x-b.x)*(z-c.z))/den,v=((c.z-a.z)*(x-c.x)+(a.x-c.x)*(z-c.z))/den,w=1-u-v;
          if(u>=-1e-7&&v>=-1e-7&&w>=-1e-7)this.values[row*this.width+column]=u*a.y+v*b.y+w*c.y;
        }
      }
    });
    for(const tile of terrain.tiles.values())if(tile.lod===0){signal.throwIfAborted();const node=terrain.resident.get(tile.id);if(node)draw(node);}
    draw(terrain.water);
  }
  sample(lon:number,lat:number){
    const p=gpsToScene(lon,lat),x=(p[0]-this.originX)/this.step,z=(p[2]-this.originZ)/this.step;
    if(x<0||z<0||x>=this.width-1||z>=this.rows-1)return null;
    const i=Math.floor(x),j=Math.floor(z),dx=x-i,dz=z-j;
    const a=this.values[j*this.width+i],b=this.values[j*this.width+i+1],c=this.values[(j+1)*this.width+i],d=this.values[(j+1)*this.width+i+1];
    if(![a,b,c,d].every(Number.isFinite))return null;
    const ground=(a*(1-dx)+b*dx)*(1-dz)+(c*(1-dx)+d*dx)*dz;
    return new THREE.Vector3(p[0],ground,p[2]);
  }
  clear(){this.values=new Float32Array();}
}
