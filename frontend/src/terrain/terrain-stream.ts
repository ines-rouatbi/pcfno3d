import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { ElevationQueries } from './source/geo.js';

export type Tile = {
  id: string; lod: number; uri: string; overviewNodeName?: string;
  projectedBounds: [number,number,number,number]; children?: string[]; parent?: string;
  elevationRangeMeters: [number,number]; elevation?: any;
};
export async function loadTerrain(base: string, changed: () => void, signal?: AbortSignal) {
  const baseUrl = new URL(base, location.href), lifetime = new AbortController();
  if(signal?.aborted)lifetime.abort();
  signal?.addEventListener('abort',()=>lifetime.abort(),{once:true});
  const response = await fetch(new URL('terrain-manifest.json', baseUrl), { signal: lifetime.signal });
  if (!response.ok) throw new Error(`Terrain : HTTP ${response.status}`);
  const manifest = await response.json() as { tiles: Tile[] };
  const tiles = new Map(manifest.tiles.map(tile => [tile.id,tile]));
  const root = new THREE.Group(), land = new THREE.Group(), water = new THREE.Group();
  root.name = 'ATOM_Wind_Tunisia_Terrain'; root.add(land,water);
  const material = new THREE.MeshStandardMaterial({vertexColors:true,roughness:1,metalness:0});
  const waterMaterial = new THREE.MeshBasicMaterial({color:'#102b36',toneMapped:false,polygonOffset:true,polygonOffsetFactor:-1,polygonOffsetUnits:-1});
  const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
  const resident = new Map<string,THREE.Object3D>(), active = new Set<string>(), epochs = new Map<string,number>();
  let requests = 0, disposed = false;
  const release = (object: THREE.Object3D) => object.traverse(node => { if (node instanceof THREE.Mesh) node.geometry.dispose(); });
  async function asset(uri: string, surface: THREE.Material, signal?: AbortSignal) {
    requests++;
    const effective = signal ? AbortSignal.any([signal,lifetime.signal]) : lifetime.signal;
    const result = await fetch(new URL(uri,baseUrl), {signal:effective});
    if (!result.ok) throw new Error(`${uri} : HTTP ${result.status}`);
    const gltf = await loader.parseAsync(await result.arrayBuffer(),baseUrl.href);
    gltf.scene.traverse(node => {
      if (!(node instanceof THREE.Mesh)) return;
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      materials.forEach(item=>item.dispose()); node.material=surface; node.castShadow=false;
    });
    if (effective.aborted || disposed) { release(gltf.scene); throw new DOMException('Cancelled','AbortError'); }
    return gltf.scene;
  }
  const overview = await asset('overview.glb',material); land.add(overview);
  for (const tile of manifest.tiles.filter(item=>item.lod===0)) {
    const node = overview.getObjectByName(tile.overviewNodeName!);
    if (!node) throw new Error(`Secteur absent : ${tile.id}`);
    resident.set(tile.id,node); active.add(tile.id);
  }
  async function refine(id: string, signal: AbortSignal) {
    const parent = tiles.get(id);
    if (!parent?.children?.length || !active.has(id)) return;
    const epoch = epochs.get(id) || 0;
    const prepared: Array<[string,THREE.Object3D]> = [];
    try {
      for (let start=0;start<parent.children.length;start+=4) {
        signal.throwIfAborted();
        const ids = parent.children.slice(start,start+4);
        const batch = await Promise.allSettled(ids.map(async child => [child,await asset(tiles.get(child)!.uri,material,signal)] as [string,THREE.Object3D]));
        for (const item of batch) if(item.status==='fulfilled') prepared.push(item.value);
        const failure=batch.find(item=>item.status==='rejected');
        if(failure?.status==='rejected') throw failure.reason;
      }
      signal.throwIfAborted();
      if(disposed || epoch!==(epochs.get(id)||0)) throw new DOMException('Obsolete camera selection','AbortError');
      // Replace atomically: a parent and its descendants never draw together.
      for (const [child,node] of prepared) { land.add(node); resident.set(child,node); active.add(child); }
      resident.get(id)!.visible=false; active.delete(id); changed();
    } catch(error) { prepared.forEach(([,node])=>release(node)); throw error; }
  }
  function collapse(id: string) {
    epochs.set(id,(epochs.get(id)||0)+1);
    const tile=tiles.get(id); if(!tile || !resident.has(id)) return;
    const discard = (child: string) => {
      epochs.set(child,(epochs.get(child)||0)+1);
      tiles.get(child)?.children?.forEach(discard);
      const node=resident.get(child); if(node) {node.removeFromParent();release(node);resident.delete(child);}
      active.delete(child);
    };
    tile.children?.forEach(discard);resident.get(id)!.visible=true;active.add(id);changed();
  }
  async function loadWater() {
    const result=await fetch(new URL('water/water-manifest.json',baseUrl),{signal:lifetime.signal});
    if(!result.ok)throw new Error(`Eau : HTTP ${result.status}`);
    const metadata=await result.json();
    for(const item of metadata.assets)water.add(await asset(item.uri,waterMaterial));
    changed();
  }
  function dispose() {disposed=true;lifetime.abort();release(root);material.dispose();waterMaterial.dispose();root.removeFromParent();}
  return {root,land,water,tiles,resident,active,refine,collapse,loadWater,dispose,
    elevation:new ElevationQueries(manifest,baseUrl),get requests(){return requests;}};
}
