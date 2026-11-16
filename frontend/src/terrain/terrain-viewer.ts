import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { gpsToScene, sceneToGps } from './source/geo.js';
import { loadTerrain, type Tile } from './terrain-stream';
import { WindFlow } from './wind-flow';
import { loadNationalForecast, type NationalForecast } from './national-forecast';
import { LOCAL_DOMAIN,NATIONAL_DOMAIN } from './wind-domain';

const mounted = new WeakMap<HTMLElement,Promise<void>>();
const BASE = `${import.meta.env.BASE_URL}terrain/tunisia/`;
type Terrain = Awaited<ReturnType<typeof loadTerrain>>;
const visible = (object: THREE.Object3D) => { for(let node: THREE.Object3D|null=object;node;node=node.parent)if(!node.visible)return false;return true; };

export function mountTerrain(host: HTMLElement) {
  let task=mounted.get(host);
  if(!task){ task=createViewer(host);mounted.set(host,task);task.catch(()=>mounted.delete(host)); }
  return task;
}

async function createViewer(host: HTMLElement) {
  const text = (id:string,value:string) => {document.getElementById(id)!.textContent=value;};
  const status=(value:string)=>text('terrain-status',value);
  const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'low-power'});
  renderer.setPixelRatio(Math.min(devicePixelRatio,matchMedia('(max-width:700px)').matches?1.25:1.5));
  renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.18;
  renderer.domElement.setAttribute('aria-label','Terrain géographique de la Tunisie. Glissez pour tourner, utilisez la molette pour zoomer.');
  host.prepend(renderer.domElement);
  const scene=new THREE.Scene();scene.background=new THREE.Color('#102b36');
  const camera=new THREE.PerspectiveCamera(38,1,30,3000000);
  const controls=new OrbitControls(camera,renderer.domElement);
  controls.enableDamping=false;controls.minDistance=900;controls.maxDistance=2200000;controls.maxPolarAngle=Math.PI*.48;
  controls.zoomSpeed=.8;controls.panSpeed=.75;
  scene.add(new THREE.HemisphereLight(0xe2f1ff,0x8b704e,2.4));
  const sun=new THREE.DirectionalLight(0xffeed3,3.4);sun.position.set(-200000,300000,-170000);scene.add(sun);
  let terrain:Terrain|undefined,frame=0,frames=0,disposed=false,inViewport=false,selection=0,lodGeneration=0;
  const lifetime=new AbortController();
  let lodAbort=new AbortController(),lodQueue=Promise.resolve(),lodTimer:ReturnType<typeof setTimeout>;
  let height=80,windEnabled=true,windLoading=false,selected:{lon:number;lat:number;ground:number|null}|null=null;
  let windSource:'national'|'local'='national',sourceReady=false,sourceRevision=0,forecast:NationalForecast|undefined,forecastIndex=0;
  let nationalReplay:ReturnType<typeof setInterval>|null=null;
  let motionPaused=matchMedia('(prefers-reduced-motion: reduce)').matches,flowClock=0,lastWindFrame=0,lastWindKey='';
  let workspaceState=window.ATOM.workspace!.snapshot();
  const flow=new WindFlow();scene.add(flow.root);
  const marker=new THREE.Mesh(new THREE.SphereGeometry(1,16,10),new THREE.MeshBasicMaterial({color:'#f4d58b',depthTest:false}));
  marker.visible=false;marker.renderOrder=8;scene.add(marker);
  const outlinePoints:THREE.Vector3[]=[];
  for(const [lon,lat] of [[10.04,36.395],[10.04+68/89.42,36.395],[10.04+68/89.42,36.395+56/110.97],[10.04,36.395+56/110.97],[10.04,36.395]])outlinePoints.push(new THREE.Vector3(...gpsToScene(lon,lat,0)));
  const outline=new THREE.Line(new THREE.BufferGeometry().setFromPoints(outlinePoints),new THREE.LineDashedMaterial({color:'#aacbce',dashSize:1700,gapSize:1100,transparent:true,opacity:.75,depthTest:false}));outline.computeLineDistances();outline.renderOrder=2;scene.add(outline);
  const canRender=()=>!disposed&&!document.hidden&&inViewport&&!!host.offsetParent;
  const animating=()=>windEnabled&&sourceReady&&flow.paths>0&&!motionPaused&&(windSource==='national'||camera.position.distanceTo(controls.target)<280000);
  function render(){
    if(frame||!canRender())return;
    frame=requestAnimationFrame(now=>{
      frame=0;if(!canRender())return;
      if(animating()){if(lastWindFrame)flowClock+=Math.min(.05,(now-lastWindFrame)/1000);lastWindFrame=now;flow.tick(flowClock);}else lastWindFrame=0;
      if(marker.visible)marker.scale.setScalar(Math.max(35,Math.min(3000,camera.position.distanceTo(controls.target)*.004)));
      renderer.render(scene,camera);frames++;
      if(animating())render();
    });
  }
  function resize(){if(!host.clientWidth||!host.clientHeight)return;renderer.setSize(host.clientWidth,host.clientHeight,false);camera.aspect=host.clientWidth/host.clientHeight;camera.updateProjectionMatrix();flow.resize(renderer.domElement.width,renderer.domElement.height,renderer.getPixelRatio());render();}
  const sizeObserver=new ResizeObserver(resize);sizeObserver.observe(host);
  const viewObserver=new IntersectionObserver(entries=>{inViewport=entries[0].isIntersecting;lastWindFrame=0;if(!inViewport){lodAbort.abort();if(frame){cancelAnimationFrame(frame);frame=0;}}else {resize();render();scheduleDetail();}});viewObserver.observe(host);
  const onVisibility=()=>{lastWindFrame=0;if(document.hidden&&frame){cancelAnimationFrame(frame);frame=0;}else render();};document.addEventListener('visibilitychange',onVisibility);
  controls.addEventListener('change',()=>{syncView();render();});
  controls.addEventListener('end',()=>scheduleDetail());
  const quality=document.getElementById('terrain-quality') as HTMLSelectElement;
  function syncView(){
    const country=camera.position.distanceTo(controls.target)>280000;
    const title=document.getElementById('terrain-title')!,name=windSource==='national'||country?'Tunisie':'Grand Tunis';if(title.textContent!==name)title.textContent=name;
    document.getElementById('terrain-return-wind')!.hidden=windSource==='national'||!country||!windEnabled;
    document.getElementById('terrain-country')!.setAttribute('aria-pressed',String(country));
    document.getElementById('terrain-local')!.setAttribute('aria-pressed',String(!country));
  }
  function country(){controls.target.set(40000,0,40000);camera.position.set(40000,1400000,520000);controls.update();syncView();render();scheduleDetail();}
  function focus(lon:number,lat:number,distance=110000){const p=new THREE.Vector3(...gpsToScene(lon,lat,0));controls.target.copy(p);camera.position.copy(p).add(new THREE.Vector3(distance*.45,distance*.7,distance*.55));controls.update();render();scheduleDetail();}
  function local(){focus(10.36,36.68,Math.max(95000,95000*1.05/camera.aspect));}
  function scheduleDetail(){clearTimeout(lodTimer);lodTimer=setTimeout(()=>{
    lodAbort.abort();lodAbort=new AbortController();const signal=lodAbort.signal,generation=++lodGeneration;
    lodQueue=lodQueue.catch(()=>{}).then(()=>updateDetail(signal,generation));
  },180);}
  const tileBox=(tile:Tile)=>{const [w,s,e,n]=tile.projectedBounds;return new THREE.Box3(new THREE.Vector3(w-500000,-100,3800000-n),new THREE.Vector3(e-500000,Math.max(2000,tile.elevationRangeMeters[1]),3800000-s));};
  async function updateDetail(signal:AbortSignal,generation:number){
    if(!terrain||signal.aborted||disposed||!host.offsetParent)return;
    updateVectors();
    const distance=camera.position.distanceTo(controls.target),mobile=matchMedia('(max-width:700px)').matches;
    const roots=[...terrain.tiles.values()].filter(tile=>tile.lod===0);
    if(distance>300000){roots.forEach(tile=>terrain!.collapse(tile.id));text('terrain-detail','Vue d’ensemble · relief 512 m');status('Terrain chargé · échelle physique 1:1');return;}
    camera.updateMatrixWorld();const frustum=new THREE.Frustum().setFromProjectionMatrix(new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse));
    const closest=(tile:Tile)=>tileBox(tile).distanceToPoint(controls.target);
    const candidates=roots.filter(tile=>frustum.intersectsBox(tileBox(tile))).sort((a,b)=>closest(a)-closest(b));
    const wanted=new Set(candidates.slice(0,mobile||quality.value==='low'?1:2).map(tile=>tile.id));
    roots.filter(tile=>!wanted.has(tile.id)).forEach(tile=>terrain!.collapse(tile.id));
    status('Chargement du relief autour de la caméra…');
    try{
      for(const id of wanted){signal.throwIfAborted();await terrain.refine(id,signal);}
      const regional=[...terrain.resident.keys()].map(id=>terrain!.tiles.get(id)!).filter(tile=>tile.lod===1);
      const fine=quality.value!=='low'&&distance<(mobile?22000:45000);
      const targetTile=fine?regional.filter(tile=>frustum.intersectsBox(tileBox(tile))).sort((a,b)=>closest(a)-closest(b))[0]:undefined;
      for(const tile of regional)if(tile.id!==targetTile?.id)terrain.collapse(tile.id);
      if(targetTile)await terrain.refine(targetTile.id,signal);
      if(generation===lodGeneration){text('terrain-detail',targetTile?'Détail local · grille 32 m':'Relief régional · grille 128 m');status('Terrain chargé · échelle physique 1:1');}
    }catch(error){if(!signal.aborted){status('Le détail n’a pas pu se charger. Le relief précédent reste visible.');console.error(error);}}
    render();
  }
  const gridLoading=new Map<string,Promise<number|null>>();
  async function ground(lon:number,lat:number):Promise<number|null>{
    if(!terrain)return null;
    const p=gpsToScene(lon,lat),key=`7-${Math.floor((p[0]+500000)/8192)}-${Math.floor(-p[2]/8192+(3800000-3276800)/8192)}`;
    const pending=gridLoading.get(key);
    if(pending)await pending;
    const query=terrain.elevation.sample(lon,lat) as Promise<number|null>;
    gridLoading.set(key,query);
    try{return await query;}finally{if(gridLoading.get(key)===query)gridLoading.delete(key);}
  }
  function syncMotion(){
    const button=document.getElementById('terrain-motion')!;
    button.textContent=motionPaused?'▶':'Ⅱ';
    button.setAttribute('aria-label',motionPaused?'Animer le vent':'Mettre le vent en pause');
    button.setAttribute('aria-pressed',String(!motionPaused));
    lastWindFrame=0;
  }
  const sourceSelect=document.getElementById('terrain-source') as HTMLSelectElement;
  const daySelect=document.getElementById('terrain-day') as HTMLSelectElement;
  const localDate=(epoch:number,options:Intl.DateTimeFormatOptions)=>new Intl.DateTimeFormat(document.documentElement.lang||'fr',{timeZone:'Africa/Tunis',...options}).format(new Date(epoch*1000));
  function fieldSample(lon:number,lat:number,z:number){
    if(!sourceReady)return null;
    const result=windSource==='national'?forecast?.sample(lon,lat,z,forecastIndex):window.ATOM.workspace?.terrainWind?.(lon,lat,z);
    if(!result)return null;
    return {...result,formatted:workspaceState.speed.includes('m/s')?`${result.speed.toFixed(1).replace('.',',')} m/s`:`${Math.round(result.speed*3.6)} km/h`};
  }
  function playerState(){
    const globalPlay=document.getElementById('play');
    const localPlaying=(window.ATOM.texteFr?.(globalPlay)??globalPlay?.textContent)==='Pause';
    document.getElementById('terrain-replay')?.setAttribute('aria-pressed',String(windSource==='national'?!!nationalReplay:localPlaying));
  }
  function stopReplay(){if(nationalReplay){clearInterval(nationalReplay);nationalReplay=null;}playerState();}
  function updateVectors(){
    const state=workspaceState,position=sceneToGps(controls.target.toArray());
    const distanceKm=camera.position.distanceTo(controls.target)/1000;
    const viewKey=windSource==='national'?`${Math.round(position.longitude*100)}/${Math.round(position.latitude*100)}/${Math.round(distanceKm)}/${camera.aspect.toFixed(2)}`:'';
    const key=`${windSource}/${height}/${windSource==='national'?forecast?.data.times[forecastIndex]:state.hour}/${state.source}/${quality.value}/${viewKey}`;
    if(windEnabled&&sourceReady&&flow.ready&&key!==lastWindKey){
      flow.rebuild(height,matchMedia('(max-width:700px)').matches||quality.value==='low',fieldSample,{longitude:position.longitude,latitude:position.latitude,distanceKm,aspect:camera.aspect});
      lastWindKey=key;
    }
    flow.root.visible=windEnabled&&sourceReady;
    outline.visible=windSource==='local';
    const national=windSource==='national',time=forecast?.data.times[forecastIndex];
    const hour=national&&forecast?forecastIndex%24:state.hour;
    const date=national&&time?localDate(time,{day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'}):'';
    const label=national?'Prévision GFS':state.source==='hors_ligne'?'Champ local indisponible':'PC-FNO3D · modèle local';
    text('terrain-wind-source',windEnabled?`${label} · ${height} m AGL · ${national?date:`${hour} h`}`:'Vent masqué');
    text('terrain-live-meta',`${label} · ${height} m du sol${national&&date?` · ${date}`:''}`);
    text('terrain-native-hour',`${hour} h`);
    (document.getElementById('terrain-hour') as HTMLInputElement).value=String(hour);
    daySelect.hidden=!national||!forecast;
    if(national&&forecast)daySelect.value=String(Math.floor(forecastIndex/24));
    const metres=state.speed.includes('m/s');text('terrain-legend-max',metres?'17,2 m/s et plus':'62 km/h et plus');
    document.getElementById('terrain-wind-legend')!.hidden=!windEnabled;
    document.getElementById('terrain-wind')!.setAttribute('aria-pressed',String(windEnabled));
    const heightSelect=document.getElementById('terrain-height') as HTMLSelectElement;
    heightSelect.disabled=!windEnabled;
    for(const option of heightSelect.options)option.hidden=national&&![10,80].includes(+option.value);
    heightSelect.value=String(height);
    if(national&&forecast){
      text('terrain-forecast-provenance',`NOAA GFS via Open-Meteo · grille de lecture 0,5° · récupéré le ${localDate(forecast.data.retrievedAt/1000,{day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'})}${forecast.stale?' · cache':''}`);
    }else text('terrain-forecast-provenance',national?'Prévision nationale en cours de chargement.':'Champ local limité au Grand Tunis.');
    sourceSelect.querySelector<HTMLOptionElement>('[value="local"]')!.textContent=state.source==='hors_ligne'?'Grand Tunis · indisponible':'Grand Tunis · PC-FNO3D';
    queueMicrotask(playerState);updatePoint();syncView();render();
  }
  async function prepareWind(){
    if(!terrain)return;
    const revision=++sourceRevision,source=windSource,domain=source==='national'?NATIONAL_DOMAIN:LOCAL_DOMAIN;
    windLoading=true;sourceReady=false;flow.root.visible=false;
    text('terrain-flow-state',source==='national'?'Chargement des prévisions nationales…':'Préparation du champ local…');
    document.getElementById('terrain-forecast-retry')!.hidden=true;
    try{
      const [nextForecast]=await Promise.all([
        source==='national'?loadNationalForecast(lifetime.signal):Promise.resolve(undefined),
        flow.prepare(terrain,lifetime.signal,domain),
      ]);
      if(disposed||revision!==sourceRevision)return;
      if(nextForecast){
        const previousTime=forecast?.data.times[forecastIndex];forecast=nextForecast;
        const previous=previousTime?forecast.data.times.indexOf(previousTime):-1;forecastIndex=previous>=0?previous:forecast.nearestHour;
        daySelect.replaceChildren();
        for(let day=0;day*24<forecast.data.times.length;day++){const option=document.createElement('option');option.value=String(day);option.textContent=localDate(forecast.data.times[day*24],{day:'2-digit',month:'short'});daySelect.appendChild(option);}
      }
      flow.activate(domain);sourceReady=true;lastWindKey='';updateVectors();text('terrain-flow-state','');
    }catch(error){if(!lifetime.signal.aborted&&revision===sourceRevision){
      text('terrain-flow-state',source==='national'?'Prévision nationale indisponible. Aucune météo de remplacement n’est affichée.':'Champ local indisponible. Réessayez le chargement.');
      document.getElementById('terrain-forecast-retry')!.hidden=false;console.error(error);
    }}finally{if(revision===sourceRevision)windLoading=false;}
  }
  function toggleWind(){windEnabled=!windEnabled;if(windEnabled&&!sourceReady)void prepareWind();updateVectors();}
  sourceSelect.onchange=()=>{
    stopReplay();windSource=sourceSelect.value==='local'?'local':'national';
    if(windSource==='national'&&![10,80].includes(height))height=80;
    if(windSource==='national')country();else local();
    void prepareWind();updateVectors();
    window.ATOM.workspace?.snapshot();
  };
  daySelect.onchange=()=>{stopReplay();forecastIndex=+daySelect.value*24+forecastIndex%24;updateVectors();};
  document.getElementById('terrain-forecast-retry')!.onclick=()=>void prepareWind();
  function changeHour(value:number){
    if(windSource==='national'){forecastIndex=Math.floor(forecastIndex/24)*24+value;updateVectors();}
    else window.ATOM.setHour(value);
  }
  function replay(){
    if(windSource==='local'){document.getElementById('play')!.click();updateVectors();return;}
    if(nationalReplay){stopReplay();return;}
    if(!forecast||!sourceReady)return;
    changeHour(0);nationalReplay=setInterval(()=>{const next=forecastIndex%24+1;if(next>=24){stopReplay();return;}changeHour(next);},700);playerState();
  }
  function updatePoint(){
    const button=document.getElementById('terrain-open-point') as HTMLButtonElement;
    if(!selected){button.hidden=true;return;}
    const sample=fieldSample(selected.lon,selected.lat,height);
    text('terrain-wind-reading',sample?`${sample.formatted} · ${sample.direction} · ${height} m du sol`:windSource==='national'?'Prévision indisponible à ce point':'Hors du champ de vent Grand Tunis');
    const localAvailable=window.ATOM.workspace?.terrainWind?.(selected.lon,selected.lat,height);
    button.hidden=!sample||!localAvailable;
    button.textContent=windSource==='national'?'Comparer au champ local ↗':'Ouvrir les conditions locales ↗';
    text('terrain-point-source',windSource==='national'?'Prévision nationale · NOAA GFS':workspaceState.source==='hors_ligne'?'Champ local indisponible':'PC-FNO3D · Grand Tunis');
  }

  async function selectPoint(lon:number,lat:number,meshHeight?:number){
    const revision=++selection;selected={lon,lat,ground:meshHeight??null};
    text('terrain-coordinate',`${lat.toFixed(5)}° N · ${lon.toFixed(5)}° E`);
    document.getElementById('terrain-selection')!.hidden=false;updatePoint();
    let h:number|null=null;
    try{h=await ground(lon,lat);}catch{ /* Display mesh remains a labelled fallback. */ }
    if(revision!==selection||disposed)return;
    const measured=h!==null;h??=meshHeight??null;selected.ground=h;
    text('terrain-elevation',h===null?'Altitude indisponible à ce point':`${measured?'Sol DEM':'Relief affiché'} : ${h.toFixed(1).replace('.',',')} m EGM2008`);
    marker.visible=h!==null;if(h!==null)marker.position.fromArray(gpsToScene(lon,lat,h+8));render();
  }
  function closePoint(){selection++;selected=null;marker.visible=false;document.getElementById('terrain-selection')!.hidden=true;render();}
  document.getElementById('terrain-close-point')!.onclick=closePoint;
  const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();let down:[number,number]|null=null;
  renderer.domElement.addEventListener('pointerdown',event=>{down=[event.clientX,event.clientY];});
  renderer.domElement.addEventListener('pointerup',event=>{
    if(!terrain||!down||Math.hypot(event.clientX-down[0],event.clientY-down[1])>5)return;
    const box=renderer.domElement.getBoundingClientRect();mouse.set((event.clientX-box.left)/box.width*2-1,1-(event.clientY-box.top)/box.height*2);ray.setFromCamera(mouse,camera);
    const hit=ray.intersectObjects([terrain.land,terrain.water],true).find(item=>visible(item.object));
    if(hit){const point=sceneToGps(hit.point.toArray());void selectPoint(point.longitude,point.latitude,point.heightEGM2008);}
  });
  host.onkeydown=event=>{
    if(event.key==='Escape'){closePoint();host.querySelector<HTMLDetailsElement>('.terrain-options')!.open=false;}
    if(event.target!==host)return;
    if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)){
      event.preventDefault();const offset=camera.position.clone().sub(controls.target),spherical=new THREE.Spherical().setFromVector3(offset);
      if(event.key==='ArrowLeft')spherical.theta-=.1;if(event.key==='ArrowRight')spherical.theta+=.1;if(event.key==='ArrowUp')spherical.phi=Math.max(.12,spherical.phi-.08);if(event.key==='ArrowDown')spherical.phi=Math.min(Math.PI*.47,spherical.phi+.08);
      camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(spherical));controls.update();render();scheduleDetail();
    }else if(event.key==='+'||event.key==='-'||event.key==='='){
      event.preventDefault();const distance=camera.position.distanceTo(controls.target),next=THREE.MathUtils.clamp(distance*(event.key==='-'?1.2:.8),controls.minDistance,controls.maxDistance);camera.position.sub(controls.target).multiplyScalar(next/distance).add(controls.target);controls.update();render();scheduleDetail();
    }
  };
  document.getElementById('terrain-country')!.onclick=country;
  document.getElementById('terrain-local')!.onclick=local;
  document.getElementById('terrain-return-wind')!.onclick=local;
  document.getElementById('terrain-wind')!.onclick=toggleWind;
  document.getElementById('terrain-motion')!.onclick=()=>{motionPaused=!motionPaused;syncMotion();render();};
  (document.getElementById('terrain-hour') as HTMLInputElement).oninput=event=>changeHour(+(event.target as HTMLInputElement).value);
  document.getElementById('terrain-replay')!.onclick=replay;
  document.getElementById('terrain-fullscreen')!.onclick=()=>{
    if(document.fullscreenElement===host)void document.exitFullscreen();else void host.requestFullscreen().catch(()=>{});
  };
  (document.getElementById('terrain-height') as HTMLSelectElement).onchange=event=>{height=+(event.target as HTMLSelectElement).value;updateVectors();};
  quality.onchange=()=>{scheduleDetail();updateVectors();};
  document.getElementById('terrain-open-point')!.onclick=()=>{if(selected)window.ATOM.workspace?.terrainPoint?.(selected.lon,selected.lat);};
  const locate=document.getElementById('terrain-locate') as HTMLButtonElement;
  locate.onclick=()=>{
    if(!navigator.geolocation){status('La géolocalisation n’est pas disponible.');return;}
    locate.disabled=true;status('Recherche de votre position…');
    navigator.geolocation.getCurrentPosition(position=>{
      locate.disabled=false;const {longitude:lon,latitude:lat}=position.coords;
      if(lon<7.52||lon>11.61||lat<30.23||lat>37.55){status('Votre position est en dehors du terrain tunisien fourni.');return;}
      focus(lon,lat,30000);void selectPoint(lon,lat);status('Position horizontale GPS · altitude du terrain DEM');
    },()=>{locate.disabled=false;status('Position indisponible. Vous pouvez sélectionner un lieu sur le terrain.');},{timeout:12000,maximumAge:60000});
  };
  const onState=(event:Event)=>{workspaceState=(event as CustomEvent).detail;if(windEnabled||selected)updateVectors();};window.addEventListener('atom:state',onState);
  window.ATOM.terrainStats=()=>({ready:!!terrain,windReady:sourceReady&&flow.paths>0,windSource,forecastTime:forecast?.data.times[forecastIndex],forecastRetrievedAt:forecast?.data.retrievedAt,forecastLocations:forecast?.data.points.length,frames,requests:terrain?.requests??0,resident:terrain?.resident.size??0,active:terrain?.active.size??0,activeIds:terrain?[...terrain.active]:[],windVectors:flow.paths,windEnabled,motionPaused,flowTime:flow.time,flowSegments:flow.segments,flowBuilds:flow.builds,windDemTiles:flow.surface.grids.size,waterSamples:flow.surface.waterSamples,height,selected,triangles:renderer.info.render.triangles,geometries:renderer.info.memory.geometries});
  window.ATOM.terrainExplore={focus,select:selectPoint,country,sample:(lon:number,lat:number)=>fieldSample(lon,lat,height)};
  function dispose(){
    if(disposed)return;disposed=true;stopReplay();lifetime.abort();selection++;lodAbort.abort();clearTimeout(lodTimer);cancelAnimationFrame(frame);
    sizeObserver.disconnect();viewObserver.disconnect();controls.dispose();terrain?.dispose();
    flow.dispose();
    scene.traverse(node=>{if(node instanceof THREE.Mesh||node instanceof THREE.Line){node.geometry.dispose();const materials=Array.isArray(node.material)?node.material:[node.material];materials.forEach(item=>item.dispose());}});
    renderer.dispose();renderer.domElement.remove();document.removeEventListener('visibilitychange',onVisibility);window.removeEventListener('atom:state',onState);
  }
  window.addEventListener('pagehide',dispose,{once:true});
  resize();country();syncMotion();status('Chargement du relief de la Tunisie…');
  try{
    terrain=await loadTerrain(BASE,render,lifetime.signal);scene.add(terrain.root);await terrain.loadWater();
    document.getElementById('terrain-loading')!.hidden=true;
    document.querySelectorAll<HTMLButtonElement|HTMLSelectElement>('[data-terrain-ready]').forEach(button=>button.disabled=false);
    updateVectors();void prepareWind();scheduleDetail();
    status('Terrain chargé · échelle physique 1:1');render();
  }catch(error){dispose();throw error;}
}
