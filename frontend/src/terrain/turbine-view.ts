import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Vue 3D de « Mon éolienne » (espace éolien) : la machine à l'échelle du modèle choisi,
// le rotor face au vent du moyeu, et une flèche par hauteur du rotor dont la longueur suit
// la vitesse — c'est le cisaillement que la carte 3D du vent permet de connaître.
// Aucun texte ici : l'extracteur i18n ne lit pas ce fichier, les libellés viennent d'atom-runtime.js.

export interface RotorProfile { z: number; sp: number; rgb: [number, number, number] }
export interface RotorData {
  hubZ: number;            // hauteur du moyeu au-dessus du sol (m)
  R: number;               // rayon du rotor (m)
  profile: RotorProfile[]; // du bas au haut des pales
  u: number; v: number;    // vent au moyeu, m/s (u est, v nord)
}
export interface RotorView { update(d: RotorData): void }

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';

function bladeGeometry(R: number) {
  // pale effilée : corde maximale vers 20 % du rayon, bord d'attaque droit, bord de fuite galbé
  const s = new THREE.Shape(), r0 = .06 * R;
  s.moveTo(-.035 * R, r0);
  s.bezierCurveTo(-.06 * R, .14 * R, -.075 * R, .3 * R, -.045 * R, .62 * R);
  s.quadraticCurveTo(-.02 * R, .9 * R, -.004 * R, R);
  s.lineTo(.012 * R, R);
  s.quadraticCurveTo(.02 * R, .55 * R, .03 * R, .2 * R);
  s.lineTo(.03 * R, r0);
  s.closePath();
  const g = new THREE.ExtrudeGeometry(s, { depth: .018 * R, bevelEnabled: false, curveSegments: 10 });
  g.translate(0, 0, -.009 * R);
  return g;
}

function disposeTree(o: THREE.Object3D) {
  o.traverse(n => {
    const m = n as THREE.Mesh;
    m.geometry?.dispose();
    const mat = m.material as THREE.Material | THREE.Material[] | undefined;
    (Array.isArray(mat) ? mat : mat ? [mat] : []).forEach(x => x.dispose());
  });
}

export async function mountTurbine(host: HTMLElement, opts: { label: string }): Promise<RotorView> {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const canvas = renderer.domElement;
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', opts.label);
  host.replaceChildren(canvas);
  host.classList.add('is-3d');

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 1, 5000);
  const controls = new OrbitControls(camera, canvas);
  controls.enablePan = false; controls.enableDamping = true; controls.dampingFactor = .12;
  controls.maxPolarAngle = Math.PI * .49; controls.minPolarAngle = Math.PI * .12;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f86, 2.2));
  const sun = new THREE.DirectionalLight(0xffffff, 2.4); sun.position.set(-1, 2, 1.4); scene.add(sun);

  // repère : l'éolienne regarde toujours +Z (vers la caméra de profil) ; c'est le sol et sa
  // flèche du nord qui tournent avec la direction du vent, la lecture des flèches reste la même.
  const ground = new THREE.Group(), machine = new THREE.Group(), rotor = new THREE.Group(), wind = new THREE.Group();
  scene.add(ground, machine, wind);

  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  let current: RotorData | null = null, pending: RotorData | null = null, dirty = true, themeKey = '';
  let omega = 0, angle = 0, last = performance.now(), lastDraw = 0, visible = false, frame = 0, built = '', framed = '';

  // flèches et pointillés créés une fois puis réutilisés : un changement d'heure ne fait que
  // déplacer, étirer et recolorer des objets existants (ni géométrie ni matériau neufs)
  const SHAFT = new THREE.CylinderGeometry(1, 1, 1, 12).translate(0, .5, 0);
  const HEAD = new THREE.ConeGeometry(1, 1, 16).translate(0, .5, 0);
  const arrows: { g: THREE.Group; shaft: THREE.Mesh; cone: THREE.Mesh; mat: THREE.MeshStandardMaterial }[] = [];
  const lineMat = new THREE.LineDashedMaterial({ transparent: true, opacity: .55 });
  const lines = Array.from({ length: 4 }, () => {
    const l = new THREE.Line(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(6), 3)), lineMat);
    l.frustumCulled = false; wind.add(l); return l;
  });
  const setLine = (l: THREE.Line, a: THREE.Vector3, b: THREE.Vector3) => {
    const pos = l.geometry.getAttribute('position') as THREE.BufferAttribute;
    pos.setXYZ(0, a.x, a.y, a.z); pos.setXYZ(1, b.x, b.y, b.z); pos.needsUpdate = true;
    l.geometry.computeBoundingBox(); l.geometry.computeBoundingSphere();   // le cadrage les lit
    l.computeLineDistances();
  };
  function arrowAt(i: number) {
    while (arrows.length <= i) {
      const mat = new THREE.MeshStandardMaterial({ roughness: .55 }), g = new THREE.Group();
      const shaft = new THREE.Mesh(SHAFT, mat), cone = new THREE.Mesh(HEAD, mat);
      g.add(shaft, cone); g.rotation.x = -Math.PI / 2;  // +Y -> -Z : le vent traverse le rotor vers l'aval
      wind.add(g); arrows.push({ g, shaft, cone, mat });
    }
    return arrows[i];
  }

  function build(d: RotorData) {
    const key = `${d.hubZ}|${d.R}`;
    if (key === built) return;
    built = key;
    for (const g of [ground, machine]) { disposeTree(g); g.clear(); }
    themeKey = '';                             // matériaux du sol neufs : à recolorer
    rotor.clear();
    const { hubZ: H, R } = d, white = new THREE.MeshStandardMaterial({ color: 0xeef0ee, roughness: .5, metalness: .05 });
    const tower = new THREE.Mesh(new THREE.CylinderGeometry(.017 * H + .004 * R, .03 * H + .006 * R, H, 28), white);
    tower.position.y = H / 2;
    const nl = .3 * R, nh = .085 * R;
    const nacelle = new THREE.Mesh(new THREE.BoxGeometry(nh * 1.05, nh, nl), white);
    nacelle.position.set(0, H + nh * .15, -nl * .22);
    const spinner = new THREE.Mesh(new THREE.SphereGeometry(nh * .62, 24, 16, 0, Math.PI * 2, 0, Math.PI / 2), white);
    spinner.rotation.x = Math.PI / 2;
    const blade = bladeGeometry(R);
    for (let i = 0; i < 3; i++) {
      const b = new THREE.Mesh(blade, white);
      b.rotation.z = i * 2 * Math.PI / 3;
      b.rotateY(.18);                          // calage de la pale autour de son axe
      rotor.add(b);
    }
    rotor.add(spinner);
    rotor.position.set(0, H, nl * .32);
    machine.add(tower, nacelle, rotor);
    // disque balayé, à peine visible
    const disk = new THREE.Mesh(new THREE.RingGeometry(.07 * R, R, 64),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: .07, side: THREE.DoubleSide, depthWrite: false }));
    disk.position.copy(rotor.position); machine.add(disk);
    // sol : disque, cercle de boussole et flèche du nord
    const G = Math.max(1.55 * R, .45 * H);
    const soil = new THREE.Mesh(new THREE.CircleGeometry(G, 72), new THREE.MeshStandardMaterial({ roughness: 1 }));
    soil.rotation.x = -Math.PI / 2; soil.name = 'soil';
    const ring = new THREE.Mesh(new THREE.RingGeometry(G * .9, G * .915, 96), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = .05; ring.name = 'ring';
    const tri = new THREE.Shape(); tri.moveTo(0, G * .99); tri.lineTo(-G * .05, G * .82); tri.lineTo(G * .05, G * .82); tri.closePath();
    const north = new THREE.Mesh(new THREE.ShapeGeometry(tri), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
    north.rotation.x = -Math.PI / 2; north.position.y = .1; north.name = 'north';
    ground.add(soil, ring, north);
  }

  // cadrage : sphère englobant la machine et les flèches, vue de trois quarts avant côté flèches
  function frameCamera(d: RotorData, force = false) {
    const key = `${d.hubZ}|${d.R}|${camera.aspect.toFixed(3)}`;
    if (key === framed && !force) return;
    framed = key;
    const sphere = new THREE.Box3().setFromObject(machine).union(new THREE.Box3().setFromObject(wind)).getBoundingSphere(new THREE.Sphere());
    const half = THREE.MathUtils.degToRad(camera.fov / 2), halfH = Math.atan(Math.tan(half) * camera.aspect);
    const dist = sphere.radius / Math.sin(Math.min(half, halfH)) * 1.02;
    const dir = new THREE.Vector3(Math.sin(.75), .3, Math.cos(.75)).normalize();
    camera.position.copy(sphere.center).addScaledVector(dir, dist);
    camera.near = dist / 50; camera.far = dist * 10; camera.updateProjectionMatrix();
    controls.target.copy(sphere.center); controls.minDistance = dist * .45; controls.maxDistance = dist * 1.8;
    controls.update();
  }

  function setWind(d: RotorData) {
    const { R, profile } = d, hub = Math.hypot(d.u, d.v);
    const mx = Math.max(...profile.map(p => p.sp), .1), L = 1.3 * R, x = R * 1.25, z0 = L * .5, rad = .013 * R;
    profile.forEach((p, i) => {
      const a = arrowAt(i), len = L * p.sp / mx, head = Math.min(len * .35, rad * 7);
      a.g.visible = p.sp > .05;
      a.g.position.set(x, p.z, z0);
      a.shaft.scale.set(rad, Math.max(len - head, .001), rad);
      a.cone.position.y = len - head; a.cone.scale.set(rad * 2.6, head, rad * 2.6);
      a.mat.color.setRGB(p.rgb[0] / 255, p.rgb[1] / 255, p.rgb[2] / 255, THREE.SRGBColorSpace);
    });
    for (let i = profile.length; i < arrows.length; i++) arrows[i].g.visible = false;
    // repère pointillé : longueur qu'aurait la flèche au vent du moyeu ; puis bas des pales, moyeu,
    // haut des pales (les trois valeurs du panneau)
    const zs = profile.map(p => p.z), zlo = Math.min(...zs), zhi = Math.max(...zs), zr = z0 - L * hub / mx;
    lineMat.dashSize = R * .05; lineMat.gapSize = R * .04;
    setLine(lines[0], new THREE.Vector3(x, zlo, zr), new THREE.Vector3(x, zhi, zr));
    [zlo, d.hubZ, zhi].forEach((z, i) => setLine(lines[i + 1], new THREE.Vector3(R * .12, z, 0), new THREE.Vector3(x, z, z0)));
    // sol tourné pour que le nord soit à sa place : le rotor (+Z) fait face au vent
    // nord du sol en -Z au repos ; on tourne de (θ − π) pour amener la direction d'où vient le vent en +Z
    const from = Math.atan2(-d.u, -d.v);        // θ, 0 = nord, sens horaire (comme fromDeg)
    ground.rotation.y = from - Math.PI;
    omega = reduce.matches || hub < 3 || hub >= 25 ? 0 : Math.min(7 * hub / R, 2.4);
  }

  // couleurs du thème lues seulement quand le thème change (getComputedStyle force un recalcul de style)
  function recolor() {
    const key = `${document.documentElement.dataset.theme || ''}|${matchMedia('(prefers-color-scheme: dark)').matches}`;
    if (key === themeKey) return;
    themeKey = key;
    lineMat.color.set(css('--ink'));
    const soil = ground.getObjectByName('soil') as THREE.Mesh | undefined;
    (soil?.material as THREE.MeshStandardMaterial | undefined)?.color.set(css('--surface-2'));
    ((ground.getObjectByName('ring') as THREE.Mesh | undefined)?.material as THREE.MeshBasicMaterial | undefined)?.color.set(css('--line'));
    ((ground.getObjectByName('north') as THREE.Mesh | undefined)?.material as THREE.MeshBasicMaterial | undefined)?.color.set(css('--accent'));
  }

  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    if (current) frameCamera(current);
  }

  // rendu à la demande : 30 images/s tant que le rotor tourne, sinon seulement après un changement
  // (données, thème, taille) ou pendant que l'utilisateur fait pivoter la vue
  function tick(now: number) {
    frame = 0;
    if (pending) { const d = pending; pending = null; current = d; build(d); setWind(d); frameCamera(d); recolor(); dirty = true; }
    const moved = controls.update(), spinning = omega > 0;
    if (dirty || moved || (spinning && now - lastDraw >= 1000 / 31)) {
      const dt = Math.min((now - last) / 1000, .1); last = now;
      angle -= omega * dt;                    // sens horaire vu de face
      rotor.rotation.z = angle;
      renderer.render(scene, camera);
      lastDraw = now; dirty = false;
    }
    if (visible && !document.hidden && (spinning || moved || dirty)) frame = requestAnimationFrame(tick);
  }
  const wake = () => { if (!frame && visible && !document.hidden) { last = performance.now(); frame = requestAnimationFrame(tick); } };
  const redraw = () => { dirty = true; wake(); };

  new ResizeObserver(() => { resize(); redraw(); }).observe(host);
  new IntersectionObserver(es => { visible = es.some(e => e.isIntersecting); redraw(); }).observe(host);
  new MutationObserver(() => { recolor(); redraw(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  document.addEventListener('visibilitychange', redraw);
  controls.addEventListener('change', wake);
  resize();

  return {
    // plusieurs mises à jour dans la même image (curseur horaire) n'en appliquent qu'une
    update(d) { pending = d; dirty = true; wake(); },
  };
}
