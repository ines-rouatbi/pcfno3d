type Point = [number, number];
type Place = { name: string; point: Point; speed: string; selected: boolean };
type Territory = {
  water: Point[][];
  lakes: number[][];
  hills: number[][];
  places: Place[];
  selected: Point;
  sample: (x: number, y: number) => Point;
  select: (index: number) => void;
  /** Emprise du domaine en km (défaut : Grand Tunis, 68 × 56 km), nom affiché et étiquette de mer. */
  width?: number;
  height?: number;
  zone?: string;
  seaLabel?: string | null;
};

const safe = (value: string) => value.replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]!));

/** The existing domain projection and wind sampler; a static, keyboard-accessible 2D view. */
export function renderTerritory(container: HTMLElement, data: Territory) {
  // cadre de 421,6 × 347,2 px : le Grand Tunis (68 × 56 km) y tient a 6,2 px/km ; un autre domaine
  // (national, 388 × 828 km) y est centre a la plus grande echelle qui tient
  const width = data.width ?? 68, height = data.height ?? 56, zone = data.zone ?? 'le Grand Tunis';
  const seaLabel = data.seaLabel === undefined ? 'Golfe de Tunis' : data.seaLabel;
  const scale = Math.min(421.6/width, 347.2/height), left = 97 + (421.6 - width*scale)/2, top = 22 + (347.2 - height*scale)/2;
  const project = ([x,y]: Point) => [left + x*scale, top + (height-y)*scale];
  const points = (polygon: Point[]) => polygon.map(p => project(p).join(',')).join(' ');
  const step = 43.4/scale, gridKm = width > 150 ? 100 : 10;      // fleches tous les 43,4 px (7 km au Grand Tunis)
  const arrows: string[] = [];
  for (let y=step*5/7; y<height; y+=step) for (let x=step*4/7; x<width; x+=step) {
    const [u,v] = data.sample(x,y), speed = Math.hypot(u,v), [px,py] = project([x,y]);
    const angle = Math.atan2(u,v)*180/Math.PI;
    arrows.push(`<g transform="translate(${px},${py}) rotate(${angle})" opacity="${Math.min(.7,.25+speed/22)}"><path d="M0 9V-9M-3-5L0-9L3-5" fill="none" stroke="var(--map-vector)" stroke-width="1.25" stroke-linecap="round"/></g>`);
  }
  const labels = new Set([0,1,2,5,6,7,8,10,11]);
  const markers = data.places.map((place,index) => {
    const [x,y] = project(place.point), name = place.name.split(' (')[0], selected = place.selected;
    const right = x>left + width*scale*343/421.6, anchor = right?'end':'start', offset=right?-11:11;
    return `<g class="territory-place${selected?' selected':''}" data-place="${index}" role="button" tabindex="0" aria-label="${safe(place.name)}, ${safe(place.speed)}" aria-pressed="${selected}">
      <title>${safe(place.name)} · ${safe(place.speed)}</title><circle cx="${x}" cy="${y}" r="15" fill="transparent"/>
      ${selected?`<circle cx="${x}" cy="${y}" r="11" fill="var(--map-selection)" opacity=".25"/>`:''}
      <circle cx="${x}" cy="${y}" r="${selected?5:3}" fill="${selected?'var(--accent)':'var(--map-vector)'}" stroke="var(--map-land)" stroke-width="2"/>
      ${labels.has(index)||selected?`<text x="${x+offset}" y="${y+4}" text-anchor="${anchor}" class="${selected?'selected-label':''}">${safe(name)}</text>`:''}</g>`;
  }).join('');
  const selectedKnown = data.places.some(p=>p.selected), [sx,sy]=project(data.selected);
  const lines = (extent: number) => Array.from({length: Math.max(0, Math.ceil(extent/gridKm) - 1)}, (_, i) => (i+1)*gridKm);
  container.innerHTML = `<svg viewBox="70 0 490 400" aria-label="Vent à 10 mètres du sol dans ${safe(zone)}. Choisissez un lieu pour lire ses conditions." class="territory-svg">
    <defs><clipPath id="territory-clip"><rect x="${left}" y="${top}" width="${width*scale}" height="${height*scale}" rx="3"/></clipPath></defs>
    <rect width="620" height="400" fill="var(--map-land)"/>
    <g clip-path="url(#territory-clip)">
      ${lines(width).map(x=>`<path d="M${project([x,0])[0]},${top}V${top+height*scale}" stroke="var(--map-grid)" stroke-width=".6"/>`).join('')}
      ${lines(height).map(y=>`<path d="M${left},${project([0,y])[1]}H${left+width*scale}" stroke="var(--map-grid)" stroke-width=".6"/>`).join('')}
      ${data.hills.map(([x,y,height,radius])=>[1,1.65,2.3].map((factor,i)=>{const [cx,cy]=project([x,y]);return `<ellipse cx="${cx}" cy="${cy}" rx="${radius*scale*factor}" ry="${radius*scale*factor*.78}" fill="${i===0?'var(--map-relief)':'none'}" fill-opacity=".35" stroke="var(--map-contour)" stroke-width=".7"/>`;}).join('')).join('')}
      ${data.water.map(p=>`<polygon points="${points(p)}" fill="var(--map-water)" stroke="var(--map-coast)" stroke-width="1.2"/>`).join('')}
      ${data.lakes.map(([x,y,rx,ry])=>{const [cx,cy]=project([x,y]);return `<ellipse cx="${cx}" cy="${cy}" rx="${rx*scale}" ry="${ry*scale}" fill="var(--map-water)" stroke="var(--map-coast)" stroke-width=".6"/>`;}).join('')}
      ${arrows.join('')}
    </g>
    ${seaLabel ? `<text class="map-sea-label" x="357" y="75" text-anchor="middle">${safe(seaLabel)}</text>` : ''}
    ${markers}
    ${selectedKnown?'':`<circle cx="${sx}" cy="${sy}" r="6" fill="var(--accent)" stroke="var(--map-land)" stroke-width="3"/>`}
    <g class="map-north" transform="translate(540 35)"><path d="M0 28V4M-4 11L0 4L4 11" fill="none" stroke="currentColor"/><text y="-4" text-anchor="middle">N</text></g>
    <g class="map-scale" transform="translate(${498 - gridKm*scale} 387)"><path d="M0-5V0H${gridKm*scale}V-5" fill="none" stroke="currentColor"/><text x="${gridKm*scale/2}" y="-10" text-anchor="middle">${gridKm} km</text></g>
  </svg>`;
  container.querySelectorAll<SVGGElement>('[data-place]').forEach(node=>{
    const activate = () => {
      const keyboard = node.matches(':focus-visible');
      data.select(Number(node.dataset.place));
      if(keyboard) container.querySelector<SVGGElement>(`[data-place="${node.dataset.place}"]`)?.focus();
    };
    node.onclick = activate;
    node.onkeydown = event => { if(event.key==='Enter'||event.key===' '){event.preventDefault();activate();} };
  });
}
