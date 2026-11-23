import { memo } from 'react';
import { Clock3, Layers3, MapPinned, Sprout } from 'lucide-react';
import './styles/hero.css';

// Slogan et pastilles flottantes d'ATOM Wind, à droite de la phrase du bulletin (bulletin.tsx, Vue d'ensemble) :
// deux lignes de slogan, puis quatre pastilles qui flottent doucement. Aucun mot sur le modèle (décision du
// 22/11/2026) ; immobiles avec « mouvement réduit ».

const PASTILLES = [
  { icone: MapPinned, texte: '24 gouvernorats' },
  { icone: Layers3, texte: 'Du sol à 1 km' },
  { icone: Clock3, texte: 'Heure par heure, 3 jours' },
  { icone: Sprout, texte: 'Traiter, arroser, produire' },
];

export const SLOGAN = <>Le vent de toute la Tunisie, <em>du sol jusqu’au ciel.</em></>;

/** Les pastilles flottantes, dans la carte du bulletin (sous le slogan). */
export const PastillesFlottantes = memo(function PastillesFlottantes() {
  return <div className="aw-flottant" aria-label="Ce que propose ATOM Wind">
    <ul>{PASTILLES.map(({ icone: I, texte }, k) => <li key={texte} style={{ animationDelay: `${-k * 1.1}s` }}><I size={14}/>{texte}</li>)}</ul>
  </div>;
});
