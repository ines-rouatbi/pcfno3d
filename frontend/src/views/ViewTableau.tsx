import { memo } from 'react';
import { BulletinDuJour } from '../bulletin';
import { ArrowRight, ArrowUpRight, ChartNoAxesCombined, Fingerprint, MapPin, ShieldCheck, ChevronDown, Leaf, Wind, ScanLine, MoveUpRight } from 'lucide-react';

export default memo(function ViewTableau() {
  return <section className="view overview" id="v-tableau">
    <BulletinDuJour/>
    <div className="territory-workbench">
      <section className="territory-panel" aria-label="Explorer le territoire">
        <header className="territory-heading"><div><span className="section-index">TERRITOIRE</span><h2>Le vent, d’un lieu à l’autre</h2></div><div id="tbhero" className="observatory-reading"/></header>
        <div className="territory-map-area"><div className="territory-locations"><span>Lieux repères</span><div id="tblocations"/></div><div className="territory-map" id="tbmap"/></div>
        <div className="territory-map-footer"><span><MoveUpRight size={14}/>Les flèches indiquent où va le vent</span><span>Relief réel du Grand Tunis · vent à 10 m du sol</span></div>
      </section>
      <aside className="point-workbench" aria-label="Conditions au lieu sélectionné">
        <label htmlFor="tb-place"><MapPin size={15}/>Votre point de lecture</label><select id="tb-place" aria-label="Lieu à inspecter"/>
        <div id="tbpoint"/>
        <a className="point-primary" href="#monvent">Que puis-je faire ici ?<ArrowRight size={17}/></a>
        <button className="point-profile" id="tb-profile"><ScanLine size={15}/>Voir le profil vertical<ArrowUpRight size={14}/></button>
      </aside>
    </div>
    <section className="activity-paths" aria-label="Passer des conditions à votre activité">
      <div className="activity-paths-heading"><span className="section-index">VOS ACTIVITÉS</span><h2>Préparer la suite</h2><p>Des outils pour chaque usage du vent.</p></div>
      <a className="activity-path agriculture-path" href="#agri" data-context="agri" data-access-view="agri"><Leaf size={21}/><div><h3>Agriculture</h3><p>Quand traiter, quand arroser.</p><span>Préparer ma parcelle<ArrowRight size={15}/></span></div></a>
      <a className="activity-path energy-path" href="#eolien" data-access-view="eolien"><Wind size={23}/><div><h3>Énergie éolienne</h3><p>Production et choix d’un site.</p><span>Étudier le potentiel<ArrowRight size={15}/></span></div></a>
      <a className="activity-path evidence-path" href="#fiabilite" data-access-view="fiabilite"><ShieldCheck size={21}/><div><h3>Données & fiabilité</h3><p>Ce que les chiffres permettent de conclure.</p><span>Vérifier les sources<ArrowRight size={15}/></span></div></a>
    </section>
    <div className="overview-layout">
      <section className="day-chart"><div className="observatory-chart-heading"><div><span className="section-index">24 HEURES</span><h2>Comment le vent évolue</h2></div><span>Moyenne du territoire · 10 m</span></div><div id="tbday" className="observatory-chart"/><div className="observatory-footnote"><Fingerprint size={14}/><span id="tb-source-note">Calcul du modèle, avec interpolation entre les instants disponibles.</span></div></section>
      <aside className="operational-briefing"><div className="briefing-heading"><h2>À surveiller</h2><a href="#alertes" aria-label="Ouvrir toutes les alertes"><ArrowUpRight size={19}/></a></div><p className="briefing-intro">Les lieux et les activités concernés à l’heure choisie.</p><div id="tbalerts" className="mine briefing-list"/><a href="#alertes" className="briefing-link">Gérer mes alertes<ArrowRight size={16}/></a></aside>
    </div>
    <details className="territory-details"><summary><div><h2>Les repères du territoire</h2><span>Production du parc virtuel, couverture et points de mesure</span></div><ChevronDown size={20}/></summary><section className="overview-metrics" id="tbkpis" aria-label="Les autres repères du territoire"/><section className="network-strip"><div className="network-title"><div><h3>Les points de mesure</h3><p>Les stations réelles et le projet de réseau ANEO, présentés séparément.</p></div><a href="#fiabilite">Comprendre la couverture<ArrowUpRight size={16}/></a></div><div className="rel3" id="tbnet"/></section></details>
    <details className="validation-disclosure"><summary><div><h2>Le modèle et ses limites</h2><span>Comparaison aux mesures et objectifs de qualité DSO</span></div><ChevronDown size={22}/></summary>
    <div className="validation-grid">
      <section className="validation-comparison"><div className="validation-title"><ChartNoAxesCombined size={18}/><h3>Le modèle face aux mesures</h3><span className="archive-label">VALIDATION</span></div><p>Erreur du vent sur des jours jamais vus à l’entraînement. Une barre plus courte indique un écart plus faible.</p><div id="tbcomp"/><a className="text-link" href="#fiabilite">Lire les limites de cette comparaison<ArrowUpRight size={15}/></a></section>
      <section className="card model-instrument"><div className="validation-title"><ShieldCheck size={18}/><h3>Qualité du modèle servi</h3></div><div className="dsos"><p className="muted" style={{margin: 0}}>Chargement de l’évaluation du modèle servi…</p></div></section>
    </div>
    </details>
  </section>;
});
