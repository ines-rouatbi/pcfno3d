import { memo } from 'react';
import { ContenuGouvernorats } from './ViewGouvernorats';
import { ArrowRight, ArrowUpRight, ChartNoAxesCombined, Fingerprint, MapPin, ShieldCheck, ChevronDown, Leaf, Wind } from 'lucide-react';

export default memo(function ViewTableau() {
  return <section className="view overview" id="v-tableau">
    <section className="overview-gouvernorats" aria-label="Le vent des 24 gouvernorats"><ContenuGouvernorats vue="tableau"/></section>
    <section className="activity-paths" aria-label="Passer des conditions à votre activité">
      <div className="activity-paths-heading"><span className="section-index">VOS ACTIVITÉS</span><h2>Préparer la suite</h2><p>Des outils pour chaque usage du vent.</p></div>
      <a className="activity-path agriculture-path" href="#agri" data-context="agri" data-access-view="agri"><Leaf size={21}/><div><h3>Agriculture</h3><p>Quand traiter, quand arroser.</p><span>Préparer ma parcelle<ArrowRight size={15}/></span></div></a>
      <a className="activity-path energy-path" href="#eolien" data-access-view="eolien"><Wind size={23}/><div><h3>Énergie éolienne</h3><p>Production et choix d’un site.</p><span>Étudier le potentiel<ArrowRight size={15}/></span></div></a>
    </section>
    <div className="overview-layout">
      <section className="day-chart"><div className="observatory-chart-heading"><div><span className="section-index">24 HEURES</span><h2>Comment le vent évolue</h2></div><span>Moyenne du territoire · 10 m</span></div><div id="tbday" className="observatory-chart"/><div className="observatory-footnote"><Fingerprint size={14}/><span id="tb-source-note">Calcul ATOM Wind, avec interpolation entre les instants disponibles.</span></div></section>
      <aside className="operational-briefing"><div className="briefing-heading"><h2>À surveiller</h2><a href="#alertes" aria-label="Ouvrir toutes les alertes"><ArrowUpRight size={19}/></a></div><p className="briefing-intro">Les lieux et les activités concernés à l’heure choisie.</p><div id="tbalerts" className="mine briefing-list"/><a href="#alertes" className="briefing-link">Gérer mes alertes<ArrowRight size={16}/></a></aside>
    </div>
    <details className="territory-details"><summary><div><h2>Les repères du territoire</h2><span>Production du parc virtuel, couverture et points de mesure</span></div><ChevronDown size={20}/></summary><section className="overview-metrics" id="tbkpis" aria-label="Les autres repères du territoire"/><section className="network-strip"><div className="network-title"><div><h3>Les points de mesure</h3><p>Les stations réelles et le projet de réseau ANEO, présentés séparément.</p></div></div><div className="rel3" id="tbnet"/></section></details>
  </section>;
});
