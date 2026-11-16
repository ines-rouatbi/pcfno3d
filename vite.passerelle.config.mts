// Front-end local de la branche, branché sur les VRAIS modèles : l'API passe par la passerelle publique
// atom.zelqane.com (le Worker y ajoute sa propre clé de lecture : aucune clé ni session dans ce fichier).
// Les comptes du service (connexion, rôles) sont ceux du service en ligne.
//     npm run dev:passerelle      ->  http://127.0.0.1:5181/app/atom-wind/   (?domaine=national : Tunisie entière)
import base from './vite.config.ts';

const passerelle = {
  target: 'https://atom.zelqane.com', changeOrigin: true, secure: true,
  configure: (proxy: any) => proxy.on('proxyReq', (req: any) => {
    req.removeHeader('origin'); req.removeHeader('referer');
  }),
};
export default { ...base, server: { ...(base as any).server, proxy: { '/api': passerelle } } };
