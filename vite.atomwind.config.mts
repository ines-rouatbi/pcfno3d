// Front-end local de la branche, branché sur le service réel par le Worker « atomwind » d'Ines (qui transmet la
// session X-ATOM-Session) ; aucune clé ni session dans ce fichier.   npx vite --config vite.atomwind.config.mts --port 5182
import base from './vite.config.ts';
const passerelle = {
  target: 'https://atomwind.tukhnanutha.workers.dev', changeOrigin: true, secure: true,
  configure: (proxy: any) => proxy.on('proxyReq', (req: any) => { req.removeHeader('origin'); req.removeHeader('referer'); }),
};
export default { ...base, server: { ...(base as any).server, proxy: { '/api': passerelle } } };
