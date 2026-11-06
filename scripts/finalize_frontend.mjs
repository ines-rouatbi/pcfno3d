import { readFile, writeFile, readdir, unlink, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
const exportHtml=await readFile('webapp/export/index.html','utf8');
await mkdir('web',{recursive:true});
await writeFile('web/index.html',exportHtml.replaceAll('/app/assets/','../webapp/assets/'));
// Remove stale hashed bundles only inside the verified generated assets directory.
const assets=resolve('webapp/assets');
if(!assets.startsWith(resolve('webapp')+'/')&&!assets.startsWith(resolve('webapp')+'\\')) throw new Error('Invalid asset target');
const referenced=new Set();
async function visit(name) {
  if(referenced.has(name))return;
  referenced.add(name);
  if(!name.endsWith('.js')&&!name.endsWith('.css'))return;
  const text=await readFile(`${assets}/${name}`,'utf8');
  for(const match of text.matchAll(/(?:\.\/|assets\/)([^/"'()\s]+\.(?:js|css|woff2?|png))/g))await visit(match[1]);
}
for(const entry of ['webapp/index.html','webapp/atom-wind/index.html','webapp/export/index.html']){
  const html=await readFile(entry,'utf8');
  for(const match of html.matchAll(/\/app\/assets\/([^"']+)/g))await visit(match[1]);
}
for(const name of await readdir(assets)) if(!referenced.has(name))await unlink(`${assets}/${name}`);
