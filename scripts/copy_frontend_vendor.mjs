import { mkdir,copyFile,readdir } from 'node:fs/promises';
const source='node_modules/leaflet',target='frontend/public/vendor/leaflet';
await mkdir(`${target}/images`,{recursive:true});
await copyFile(`${source}/dist/leaflet.js`,`${target}/leaflet.js`);
await copyFile(`${source}/LICENSE`,`${target}/LICENSE`);
for(const name of await readdir(`${source}/dist/images`))await copyFile(`${source}/dist/images/${name}`,`${target}/images/${name}`);
