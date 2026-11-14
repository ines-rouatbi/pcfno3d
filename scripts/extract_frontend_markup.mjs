// Render the actual React components to HTML for the existing translation lexer.
import { createServer } from 'vite';
import { createElement, Fragment } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { writeFile } from 'node:fs/promises';
const server = await createServer({ cacheDir:'node_modules/.vite-markup',optimizeDeps:{noDiscovery:true,include:[]},server: { middlewareMode: true }, appType: 'custom' });
try {
  const ui = await server.ssrLoadModule('/src/components.tsx');
  const { default: Views } = await server.ssrLoadModule('/src/views/InstrumentViews.tsx');
  const panel = await server.ssrLoadModule('/src/assistant/AssistantPanel.tsx');
  const state = {ready:true,view:'monvent',hour:15,location:'Grand Tunis',speed:'—',direction:'—',altitude:10,source:'demonstration'};
  const nodes = [createElement(ui.Navigation,{open:false,close:()=>{}}),createElement(ui.Topbar,{onMenu:()=>{}}),createElement(ui.DataContext,{state}),createElement(ui.WindReadout,{state}),createElement(Views),createElement(ui.Assistant,{state}),createElement(panel.AssistantPanel,{state,open:true,onClose:()=>{}})];
  for(const view of ['monvent','agri','tableau','carte3d','eolien','fiabilite','alertes','admin']) nodes.push(createElement(ui.PageHeading,{state:{...state,view}}));
  await writeFile('frontend/i18n-source.html',renderToStaticMarkup(createElement(Fragment,null,...nodes)), 'utf8');
} finally { await server.close(); }
