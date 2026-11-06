import { chromium } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
const browser=await chromium.launch({channel:'chrome',headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1050}});
await page.addInitScript(()=>{
  localStorage.setItem('aw-tour','1');localStorage.setItem('aw-theme',JSON.stringify('light'));
  window.__workspaceMetrics={cls:0,lcp:0,ready:0};
  new PerformanceObserver(list=>{for(const entry of list.getEntries())if(!entry.hadRecentInput)window.__workspaceMetrics.cls+=entry.value;}).observe({type:'layout-shift',buffered:true});
  new PerformanceObserver(list=>{window.__workspaceMetrics.lcp=list.getEntries().at(-1).startTime;}).observe({type:'largest-contentful-paint',buffered:true});
  addEventListener('atom:state',()=>{window.__workspaceMetrics.ready ||= performance.now();});
});
await page.goto('http://127.0.0.1:8000/app/atom-wind/');await page.waitForFunction(()=>window.ATOM?.workspace);await page.evaluate(()=>document.fonts.ready);await page.waitForTimeout(700);
const startup=await page.evaluate(()=>({...window.__workspaceMetrics}));
await page.locator('#b-login').click();await page.locator('[data-demo="admin"]').click();await page.locator('a[data-v="tableau"]').click();
const updates=[];
for(const hour of [3,6,9,12,15,18,21]) updates.push(await page.evaluate(async value=>{
  const start=performance.now();const slider=document.getElementById('hour');slider.value=value;slider.dispatchEvent(new Event('input',{bubbles:true}));await new Promise(requestAnimationFrame);return performance.now()-start;
},hour));
const sorted=updates.slice().sort((a,b)=>a-b);
const report={environment:'Chrome headless, Windows, 1440×1050, local HTTP, synthetic demonstration',startup,updateToAnimationFrameMs:updates,medianUpdateMs:sorted[Math.floor(sorted.length/2)],maxUpdateMs:Math.max(...updates),threeLoadedOnDashboard:await page.evaluate(()=>performance.getEntriesByType('resource').some(entry=>entry.name.includes('/atmosphere-')))};
await writeFile('output/redesign/dashboard-performance.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));await browser.close();
