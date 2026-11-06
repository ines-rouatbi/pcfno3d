// Use the same preference in the decision workspace, laboratory, and exports.
export function initializeTheme() {
  let theme = 'light';
  try { theme = JSON.parse(localStorage.getItem('aw-theme') || '"light"'); } catch { /* private browser */ }
  if (theme === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme === 'dark' ? 'dark' : 'light';
}
