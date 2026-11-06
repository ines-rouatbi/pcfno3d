// The retained instruments create their dialogs imperatively. Keep focus handling
// in one place so their keyboard behaviour matches the React shell.
export function installDialogFocus() {
  let lastExternal: HTMLElement | null = null;
  let previous: HTMLElement | null = null;
  const modal = () => document.querySelector<HTMLElement>('.tour:not([hidden]),.palette:not([hidden])');
  const focusable = (parent: HTMLElement) => Array.from(parent.querySelectorAll<HTMLElement>('button:not([disabled]),a[href],input:not([disabled]),select:not([disabled]),textarea,[tabindex="0"]')).filter(node => node.offsetParent !== null);
  document.addEventListener('focusin', event => {
    const target = event.target as HTMLElement;
    if (!target.closest('.tour,.palette,.sheet')) lastExternal = target;
  });
  document.addEventListener('keydown', event => {
    const active = modal();
    if (event.key === 'Escape') {
      if (active) active.hidden = true;
      const sheet = document.getElementById('sheet');
      if (sheet) sheet.hidden = true;
      document.querySelectorAll<HTMLDetailsElement>('.workspace-preferences[open],.scene-tools[open]').forEach(details => { details.open = false; });
    }
    if (!active || event.key !== 'Tab') return;
    const nodes = focusable(active), first = nodes[0], last = nodes.at(-1);
    if (event.shiftKey && (document.activeElement === first || !active.contains(document.activeElement))) {
      event.preventDefault(); last?.focus();
    } else if (!event.shiftKey && (document.activeElement === last || !active.contains(document.activeElement))) {
      event.preventDefault(); first?.focus();
    }
  });
  const observer = new MutationObserver(() => {
    const active = modal();
    if (active && active !== previous && !active.contains(document.activeElement)) focusable(active)[0]?.focus();
    if (!active && previous) lastExternal?.focus();
    previous = active;
  });
  for (const id of ['tour','authm','palette']) {
    const element = document.getElementById(id);
    if (element) observer.observe(element,{attributes:true,attributeFilter:['hidden']});
  }
}
