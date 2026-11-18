const copyButton = document.getElementById('copy');
const bibtex = document.getElementById('bibtex');
const copyStatus = document.getElementById('copy-status');
let resetLabel;
copyButton.hidden = false;
copyButton.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(bibtex.textContent);
    copyButton.textContent = 'Copied';
    copyStatus.textContent = 'BibTeX copied to clipboard.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(bibtex);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyButton.textContent = 'Selected';
    copyStatus.textContent = 'BibTeX selected. Use your device’s copy command.';
  }
  clearTimeout(resetLabel);
  resetLabel = setTimeout(() => {
    copyButton.textContent = 'Copy BibTeX';
    copyStatus.textContent = '';
  }, 3000);
});

const viewer = document.getElementById('viewer');
if (viewer && typeof viewer.showModal === 'function') {
  const title = document.getElementById('viewer-title');
  const image = document.getElementById('viewer-image');
  const stage = viewer.querySelector('.viewer-stage');
  const closeButton = document.getElementById('viewer-close');
  const zoomButton = document.getElementById('viewer-zoom');
  let trigger;

  function setZoom(zoomed) {
    const width = zoomed ? Math.max(image.naturalWidth, Math.ceil(image.clientWidth * 1.6)) : 0;
    stage.classList.toggle('zoomed', zoomed);
    image.style.width = zoomed ? width + 'px' : '';
    zoomButton.textContent = zoomed ? 'Fit image' : 'Zoom in';
    stage.scrollTo(0, 0);
  }

  for (const link of document.querySelectorAll('a[data-view]')) {
    link.setAttribute('aria-haspopup', 'dialog');
    link.addEventListener('click', event => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      trigger = link;
      title.textContent = link.dataset.view;
      image.alt = link.closest('figure').querySelector('img').alt;
      image.src = link.href;
      setZoom(false);
      viewer.showModal();
      closeButton.focus();
    });
  }
  zoomButton.addEventListener('click', () => setZoom(!stage.classList.contains('zoomed')));
  closeButton.addEventListener('click', () => viewer.close());
  viewer.addEventListener('click', event => {
    if (event.target !== viewer) return;
    const bounds = viewer.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right ||
        event.clientY < bounds.top || event.clientY > bounds.bottom) viewer.close();
  });
  viewer.addEventListener('close', () => trigger?.focus({ preventScroll: true }));
}
