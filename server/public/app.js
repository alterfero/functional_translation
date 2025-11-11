const d3select = d3.select;
const POS_CATS = ['Noun','Verb','Adjective','Adverb','Other'];

const els = {
  status: document.getElementById('status'),
  pairs: document.getElementById('pairs'),
  target: document.getElementById('target'),
  k: document.getElementById('k'),
  kVal: document.getElementById('kVal'),
  template: document.getElementById('template'),
  includeSeeds: document.getElementById('includeSeeds'),
  excludeInputs: document.getElementById('excludeInputs'),
  runBtn: document.getElementById('runBtn'),
  rebuildBtn: document.getElementById('rebuildBtn'),
  chart: document.getElementById('chart'),
  tableBody: document.querySelector('#neighborsTable tbody'),
  posChecks: Array.from(document.querySelectorAll('.pos')),
};

let lastResult = null;

function parsePairs(text) {
  const lines = text.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  return lines.map(line => {
    let parts = line.split('→');
    if (parts.length < 2) parts = line.split(',');
    const a = (parts[0] || '').trim();
    const b = (parts[1] || '').trim();
    return a && b ? [a, b] : null;
  }).filter(Boolean);
}

function classifyPOS(word) {
  const doc = window.nlp(word);
  if (doc.nouns().out('array').includes(word)) return 'Noun';
  if (doc.verbs().out('array').includes(word)) return 'Verb';
  if (doc.adjectives().out('array').includes(word)) return 'Adjective';
  if (doc.adverbs().out('array').includes(word)) return 'Adverb';
  return 'Other';
}

function posFilterActive() {
  const keeps = new Set(els.posChecks.filter(c => c.checked).map(c => c.value));
  return (pos) => keeps.has(pos);
}

async function initStatus() {
  const r = await fetch('/api/status').then(r => r.json());
  els.status.textContent = `Model: ${r.model} • dim=${r.dim || '…'} • vocab=${r.vocabSize || 0}`;
}

async function rebuildCache() {
  const contextTemplate = els.template.value.trim() || '{w}';
  els.rebuildBtn.disabled = true;
  els.status.textContent = 'Rebuilding cache… (first time takes a bit)';
  try {
    await fetch('/api/rebuild', { method: 'POST', headers: { 'Content-Type':'application/json' },
      body: JSON.stringify({ contextTemplate }) });
    await initStatus();
  } catch (e) {
    console.error(e);
    alert('Rebuild failed: ' + e);
  } finally {
    els.rebuildBtn.disabled = false;
  }
}

function renderNeighborsTable(neighbors) {
  const keep = posFilterActive();
  const rows = neighbors
    .map((n, i) => ({ ...n, pos: classifyPOS(n.word) }))
    .filter(n => keep(n.pos));

  els.tableBody.innerHTML = '';
  rows.forEach((n, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${i+1}</td><td>${n.word}</td><td>${n.score.toFixed(4)}</td><td><span class="badge">${n.pos}</span></td>`;
    els.tableBody.appendChild(tr);
  });
}

function drawChart(points) {
  const keep = posFilterActive();

  const w = els.chart.clientWidth - 20;
  const h = els.chart.clientHeight - 20;
  els.chart.innerHTML = '';

  const svg = d3select(els.chart).append('svg')
    .attr('width', w).attr('height', h)
    .style('border-radius', '10px');

  const margin = { top: 16, right: 16, bottom: 24, left: 24 };
  const innerW = w - margin.left - margin.right;
  const innerH = h - margin.top - margin.bottom;

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const x = d3.scaleLinear().domain(d3.extent(xs)).nice().range([0, innerW]);
  const y = d3.scaleLinear().domain(d3.extent(ys)).nice().range([innerH, 0]);

  const xAxis = g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(6));
  const yAxis = g.append('g').call(d3.axisLeft(y).ticks(6));

  const zoom = d3.zoom().scaleExtent([0.5, 10]).on('zoom', (event) => {
    const t = event.transform;
    const zx = t.rescaleX(x);
    const zy = t.rescaleY(y);
    pointsSel.attr('cx', d => zx(d.x)).attr('cy', d => zy(d.y));
    labels.attr('x', d => zx(d.x) + 8).attr('y', d => zy(d.y) + 4);
    arrow.attr('d', arrowPath(zx, zy));
    xAxis.call(d3.axisBottom(zx).ticks(6));
    yAxis.call(d3.axisLeft(zy).ticks(6));
  });
  svg.call(zoom);

  // Tooltip
  const tooltip = d3select(els.chart).append('div').attr('class','tooltip').style('opacity',0);

  function color(d) {
    switch (d.kind) {
      case 'neighbor': return getComputedStyle(document.documentElement).getPropertyValue('--neighbor');
      case 'target': return getComputedStyle(document.documentElement).getPropertyValue('--target');
      case 'predicted': return getComputedStyle(document.documentElement).getPropertyValue('--pred');
      case 'seedFrom': return getComputedStyle(document.documentElement).getPropertyValue('--seedFrom');
      case 'seedTo': return getComputedStyle(document.documentElement).getPropertyValue('--seedTo');
      default: return '#ccc';
    }
  }

  const filtered = points.filter(p => {
    if (p.kind !== 'neighbor') return true; // always keep special points
    const pos = classifyPOS(p.label);
    return keep(pos);
  });

  const pointsSel = g.selectAll('circle.point')
    .data(filtered)
    .enter()
    .append('circle')
    .attr('class','point')
    .attr('r', d => (d.kind === 'neighbor' ? 3.5 : 5.5))
    .attr('cx', d => x(d.x))
    .attr('cy', d => y(d.y))
    .attr('fill', d => color(d))
    .attr('opacity', d => (d.kind === 'neighbor' ? 0.85 : 1))
    .on('mousemove', (event, d) => {
      tooltip.style('opacity', 1)
        .style('left', (event.offsetX + 12) + 'px')
        .style('top',  (event.offsetY - 10) + 'px')
        .html(`<b>${d.label}</b><br><span class="badge">${d.kind}</span>`);
    })
    .on('mouseleave', () => tooltip.style('opacity', 0));

  const labels = g.selectAll('text.label')
    .data(filtered.filter(d => d.kind !== 'neighbor')) // label special points
    .enter().append('text')
    .attr('x', d => x(d.x) + 8)
    .attr('y', d => y(d.y) + 4)
    .attr('fill', '#d8dee9')
    .attr('font-size', 12)
    .text(d => d.label);

  // Arrow from target to predicted
  const target = points.find(p => p.kind === 'target');
  const pred = points.find(p => p.kind === 'predicted');
  const arrowPath = (sx, sy) => {
    if (!target || !pred) return '';
    return `M ${sx(target.x)} ${sy(target.y)} L ${sx(pred.x)} ${sy(pred.y)}`;
  };
  const marker = svg.append('defs').append('marker')
    .attr('id', 'arrowhead')
    .attr('viewBox', '0 0 10 10')
    .attr('refX', 10).attr('refY', 5)
    .attr('markerWidth', 6).attr('markerHeight', 6)
    .attr('orient', 'auto-start-reverse')
    .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z').attr('fill', getComputedStyle(document.documentElement).getPropertyValue('--pred'));

  const arrow = g.append('path')
    .attr('d', arrowPath(x, y))
    .attr('stroke', getComputedStyle(document.documentElement).getPropertyValue('--pred'))
    .attr('stroke-width', 2)
    .attr('fill', 'none')
    .attr('marker-end', 'url(#arrowhead)');
}

async function runSearch() {
  const pairs = parsePairs(els.pairs.value);
  if (!pairs.length) {
    alert('Please provide at least one valid pair.');
    return;
  }
  const body = {
    pairs,
    target: els.target.value.trim(),
    k: Number(els.k.value),
    contextTemplate: els.template.value.trim() || '{w}',
    includeSeeds: els.includeSeeds.checked,
    excludeInputs: els.excludeInputs.checked
  };
  els.runBtn.disabled = true;
  els.status.textContent = 'Running…';
  try {
    const r = await fetch('/api/search', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    const json = await r.json();
    if (json.error) throw new Error(json.error);
    lastResult = json;

    // POS filter reacts live
    renderNeighborsTable(json.neighbors);
    drawChart(json.points);

    const p1 = (json.explainedVariance[0] * 100).toFixed(1);
    const p2 = (json.explainedVariance[1] * 100).toFixed(1);
    els.status.textContent = `k=${json.meta.k} • vocab=${json.meta.vocabSize} • PCA var: PC1 ${p1}% / PC2 ${p2}%`;
  } catch (e) {
    console.error(e);
    alert('Search failed: ' + e.message);
    els.status.textContent = 'Error.';
  } finally {
    els.runBtn.disabled = false;
  }
}

// UI wiring
els.k.addEventListener('input', () => els.kVal.textContent = els.k.value);
els.runBtn.addEventListener('click', runSearch);
els.rebuildBtn.addEventListener('click', rebuildCache);
els.posChecks.forEach(cb => cb.addEventListener('change', () => {
  if (!lastResult) return;
  renderNeighborsTable(lastResult.neighbors);
  drawChart(lastResult.points);
}));

// boot
initStatus();
