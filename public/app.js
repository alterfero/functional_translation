import * as THREE from 'three';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';

const KIND_COLOR_VARS = {
  neighbor: '--neighbor',
  target: '--target',
  predicted: '--pred',
  seedFrom: '--seedFrom',
  seedTo: '--seedTo'
};
const KIND_LABELS = {
  neighbor: 'Neighbor',
  target: 'Target',
  predicted: 'Predicted',
  seedFrom: 'Seed (from)',
  seedTo: 'Seed (to)'
};

const POS_LABELS = {
  noun: 'Noun',
  verb: 'Verb',
  adjective: 'Adjective',
  adverb: 'Adverb',
  other: 'Other',
};

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
let threeCtx = null;
let statusRetryTimer = null;

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

function normalizePOS(pos) {
  if (!pos) return 'other';
  const lower = String(pos).toLowerCase();
  if (POS_LABELS[lower]) return lower;
  if (lower === 'adj') return 'adjective';
  if (lower === 'adv') return 'adverb';
  if (lower === 'n') return 'noun';
  if (lower === 'v') return 'verb';
  return 'other';
}

function formatPOS(pos) {
  const norm = normalizePOS(pos);
  return POS_LABELS[norm] || POS_LABELS.other;
}

function posFilterActive() {
  const keeps = new Set(els.posChecks.filter(c => c.checked).map(c => c.value));
  return pos => keeps.has(pos);
}

function getCssVar(name) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name);
  return value && value.trim() ? value.trim() : '#ffffff';
}

function ensureThreeContext() {
  if (threeCtx) {
    threeCtx.resize();
    return threeCtx;
  }

  const container = els.chart;
  container.innerHTML = '';

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(getCssVar('--surface'));

  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  camera.position.set(0, 0, 6);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.enablePan = false;
  controls.minDistance = 2.4;
  controls.maxDistance = 12;

  const ambient = new THREE.AmbientLight(0xffffff, 0.45);
  const keyLight = new THREE.DirectionalLight(0xffffff, 0.85);
  keyLight.position.set(3, 4, 5);
  const fill = new THREE.PointLight(0x60a5fa, 0.35, 12);
  fill.position.set(-4, -3, -5);
  scene.add(ambient, keyLight, fill);

  const group = new THREE.Group();
  scene.add(group);

  const frameGeometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(4, 4, 4));
  const frameMaterial = new THREE.LineBasicMaterial({ color: new THREE.Color(getCssVar('--frame')) });
  const frame = new THREE.LineSegments(frameGeometry, frameMaterial);
  scene.add(frame);

  const tooltip = document.createElement('div');
  tooltip.className = 'gl-tooltip';
  container.appendChild(tooltip);

  const emptyState = document.createElement('div');
  emptyState.className = 'chart-empty';
  emptyState.textContent = 'Run the translation to populate the 3D projection.';
  container.appendChild(emptyState);

  const ctx = {
    container,
    renderer,
    scene,
    camera,
    controls,
    group,
    frame,
    tooltip,
    emptyState,
    raycaster: new THREE.Raycaster(),
    pointer: new THREE.Vector2(),
    pickables: [],
    hovered: null,
    autoRotate: true,
  };

  function resize() {
    const rect = container.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    renderer.setSize(rect.width, rect.height, false);
    camera.aspect = rect.width / rect.height;
    camera.updateProjectionMatrix();
  }

  ctx.resize = resize;
  resize();
  window.addEventListener('resize', resize);

  controls.addEventListener('start', () => {
    ctx.autoRotate = false;
  });
  controls.addEventListener('end', () => {
    setTimeout(() => { ctx.autoRotate = true; }, 1200);
  });

  function animate() {
    requestAnimationFrame(animate);
    if (ctx.autoRotate) {
      ctx.group.rotation.y += 0.0025;
    }
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  function handlePointer(event) {
    if (!ctx.pickables.length) {
      tooltip.style.opacity = 0;
      return;
    }

    const rect = renderer.domElement.getBoundingClientRect();
    ctx.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    ctx.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    ctx.raycaster.setFromCamera(ctx.pointer, camera);
    const intersects = ctx.raycaster.intersectObjects(ctx.pickables, false);

    if (intersects.length) {
      const obj = intersects[0].object;
      if (ctx.hovered && ctx.hovered !== obj) {
        ctx.hovered.scale.setScalar(ctx.hovered.userData.baseScale || 1);
      }
      ctx.hovered = obj;
      const baseScale = obj.userData.baseScale || obj.scale.x;
      obj.userData.baseScale = baseScale;
      obj.scale.setScalar(baseScale * 10);

      const { label, kind, score } = obj.userData;
      const kindLabel = KIND_LABELS[kind] || kind;
      let html = `<strong>${label}</strong><br/><span class="badge">${kindLabel}</span>`;
      if (obj.userData.pos) {
        html += `<div>POS: ${obj.userData.pos}</div>`;
      }
      if (typeof score === 'number') {
        html += `<div>cosine: ${score.toFixed(4)}</div>`;
      }
      tooltip.innerHTML = html;

      const cx = event.clientX - rect.left;
      const cy = event.clientY - rect.top;
      tooltip.style.left = `${cx}px`;
      tooltip.style.top = `${cy}px`;
      tooltip.style.opacity = 1;
    } else {
      if (ctx.hovered) {
        ctx.hovered.scale.setScalar(ctx.hovered.userData.baseScale || 1);
        ctx.hovered = null;
      }
      tooltip.style.opacity = 0;
    }
  }

  renderer.domElement.addEventListener('pointermove', handlePointer);
  renderer.domElement.addEventListener('pointerleave', () => {
    if (ctx.hovered) {
      ctx.hovered.scale.setScalar(ctx.hovered.userData.baseScale || 1);
      ctx.hovered = null;
    }
    tooltip.style.opacity = 0;
  });

  threeCtx = ctx;
  return ctx;
}

function drawChart(result) {
  const ctx = ensureThreeContext();
  ctx.resize();

  const keep = posFilterActive();
  const filteredPoints = result.points.filter(p => {
    if (p.kind !== 'neighbor') return true;
    const pos = normalizePOS(p.pos);
    return keep(pos);
  });

  ctx.tooltip.style.opacity = 0;
  ctx.hovered = null;

  while (ctx.group.children.length) ctx.group.remove(ctx.group.children[0]);
  ctx.group.rotation.set(-0.35, 0, 0);
  ctx.pickables = [];

  if (ctx.scene.background instanceof THREE.Color) {
    ctx.scene.background.set(getCssVar('--surface'));
  } else {
    ctx.scene.background = new THREE.Color(getCssVar('--surface'));
  }
  ctx.frame.material.color.set(getCssVar('--frame'));

  if (!filteredPoints.length) {
    ctx.emptyState.style.opacity = 1;
    return;
  }
  ctx.emptyState.style.opacity = 0;

  const xs = filteredPoints.map(p => p.x);
  const ys = filteredPoints.map(p => p.y);
  const zs = filteredPoints.map(p => p.z || 0);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const minZ = Math.min(...zs);
  const maxZ = Math.max(...zs);

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const centerZ = (minZ + maxZ) / 2;
  const rangeX = maxX - minX;
  const rangeY = maxY - minY;
  const rangeZ = maxZ - minZ;
  const maxRange = Math.max(rangeX, rangeY, rangeZ) || 1;
  const scale = 2.4;

  const smallGeom = new THREE.SphereGeometry(0.055, 24, 24);
  const mediumGeom = new THREE.SphereGeometry(0.075, 28, 28);
  const largeGeom = new THREE.SphereGeometry(0.095, 32, 32);

  function createLabelSprite(text) {
    const canvas = document.createElement('canvas');
    const padding = 5;
    const fontSize = 36;
    const font = `${fontSize}px "Inter", "Segoe UI", sans-serif`;
    const ctx2d = canvas.getContext('2d');
    ctx2d.font = font;
    const textWidth = Math.ceil(ctx2d.measureText(text).width);
    const width = textWidth + padding * 2;
    const height = fontSize + padding * 1;
    canvas.width = width;
    canvas.height = Math.ceil(height);

    const ctxFinal = canvas.getContext('2d');
    ctxFinal.font = font;
    ctxFinal.textBaseline = 'middle';
    ctxFinal.textAlign = 'left';
    ctxFinal.fillStyle = 'rgba(15, 23, 42, 0.2)';
    ctxFinal.fillRect(0, 0, canvas.width, canvas.height);
    ctxFinal.fillStyle = '#f8fafc';
    ctxFinal.fillText(text, padding, canvas.height / 3);

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
      depthWrite: false
    });
    const sprite = new THREE.Sprite(material);
    const maxDim = Math.max(canvas.width, canvas.height) || 1;
    const labelScale = 0.6;
    sprite.scale.set(
      (canvas.width / maxDim) * labelScale,
      (canvas.height / maxDim) * labelScale,
      1
    );
    return sprite;
  }

  const idToObj = new Map();
  const pickables = [];

  filteredPoints.forEach(p => {
    const colorVar = KIND_COLOR_VARS[p.kind] || '--neighbor';
    const color = new THREE.Color(getCssVar(colorVar));
    const geometry = p.kind === 'neighbor' ? smallGeom : (p.kind === 'predicted' ? largeGeom : mediumGeom);
    const material = new THREE.MeshStandardMaterial({
      color,
      emissive: color.clone().multiplyScalar(0.25),
      metalness: 0.2,
      roughness: 0.35
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(
      ((p.x - centerX) / maxRange) * scale,
      ((p.y - centerY) / maxRange) * scale,
      ((p.z - centerZ) / maxRange) * scale
    );

    const baseScale = p.kind === 'neighbor' ? 0.7 : (p.kind === 'predicted' ? 1.1 : 0.9);
    mesh.scale.setScalar(baseScale);
    mesh.userData = {
      label: p.label,
      kind: p.kind,
      pos: p.kind === 'neighbor' ? formatPOS(p.pos) : null,
      score: typeof p.score === 'number' ? p.score : null,
      baseScale
    };

    ctx.group.add(mesh);
    pickables.push(mesh);
    idToObj.set(p.id, { mesh, position: mesh.position.clone() });

    const label = createLabelSprite(p.label);
    label.position.copy(mesh.position).add(new THREE.Vector3(baseScale * 0.28, baseScale * 0.12, 0));
    ctx.group.add(label);
  });

  ctx.pickables = pickables;

  const seedColor = new THREE.Color(getCssVar('--seedVector'));
  (result.seedLinks || []).forEach(link => {
    const from = idToObj.get(link.fromId);
    const to = idToObj.get(link.toId);
    if (!from || !to) return;
    const dir = new THREE.Vector3().subVectors(to.position, from.position);
    const length = dir.length();
    if (!length) return;
    const arrow = new THREE.ArrowHelper(dir.clone().normalize(), from.position, length, seedColor.getHex(), 0.14, 0.1);
    arrow.line.material.transparent = true;
    arrow.line.material.opacity = 0.75;
    arrow.cone.material.transparent = true;
    arrow.cone.material.opacity = 0.85;
    ctx.group.add(arrow);
  });

  const target = idToObj.get('target');
  const predicted = idToObj.get('predicted');
  if (target && predicted) {
    const arrowColor = new THREE.Color(getCssVar('--pred'));
    const dir = new THREE.Vector3().subVectors(predicted.position, target.position);
    const length = dir.length();
    if (length) {
      const arrow = new THREE.ArrowHelper(dir.clone().normalize(), target.position, length, arrowColor.getHex(), 0.2, 0.14);
      arrow.line.material.transparent = true;
      arrow.line.material.opacity = 0.95;
      arrow.cone.material.transparent = true;
      arrow.cone.material.opacity = 0.95;
      ctx.group.add(arrow);
    }
  }
}

function renderNeighborsTable(neighbors) {
  const keep = posFilterActive();
  const rows = neighbors
    .map((n, i) => ({
      ...n,
      normPos: normalizePOS(n.pos),
      posLabel: formatPOS(n.pos),
      rank: i + 1,
    }))
    .filter(n => keep(n.normPos));

  els.tableBody.innerHTML = '';
  rows.forEach(n => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${n.rank}</td><td>${n.word}</td><td>${n.score.toFixed(4)}</td><td><span class="badge">${n.posLabel}</span></td>`;
    els.tableBody.appendChild(tr);
  });
}

async function initStatus({ retry = true } = {}) {
  if (statusRetryTimer) {
    clearTimeout(statusRetryTimer);
    statusRetryTimer = null;
  }

  try {
    const resp = await fetch('/api/status');
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const r = await resp.json();
    els.status.textContent = `Model: ${r.model} • dim=${r.dim || '…'} • vocab=${r.vocabSize || 0}`;
    return true;
  } catch (err) {
    console.error('Status check failed:', err);
    els.status.textContent = 'Waiting for server…';
    if (retry) {
      statusRetryTimer = setTimeout(() => initStatus({ retry: true }), 2000);
    }
    return false;
  }
}

async function rebuildCache() {
  const contextTemplate = els.template.value.trim() || '{w}';
  els.rebuildBtn.disabled = true;
  els.status.textContent = 'Rebuilding cache… (first time takes a bit)';
  try {
    await fetch('/api/rebuild', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ contextTemplate })
    });
    await initStatus();
  } catch (e) {
    console.error(e);
    alert('Rebuild failed: ' + e);
  } finally {
    els.rebuildBtn.disabled = false;
  }
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
    const resp = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const json = await resp.json();
    if (json.error) throw new Error(json.error);
    lastResult = json;

    renderNeighborsTable(json.neighbors);
    drawChart(json);

    const pcsRaw = json.explainedVariance || [];
    const pcs = pcsRaw.length
      ? pcsRaw.map((v, i) => `PC${i + 1} ${(v * 100).toFixed(1)}%`).join(' / ')
      : 'n/a';
    els.status.textContent = `k=${json.meta.k} • vocab=${json.meta.vocabSize} • PCA var: ${pcs}`;
  } catch (e) {
    console.error(e);
    alert('Search failed: ' + e.message);
    els.status.textContent = 'Error.';
  } finally {
    els.runBtn.disabled = false;
  }
}

// UI wiring
els.k.addEventListener('input', () => {
  els.kVal.textContent = els.k.value;
});
els.runBtn.addEventListener('click', runSearch);
els.rebuildBtn.addEventListener('click', rebuildCache);
els.posChecks.forEach(cb => cb.addEventListener('change', () => {
  if (!lastResult) return;
  renderNeighborsTable(lastResult.neighbors);
  drawChart(lastResult);
}));

// boot
initStatus();
ensureThreeContext();
