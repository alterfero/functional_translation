import express from 'express';
import cors from 'cors';
import path from 'path';
import fs from 'fs-extra';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import PCA from 'ml-pca';
import { pipeline } from '@xenova/transformers';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- Config ---
const PORT = process.env.PORT || 3000;
const PUBLIC_DIR = path.join(__dirname, 'public');
const DATA_DIR = process.env.DATA_DIR || path.join(__dirname, 'data');
const CACHE_DIR = process.env.CACHE_DIR || process.env.RAILWAY_VOLUME_MOUNT_PATH || path.join(__dirname, 'cache');
const VOCAB_PATH = process.env.VOCAB_PATH || path.join(DATA_DIR, 'vocab.txt');
const MODEL_ID = process.env.MODEL_ID || 'Xenova/all-MiniLM-L6-v2'; // 384-dim, fast, good quality
const BATCH_SIZE = Number(process.env.BATCH_SIZE || 64);
const DEFAULT_K = 200;

// Ensure dirs
await fs.ensureDir(CACHE_DIR);
await fs.ensureDir(DATA_DIR);

// --- Globals (populated on boot) ---
let embedder = null;         // @xenova/transformers pipeline
let vocab = [];              // array of words
let dim = 0;                 // embedding dimension (should be 384 here)
let matrix = null;           // Float32Array, size vocab.length * dim (row-major)
let normalized = false;      // whether matrix rows are unit-length

// --- Helpers ---
const hashJSON = (obj) => crypto.createHash('sha256').update(JSON.stringify(obj)).digest('hex');

function normalizeVec(vec) {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
  const norm = Math.sqrt(s) || 1;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / norm;
  return out;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function getRow(i) {
  const start = i * dim;
  return matrix.subarray(start, start + dim);
}

function wordsToTexts(words, template = '{w}') {
  // Allow giving context like "I saw {w} yesterday."
  return words.map(w => (template.includes('{w}') ? template.replaceAll('{w}', w) : w));
}

async function ensureEmbedder() {
  if (embedder) return embedder;
  console.log(`[model] loading ${MODEL_ID}…`);
  embedder = await pipeline('feature-extraction', MODEL_ID, { quantized: true });
  // We request pooled + normalized vectors so we can use dot() for cosine
  const test = await embedder('test', { pooling: 'mean', normalize: true });
  dim = test.data ? test.data.length : test.length; // supports Tensor or plain array
  console.log(`[model] ready, dim=${dim}`);
  return embedder;
}

async function embedMany(texts) {
  const pipe = await ensureEmbedder();
  const out = await pipe(texts, { pooling: 'mean', normalize: true });

  const toVec = (x) => {
    if (x?.data) return (x.data instanceof Float32Array) ? x.data : Float32Array.from(x.data);
    if (x instanceof Float32Array) return x;
    if (Array.isArray(x)) return Float32Array.from(x);
    return null;
  };

  // Case 1: pipeline already returns an array of per-item outputs
  if (Array.isArray(out)) {
    const vecs = out.map(toVec);
    if (vecs.some(v => v == null)) throw new Error('Unexpected embedding output (array case)');
    return vecs;
  }

  // Case 2: single object / tensor with concatenated data → split by `dim`
  const vec = toVec(out);
  if (!vec) throw new Error('Unexpected embedding output (tensor case)');
  if (!dim) throw new Error('Embedding dimension not initialized');

  const n = Math.round(vec.length / dim);
  if (n * dim !== vec.length) {
    // Fallback: treat as single vector
    return [vec];
  }
  if (n === 1) return [vec];

  const res = new Array(n);
  for (let i = 0; i < n; i++) {
    res[i] = vec.subarray(i * dim, (i + 1) * dim);
  }
  return res;
}

async function loadVocab() {
  if (await fs.pathExists(VOCAB_PATH)) {
    const raw = await fs.readFile(VOCAB_PATH, 'utf8');
    const lines = raw.split(/\r?\n/).map(x => x.trim()).filter(Boolean);
    vocab = Array.from(new Set(lines)); // dedupe
  } else {
    // A tiny default vocabulary so things run even without a file.
    vocab = [
      'garden','gardening','belief','believing','fight','fighting',
      'run','running','walk','walking','play','playing','think','thinking',
      'write','writing','drive','driving','code','coding','swim','swimming',
      'invent','inventing','create','creating','design','designing','build','building',
      'teacher','student','pilot','sailor','engineer','scientist','company','market',
      'product','prototype','research','energy','health','transportation','data','model',
      'language','word','verb','noun','adjective','adverb','plural','past','future',
      'fast','faster','fastest','smart','smarter','smartest','happy','happier','happiest',
      'good','better','best','bad','worse','worst','large','larger','largest',
      'dog','dogs','cat','cats','city','cities','child','children','mouse','mice'
    ];
  }
  console.log(`[vocab] size=${vocab.length}`);
}

async function buildOrLoadEmbeddingMatrix(contextTemplate = '{w}') {
  await loadVocab();
  await ensureEmbedder();

  const signature = hashJSON({ model: MODEL_ID, dim, vocab, contextTemplate });
  const metaPath = path.join(CACHE_DIR, 'meta.json');
  const binPath  = path.join(CACHE_DIR, 'vocab_embeddings.bin');

  if (await fs.pathExists(metaPath) && await fs.pathExists(binPath)) {
    const meta = JSON.parse(await fs.readFile(metaPath, 'utf8'));
    if (meta.signature === signature) {
      const buf = await fs.readFile(binPath);
      const u8 = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
      matrix = new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
      dim = meta.dim;
      normalized = meta.normalized;
      console.log(`[cache] loaded matrix n=${meta.n} dim=${dim} normalized=${normalized}`);
      return;
    }
  }

  console.log('[cache] building vocab embedding matrix (first run or changed vocab/template)…');
  const texts = wordsToTexts(vocab, contextTemplate);
  const n = vocab.length;
  matrix = new Float32Array(n * dim);

  for (let i = 0; i < n; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE);
    const embs = await embedMany(batch);
    for (let j = 0; j < embs.length; j++) {
      matrix.set(embs[j], (i + j) * dim);
    }
    if (((i / BATCH_SIZE) % 10) === 0) {
      console.log(`  embedded ${Math.min(i + BATCH_SIZE, n)} / ${n}`);
    }
  }
  // normalize rows for cosine
  for (let i = 0; i < n; i++) {
    const row = getRow(i);
    const normed = normalizeVec(row);
    matrix.set(normed, i * dim);
  }
  normalized = true;

  await fs.writeFile(binPath, Buffer.from(matrix.buffer));
  await fs.writeFile(metaPath, JSON.stringify({ n: vocab.length, dim, normalized, signature }, null, 2));
  console.log('[cache] matrix saved');
}

function topKSimilar(targetVec, k = DEFAULT_K, exclude = new Set()) {
  const q = normalized ? normalizeVec(targetVec) : targetVec;
  const sims = new Float32Array(vocab.length);
  for (let i = 0; i < vocab.length; i++) {
    sims[i] = dot(getRow(i), q);
  }
  // build indices and partial sort
  const idxs = [...sims.keys()];
  idxs.sort((a, b) => sims[b] - sims[a]);

  const res = [];
  for (const i of idxs) {
    const w = vocab[i];
    if (exclude.has(w)) continue;
    res.push({ word: w, score: sims[i], index: i });
    if (res.length >= k) break;
  }
  return res;
}

function makePCA(points) {
  // points: array of { id, label, vec: Float32Array }
  if (!points.length) {
    return { points: [], explainedVariance: [] };
  }

  const X = points.map(p => Array.from(p.vec));
  const nDims = X[0].length || 0;
  const components = Math.max(1, Math.min(3, nDims));
  const pca = new PCA(X, { center: true, scale: false });
  const Y = pca.predict(X, { nComponents: components }).to2DArray();
  const ev = pca.getExplainedVariance(); // fraction per component

  for (let i = 0; i < points.length; i++) {
    points[i].x = Y[i][0] || 0;
    points[i].y = Y[i][1] || 0;
    points[i].z = Y[i][2] || 0;
  }

  return { points, explainedVariance: ev.slice(0, components) };
}

// --- Express app ---
const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));
app.use(express.static(PUBLIC_DIR));

// quick health + metadata
app.get('/api/status', async (req, res) => {
  try {
    const ready = !!matrix;
    res.json({
      ready,
      vocabSize: vocab.length || 0,
      dim,
      model: MODEL_ID,
      cacheDir: CACHE_DIR
    });
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

// (re)build cache on demand with (optional) template change
app.post('/api/rebuild', async (req, res) => {
  try {
    const { contextTemplate = '{w}' } = req.body || {};
    await buildOrLoadEmbeddingMatrix(contextTemplate);
    res.json({ ok: true, vocabSize: vocab.length, dim, model: MODEL_ID });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

// main search endpoint
app.post('/api/search', async (req, res) => {
  try {
    const {
      pairs = [['garden','gardening'], ['belief','believing'], ['fight','fighting']],
      target = 'work',
      k = DEFAULT_K,
      contextTemplate = '{w}',
      includeSeeds = true,
      excludeInputs = true
    } = req.body || {};

    if (!matrix) {
      // ensure cached with the requested template
      await buildOrLoadEmbeddingMatrix(contextTemplate);
    }

    // embed pairs and target with context template
    const flatWords = [];
    const pairObjs = [];
    for (const [a, b] of pairs) {
      if (typeof a !== 'string' || typeof b !== 'string') continue;
      pairObjs.push({ a, b });
      flatWords.push(a, b);
    }
    flatWords.push(target);

    const texts = wordsToTexts(flatWords, contextTemplate);
    const embs = await embedMany(texts);

    // map back
    const targetEmb = embs[embs.length - 1];
    const deltas = [];
    for (let i = 0; i < pairObjs.length; i++) {
      const fromEmb = embs[2*i];
      const toEmb   = embs[2*i + 1];
      const delta = new Float32Array(dim);
      for (let j = 0; j < dim; j++) delta[j] = toEmb[j] - fromEmb[j];
      deltas.push(delta);
    }
    // average delta
    const avgDelta = new Float32Array(dim);
    for (const d of deltas) for (let j = 0; j < dim; j++) avgDelta[j] += d[j];
    if (deltas.length > 0) for (let j = 0; j < dim; j++) avgDelta[j] /= deltas.length;

    // transformed target vector (raw sum of target + delta)
    const transformed = new Float32Array(dim);
    for (let j = 0; j < dim; j++) transformed[j] = targetEmb[j] + avgDelta[j];

    // cosine similarity works on unit vectors, so align the translated target
    const translated = normalized ? normalizeVec(transformed) : transformed;

    // knn among vocab using the translated target vector
    const excludeSet = new Set(excludeInputs ? [...pairs.flat(), target] : []);
    const neighbors = topKSimilar(translated, k, excludeSet);

    // assemble points for PCA
    const pPoints = [];
    const seedLinks = [];

    // neighbors
    for (const n of neighbors) {
      pPoints.push({
        id: `neighbor:${n.word}`,
        label: n.word,
        kind: 'neighbor',
        vec: getRow(n.index)
      });
    }

    // target & predicted
    pPoints.push({ id: 'target', label: target, kind: 'target', vec: targetEmb });
    pPoints.push({ id: 'predicted', label: `${target}*`, kind: 'predicted', vec: translated });

    // seeds
    if (includeSeeds) {
      for (let i = 0; i < pairObjs.length; i++) {
        const a = pairObjs[i].a;
        const b = pairObjs[i].b;
        const aEmb = embs[2*i];
        const bEmb = embs[2*i + 1];
        const fromId = `seed:${a}`;
        const toId = `seed:${b}`;
        pPoints.push({ id: fromId, label: a, kind: 'seedFrom', vec: aEmb });
        pPoints.push({ id: toId,   label: b, kind: 'seedTo',   vec: bEmb });
        seedLinks.push({
          id: `pair:${i}`,
          fromId,
          toId,
          fromLabel: a,
          toLabel: b
        });
      }
    }

    const { points, explainedVariance } = makePCA(pPoints);

    const pointMap = new Map(points.map(p => [p.id, p]));
    const enrichedSeedLinks = seedLinks.map(link => ({
      id: link.id,
      fromId: link.fromId,
      toId: link.toId,
      fromLabel: link.fromLabel,
      toLabel: link.toLabel,
      from: pointMap.has(link.fromId) ? {
        x: pointMap.get(link.fromId).x,
        y: pointMap.get(link.fromId).y,
        z: pointMap.get(link.fromId).z || 0
      } : null,
      to: pointMap.has(link.toId) ? {
        x: pointMap.get(link.toId).x,
        y: pointMap.get(link.toId).y,
        z: pointMap.get(link.toId).z || 0
      } : null
    }));

    const neighborScoreMap = new Map(neighbors.map(n => [`neighbor:${n.word}`, n.score]));

    res.json({
      avgDelta: Array.from(avgDelta),
      targetEmbedding: Array.from(targetEmb),
      transformed: Array.from(translated),
      transformedRaw: Array.from(transformed),
      neighbors: neighbors.map(n => ({ word: n.word, score: n.score })),
      points: points.map(p => ({
        id: p.id,
        label: p.label,
        kind: p.kind,
        x: p.x,
        y: p.y,
        z: p.z || 0,
        score: neighborScoreMap.get(p.id) || null
      })),
      seedLinks: enrichedSeedLinks,
      explainedVariance,
      meta: {
        vocabSize: vocab.length,
        dim,
        model: MODEL_ID,
        k
      }
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

// serve SPA
app.get('*', (req, res) => res.sendFile(path.join(PUBLIC_DIR, 'index.html')));

// boot
await buildOrLoadEmbeddingMatrix('{w}'); // prime default template on startup
app.listen(PORT, () => console.log(`Server listening on http://localhost:${PORT}`));
