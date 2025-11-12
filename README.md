# Functional Translation

Functional Translation is an interactive analogy explorer that applies sentence-transformer embeddings to "translate" a target word into a new semantic neighborhood. Seed word pairs (e.g., `garden → gardening`) define a transformation that is transferred onto a chosen target (e.g., `work`). The app exposes the results via a small Express API and a browser-based visualization.

## Getting started

1. **Install dependencies**

   ```bash
   npm install
   ```

2. **Run the development server**

   ```bash
   npm start
   ```

   The server boots on <http://localhost:3000> by default, primes the embedding cache, and serves both the API and the single-page app frontend.

## Data and caching

* **Vocabulary** – The server loads `data/vocab.txt` if present and falls back to a bundled default list of words. 【F:index.js†L71-L101】
* **Embeddings** – Vocabulary terms (optionally wrapped in a context template) are embedded with `@xenova/transformers` and normalized so cosine similarity can be computed with simple dot products. 【F:index.js†L36-L115】【F:index.js†L118-L157】
* **Cache** – The generated embedding matrix is cached on disk (under `cache/` by default) and transparently reused when the model, vocabulary, and template signature matches. 【F:index.js†L103-L155】

## Algorithm walkthrough

1. **Seed translation delta** – For each provided seed pair `(a, b)`, the service embeds both words (respecting the optional context template) and computes the vector difference `b − a`. Averaging these differences produces a single "translation" vector. 【F:index.js†L215-L245】
2. **Target projection** – The target word is embedded, shifted by the average translation vector, and re-normalized to stay on the unit hypersphere used for cosine similarity. 【F:index.js†L245-L262】
3. **Neighbor search** – The translated target vector is compared against the cached vocabulary embeddings via dot product. The top-`k` most similar words are returned, excluding the inputs if requested. 【F:index.js†L157-L208】【F:index.js†L262-L271】
4. **Visualization prep** – The API packages the neighbors, seeds, and both the original and translated target vectors into a PCA-reduced point cloud, along with edge metadata used for front-end rendering. 【F:index.js†L208-L310】

This pipeline enables fast experimentation with analogical transformations while keeping the heavy embedding work cached between requests.
