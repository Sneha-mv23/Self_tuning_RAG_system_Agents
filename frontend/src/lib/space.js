import { PRESETS } from './data.js'

export const spaceLabel = (p) => `${p.retriever} | ${p.chunk_size}/${p.overlap} | top_k ${p.top_k}`

export const toPoint = (p) => ({ ...p, x: p.prompt_tokens_estimate, y: p.covered })

export function filterSpace(space, { retriever = 'all', chunkSize = 'all' } = {}) {
  return space.filter(
    (p) =>
      (retriever === 'all' || p.retriever === retriever) &&
      (chunkSize === 'all' || p.chunk_size === Number(chunkSize)),
  )
}

/** The frontier flag was computed over the full space, so always pass the full space here. */
export function frontier(space) {
  return space
    .filter((p) => p.pareto)
    .map(toPoint)
    .sort((a, b) => a.x - b.x || a.y - b.y)
}

/** Find the three evaluated configs in the space. Prompt and temperature do not affect retrieval. */
export function presetPoints(space) {
  const out = {}
  for (const [name, preset] of Object.entries(PRESETS)) {
    const hit = space.find(
      (p) =>
        p.retriever === preset.retriever &&
        p.chunk_size === preset.chunk_size &&
        p.overlap === preset.overlap &&
        p.top_k === preset.top_k,
    )
    if (hit) out[name] = toPoint(hit)
  }
  return out
}

export const bestCoverage = (space) => Math.max(0, ...space.map((p) => p.covered))
export const configsBeating = (space, point) => space.filter((p) => p.covered > point.covered).length
export const tokenDelta = (point, base) => (base && base.x ? (point.x / base.x - 1) * 100 : null)