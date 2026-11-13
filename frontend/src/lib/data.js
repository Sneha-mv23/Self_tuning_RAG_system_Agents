export const PRESETS = {
  baseline: { retriever: 'dense', chunk_size: 256, overlap: 32, top_k: 4, prompt: 'strict', temperature: 0 },
  optimized: { retriever: 'dense', chunk_size: 256, overlap: 32, top_k: 6, prompt: 'strict', temperature: 0 },
  cheap: { retriever: 'hybrid', chunk_size: 256, overlap: 32, top_k: 3, prompt: 'strict', temperature: 0 },
}
const ORDER = Object.keys(PRESETS)
const rank = (name) => {
  const i = ORDER.indexOf(name)
  return i === -1 ? ORDER.length : i
}

export function presetName(run) {
  for (const [name, preset] of Object.entries(PRESETS)) {
    if (Object.keys(preset).every((key) => run[key] === preset[key])) return name
  }
  return null
}

export const configLabel = (run) =>
  `${run.retriever} | ${run.chunk_size}/${run.overlap} | top_k ${run.top_k} | ${run.prompt}`

export const pct = (x) => (x == null ? '-' : `${Math.round(x * 100)}%`)
export const fmt = (x, digits = 2) => (x == null ? '-' : Number(x).toFixed(digits))

export const splitRuns = (runs, split) => runs.filter((r) => r.split === split)

const mean = (rows, field) => rows.reduce((sum, r) => sum + r[field], 0) / rows.length

export function summarizeTest(runs) {
  const groups = new Map()
  for (const run of runs) {
    if (run.split !== 'test') continue
    const name = presetName(run) ?? configLabel(run)
    if (!groups.has(name)) groups.set(name, [])
    groups.get(name).push(run)
  }
  const summaries = [...groups].map(([name, rows]) => {
    rows.sort((a, b) => a.replicate - b.replicate)
    const passes = rows.map((r) => r.passes)
    return {
      name,
      replicates: rows.length,
      passes,
      meanPasses: passes.reduce((a, b) => a + b, 0) / passes.length,
      nQuestions: rows[0].n_questions,
      recall: rows[0].recall_at_k,
      correct: mean(rows, 'answer_correctness'),
      faithful: mean(rows, 'faithfulness'),
      decline: mean(rows, 'proper_refusal'),
      tokens: mean(rows, 'mean_prompt_tokens'),
    }
  })
  return summaries.sort((a, b) => rank(a.name) - rank(b.name))
}

/** Percent change in mean prompt tokens against the baseline, or null when there is no baseline. */
export function tokenChange(summary, baseline) {
  if (!baseline || !baseline.tokens) return null
  return (summary.tokens / baseline.tokens - 1) * 100
}