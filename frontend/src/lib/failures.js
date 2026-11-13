export const CATEGORIES = {
  pass: { stage: 'none', text: 'Correct and faithful.' },
  retrieval_miss: { stage: 'retrieval', text: 'None of the needed evidence was found in the top 20 results.' },
  partial_retrieval: { stage: 'retrieval', text: 'Some of the needed evidence was found and some was not (multi-hop).' },
  ranking_failure: { stage: 'retrieval', text: 'The evidence was found but ranked below top_k, so the model never saw it.' },
  over_refusal: { stage: 'generation', text: 'The evidence was in the prompt, but the model said it did not know.' },
  mixed_refusal: { stage: 'generation', text: 'The answer gives content and then also says it does not know.' },
  hallucination: { stage: 'generation', text: 'The evidence was in the prompt, but the answer was wrong and unsupported.' },
  incorrect_answer: { stage: 'generation', text: 'The evidence was in the prompt, but the answer missed or contradicted the key point.' },
  unsupported_claims: { stage: 'generation', text: 'The answer was right but added claims the context does not support.' },
  failed_to_decline: { stage: 'generation', text: 'The docs do not answer this question, but the model gave an answer.' },
  error: { stage: 'error', text: 'Generation or judging failed.' },
}
export const CATEGORY_ORDER = Object.keys(CATEGORIES)

export const devRuns = (runs) => runs.filter((r) => r.split === 'dev')

export function categoryCounts(rows) {
  const counts = {}
  for (const row of rows) counts[row.category] = (counts[row.category] ?? 0) + 1
  return counts
}

export function stageCounts(rows) {
  const out = { retrieval: 0, generation: 0, none: 0, error: 0 }
  for (const row of rows) {
    const stage = CATEGORIES[row.category]?.stage ?? 'error'
    out[stage] += 1
  }
  return out
}

export function filterQuestions(rows, { runId, category = 'all', type = 'all', text = '' }) {
  const needle = text.trim().toLowerCase()
  return rows
    .filter(
      (r) =>
        r.run_id === runId &&
        (category === 'all' || r.category === category) &&
        (type === 'all' || r.type === type) &&
        (!needle || r.question.toLowerCase().includes(needle) || r.question_id.toLowerCase().includes(needle)),
    )
    .sort((a, b) => a.question_id.localeCompare(b.question_id))
}

export function rankLabel(row, topK) {
  if (row.type === 'unanswerable') return 'no evidence expected'
  if (row.rank_needed == null) return 'evidence not in the top 20'
  if (row.rank_needed <= topK) return `evidence at rank ${row.rank_needed} (in the prompt)`
  return `evidence at rank ${row.rank_needed} (below top_k ${topK})`
}

export function categoryMap(rows, runId) {
  const map = new Map()
  for (const r of rows) if (r.run_id === runId) map.set(r.question_id, r.category)
  return map
}