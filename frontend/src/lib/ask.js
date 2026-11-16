import { PRESETS } from './data.js'

export const EXAMPLES = [
  'How do I make a query parameter required?',
  'How do I declare a path parameter?',
  'How do I set a rate limit on a FastAPI endpoint?',
]

export function presetConfig(name) {
  const preset = PRESETS[name]
  if (!preset) throw new Error(`unknown preset: ${name}`)
  return { ...preset }
}

export const scoreKind = (retriever) =>
  retriever === 'hybrid' ? 'fused rank scores (not similarities)' : 'cosine similarities'

export function describeDetail(detail) {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((d) => d?.msg ?? JSON.stringify(d)).join('; ')
  return ''
}

export function formatWait(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return 'a while'
  if (seconds < 90) return `${Math.ceil(seconds)} seconds`
  if (seconds < 5400) return `${Math.round(seconds / 60)} minutes`
  return `${Math.round(seconds / 3600)} hours`
}

export function errorMessage(status, body, retryAfter) {
  const detail = describeDetail(body?.detail)
  if (status === 429) {
    const hasWait = retryAfter != null && retryAfter !== '' && Number.isFinite(Number(retryAfter))
    const wait = hasWait ? ` (retry in about ${formatWait(Number(retryAfter))})` : ''
    return `${detail || 'Rate limit reached.'}${wait}`
  }
  if (status === 422) return `That request was not accepted: ${detail || 'invalid input'}`
  if (status >= 500) return detail || 'The demo server had a problem. Try again in a moment.'
  return detail || `The request failed (${status}).`
}

export async function askQuestion(question, config, { fetchImpl = fetch, signal } = {}) {
  let response
  try {
    response = await fetchImpl('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, config }),
      signal,
    })
  } catch (error) {
    if (error?.name === 'AbortError') throw new Error('The request was cancelled or timed out.')
    throw new Error('Could not reach the demo server. You can still browse the saved results.')
  }
  let body = null
  try {
    body = await response.json()
  } catch {
    // an error page that is not JSON
  }
  if (!response.ok) throw new Error(errorMessage(response.status, body, response.headers.get('Retry-After')))
  return body
}

/** The health body when the API is up, otherwise null (also null on a static host that returns HTML). */
export async function checkApi(fetchImpl = fetch) {
  try {
    const response = await fetchImpl('/api/health')
    if (!response.ok) return null
    const body = await response.json()
    return body?.status === 'ok' ? body : null
  } catch {
    return null
  }
}