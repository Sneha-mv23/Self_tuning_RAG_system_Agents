import { describeDetail } from './ask.js'

export const ACTIVE = ['queued', 'running', 'awaiting_approval']
export const isActive = (status) => ACTIVE.includes(status)

export function jobErrorMessage(status, body) {
  const detail = describeDetail(body?.detail)
  if (status === 401) return 'Wrong admin token.'
  if (status === 403) return 'Optimizer jobs are disabled on this server.'
  if (status === 409) return detail || 'Another job is already running.'
  if (status === 422) return `Not accepted: ${detail || 'invalid input'}`
  if (status >= 500) return detail || 'The server had a problem.'
  return detail || `The request failed (${status}).`
}

export async function api(path, { token, method = 'GET', body, fetchImpl = fetch, signal } = {}) {
  let response
  try {
    response = await fetchImpl(path, {
      method,
      signal,
      headers: { 'X-Admin-Token': token, ...(body ? { 'Content-Type': 'application/json' } : {}) },
      body: body ? JSON.stringify(body) : undefined,
    })
  } catch (error) {
    if (error?.name === 'AbortError') throw error
    throw new Error('Could not reach the server.')
  }
  let parsed = null
  try {
    parsed = await response.json()
  } catch {
    // not JSON
  }
  if (!response.ok) throw new Error(jobErrorMessage(response.status, parsed))
  return parsed
}

/** Split a server-sent-events buffer into complete events plus the unfinished remainder. */
export function parseSse(buffer) {
  const events = []
  const parts = buffer.replace(/\r\n/g, '\n').split('\n\n')
  const rest = parts.pop() ?? ''
  for (const part of parts) {
    const data = part
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n')
    if (!data) continue // a keepalive comment
    try {
      events.push(JSON.parse(data))
    } catch {
      // ignore a malformed event
    }
  }
  return { events, rest }
}

/** Stream a job's events with fetch, because EventSource cannot send the token header. */
export async function streamEvents(jobId, { token, after = -1, onEvent, signal, fetchImpl = fetch }) {
  const response = await fetchImpl(`/api/jobs/${jobId}/events?after=${after}`, {
    headers: { 'X-Admin-Token': token },
    signal,
  })
  if (!response.ok) throw new Error(jobErrorMessage(response.status, null))
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const { events, rest } = parseSse(buffer)
    buffer = rest
    events.forEach(onEvent)
  }
}

export const startBody = (form) => ({
  start_run: Number(form.startRun),
  budget: Number(form.budget),
  max_experiments: Number(form.maxExperiments),
  dry_run: form.dryRun,
  approve: form.approve,
})

export function validateForm(form, runIds, limits = { budget: 250, experiments: 3 }) {
  if (!runIds.includes(Number(form.startRun))) return 'Pick a starting run from the list.'
  const budget = Number(form.budget)
  if (!Number.isInteger(budget) || budget < 1 || budget > limits.budget) {
    return `Budget must be a whole number from 1 to ${limits.budget}.`
  }
  const experiments = Number(form.maxExperiments)
  if (!Number.isInteger(experiments) || experiments < 1 || experiments > limits.experiments) {
    return `Experiments must be a whole number from 1 to ${limits.experiments}.`
  }
  return ''
}

export function describeProposal(p) {
  if (!p) return ''
  const change = Object.entries(p.change ?? {})
    .map(([field, [from, to]]) => `${field}: ${from} -> ${to}`)
    .join(', ')
  const gain = p.screen_gain == null ? 'n/a (cannot be screened for free)' : p.screen_gain
  return `${p.lever} (${change}) | free screen gain ${gain} | about ${Math.round((p.est ?? 0) * 2.5)} API calls`
}