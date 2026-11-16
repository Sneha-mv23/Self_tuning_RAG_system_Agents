import { describe, expect, it } from 'vitest'
import { api, describeProposal, isActive, jobErrorMessage, parseSse, startBody, validateForm } from './jobs.js'

const res = (status, body) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => {
    if (body === undefined) throw new Error('not json')
    return body
  },
})

describe('isActive', () => {
  it('knows which statuses are still going', () => {
    expect(isActive('running')).toBe(true)
    expect(isActive('awaiting_approval')).toBe(true)
    expect(isActive('finished')).toBe(false)
    expect(isActive('failed')).toBe(false)
  })
})

describe('jobErrorMessage', () => {
  it('explains each status', () => {
    expect(jobErrorMessage(401)).toBe('Wrong admin token.')
    expect(jobErrorMessage(403)).toContain('disabled')
    expect(jobErrorMessage(409, { detail: 'another optimizer job is still running' })).toBe(
      'another optimizer job is still running',
    )
    expect(jobErrorMessage(422, { detail: 'budget is capped' })).toBe('Not accepted: budget is capped')
    expect(jobErrorMessage(500, null)).toBe('The server had a problem.')
  })
})

describe('api', () => {
  it('sends the token header and a JSON body', async () => {
    let seen
    const fetchImpl = async (path, init) => {
      seen = { path, init }
      return res(200, { ok: true })
    }
    const out = await api('/api/jobs', { token: 't', method: 'POST', body: { a: 1 }, fetchImpl })
    expect(out).toEqual({ ok: true })
    expect(seen.init.headers['X-Admin-Token']).toBe('t')
    expect(seen.init.headers['Content-Type']).toBe('application/json')
    expect(JSON.parse(seen.init.body)).toEqual({ a: 1 })
  })
  it('turns an error status into a readable error', async () => {
    await expect(api('/x', { token: 't', fetchImpl: async () => res(401, { detail: 'x' }) })).rejects.toThrow(
      'Wrong admin token.',
    )
  })
  it('explains an unreachable server', async () => {
    const fetchImpl = async () => {
      throw new TypeError('down')
    }
    await expect(api('/x', { token: 't', fetchImpl })).rejects.toThrow('Could not reach the server.')
  })
})

describe('parseSse', () => {
  it('parses complete events and keeps the unfinished rest', () => {
    const text = 'id: 0\ndata: {"i":0,"text":"a"}\n\nid: 1\ndata: {"i":1,"te'
    const { events, rest } = parseSse(text)
    expect(events).toEqual([{ i: 0, text: 'a' }])
    expect(rest).toBe('id: 1\ndata: {"i":1,"te')
  })
  it('skips keepalive comments and malformed events', () => {
    const { events, rest } = parseSse(': keepalive\n\ndata: not json\n\ndata: {"i":2}\n\n')
    expect(events).toEqual([{ i: 2 }])
    expect(rest).toBe('')
  })
  it('handles Windows line endings', () => {
    expect(parseSse('data: {"i":3}\r\n\r\n').events).toEqual([{ i: 3 }])
  })
})

describe('startBody and validateForm', () => {
  const form = { startRun: '5', budget: '150', maxExperiments: '2', dryRun: true, approve: false }
  it('builds the request body with numbers', () => {
    expect(startBody(form)).toEqual({ start_run: 5, budget: 150, max_experiments: 2, dry_run: true, approve: false })
  })
  it('accepts a valid form and rejects bad values', () => {
    expect(validateForm(form, [1, 3, 5])).toBe('')
    expect(validateForm({ ...form, startRun: '9' }, [1, 3, 5])).toContain('starting run')
    expect(validateForm({ ...form, budget: '9999' }, [1, 3, 5])).toContain('Budget')
    expect(validateForm({ ...form, budget: '1.5' }, [1, 3, 5])).toContain('Budget')
    expect(validateForm({ ...form, maxExperiments: '0' }, [1, 3, 5])).toContain('Experiments')
  })
})

describe('describeProposal', () => {
  it('summarises the change, the free screen and the cost', () => {
    const text = describeProposal({ lever: 'top_k', change: { top_k: [4, 6] }, screen_gain: 5, est: 10 })
    expect(text).toBe('top_k (top_k: 4 -> 6) | free screen gain 5 | about 25 API calls')
  })
  it('handles a missing screen gain and no proposal', () => {
    expect(describeProposal({ lever: 'prompt', change: {}, screen_gain: null, est: 0 })).toContain('cannot be screened')
    expect(describeProposal(null)).toBe('')
  })
})