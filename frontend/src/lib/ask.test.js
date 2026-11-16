import { describe, expect, it } from 'vitest'
import {
  askQuestion, checkApi, describeDetail, errorMessage, formatWait, presetConfig, scoreKind,
} from './ask.js'

const res = (status, body, headers = {}) => ({
  ok: status >= 200 && status < 300,
  status,
  headers: { get: (key) => headers[key] ?? null },
  json: async () => {
    if (body === undefined) throw new Error('not json')
    return body
  },
})

describe('presetConfig', () => {
  it('returns a copy of the preset', () => {
    const a = presetConfig('optimized')
    a.top_k = 99
    expect(presetConfig('optimized').top_k).toBe(6)
    expect(presetConfig('baseline')).toEqual({
      retriever: 'dense', chunk_size: 256, overlap: 32, top_k: 4, prompt: 'strict', temperature: 0,
    })
  })
  it('rejects unknown names', () => {
    expect(() => presetConfig('nope')).toThrow('unknown preset')
  })
})

describe('scoreKind', () => {
  it('names the score type', () => {
    expect(scoreKind('dense')).toBe('cosine similarities')
    expect(scoreKind('hybrid')).toContain('fused')
  })
})

describe('describeDetail', () => {
  it('handles strings, validation lists and other values', () => {
    expect(describeDetail('plain')).toBe('plain')
    expect(describeDetail([{ msg: 'a' }, { msg: 'b' }])).toBe('a; b')
    expect(describeDetail([{ loc: 1 }])).toBe('{"loc":1}')
    expect(describeDetail(undefined)).toBe('')
  })
})

describe('formatWait', () => {
  it('picks a readable unit', () => {
    expect(formatWait(NaN)).toBe('a while')
    expect(formatWait(0)).toBe('a while')
    expect(formatWait(45)).toBe('45 seconds')
    expect(formatWait(120)).toBe('2 minutes')
    expect(formatWait(7200)).toBe('2 hours')
  })
})

describe('errorMessage', () => {
  it('adds the retry time to a 429 when the server sent one', () => {
    const msg = errorMessage(429, { detail: 'Rate limit: too many questions from this client.' }, '1800')
    expect(msg).toBe('Rate limit: too many questions from this client. (retry in about 30 minutes)')
  })
  it('leaves a 429 without Retry-After alone', () => {
    expect(errorMessage(429, { detail: 'Quota used.' }, null)).toBe('Quota used.')
    expect(errorMessage(429, null, null)).toBe('Rate limit reached.')
  })
  it('explains rejected input', () => {
    expect(errorMessage(422, { detail: [{ msg: 'too short' }] })).toBe('That request was not accepted: too short')
  })
  it('falls back for server and other errors', () => {
    expect(errorMessage(503, { detail: 'Model unreachable.' })).toBe('Model unreachable.')
    expect(errorMessage(500, null)).toBe('The demo server had a problem. Try again in a moment.')
    expect(errorMessage(404, null)).toBe('The request failed (404).')
  })
})

describe('askQuestion', () => {
  it('posts the question and config and returns the body', async () => {
    let seen
    const fetchImpl = async (url, init) => {
      seen = { url, init }
      return res(200, { answer: 'hi' })
    }
    const out = await askQuestion('What?', { top_k: 3 }, { fetchImpl })
    expect(out).toEqual({ answer: 'hi' })
    expect(seen.url).toBe('/api/ask')
    expect(seen.init.method).toBe('POST')
    expect(JSON.parse(seen.init.body)).toEqual({ question: 'What?', config: { top_k: 3 } })
  })
  it('turns a 429 into a readable error', async () => {
    const fetchImpl = async () => res(429, { detail: 'Rate limit: slow down.' }, { 'Retry-After': '1800' })
    await expect(askQuestion('q', {}, { fetchImpl })).rejects.toThrow('retry in about 30 minutes')
  })
  it('explains an unreachable server', async () => {
    const fetchImpl = async () => {
      throw new TypeError('fetch failed')
    }
    await expect(askQuestion('q', {}, { fetchImpl })).rejects.toThrow('Could not reach the demo server')
  })
  it('explains a cancelled request', async () => {
    const fetchImpl = async () => {
      const error = new Error('aborted')
      error.name = 'AbortError'
      throw error
    }
    await expect(askQuestion('q', {}, { fetchImpl })).rejects.toThrow('cancelled or timed out')
  })
  it('survives an error response that is not JSON', async () => {
    const fetchImpl = async () => res(500, undefined)
    await expect(askQuestion('q', {}, { fetchImpl })).rejects.toThrow('The demo server had a problem')
  })
})

describe('checkApi', () => {
  it('returns the health body when the API answers', async () => {
    expect(await checkApi(async () => res(200, { status: 'ok' }))).toEqual({ status: 'ok' })
  })
  it('returns null for an error status', async () => {
    expect(await checkApi(async () => res(500, {}))).toBeNull()
  })
  it('returns null when the request throws or the body is not JSON', async () => {
    expect(await checkApi(async () => { throw new Error('down') })).toBeNull()
    expect(await checkApi(async () => res(200, undefined))).toBeNull()
  })
})