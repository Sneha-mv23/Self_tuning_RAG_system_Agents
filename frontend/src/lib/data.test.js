import { describe, expect, it } from 'vitest'
import { fmt, pct, presetName, summarizeTest, tokenChange } from './data.js'

const run = (over = {}) => ({
  split: 'test', retriever: 'dense', chunk_size: 256, overlap: 32, top_k: 4, prompt: 'strict',
  temperature: 0.0, replicate: 0, passes: 37, n_questions: 40, recall_at_k: 0.84,
  answer_correctness: 0.94, faithfulness: 1, proper_refusal: 0.88, mean_prompt_tokens: 1000, ...over,
})

describe('presetName', () => {
  it('recognises the three evaluated configs', () => {
    expect(presetName(run())).toBe('baseline')
    expect(presetName(run({ top_k: 6 }))).toBe('optimized')
    expect(presetName(run({ retriever: 'hybrid', top_k: 3 }))).toBe('cheap')
  })
  it('returns null for anything else', () => {
    expect(presetName(run({ top_k: 5 }))).toBeNull()
  })
})

describe('summarizeTest', () => {
  const runs = [
    run({ top_k: 6, replicate: 1, passes: 38, mean_prompt_tokens: 1600 }),
    run({ retriever: 'hybrid', top_k: 3, passes: 38, mean_prompt_tokens: 800 }),
    run({ replicate: 1, passes: 36 }),
    run({ top_k: 6, replicate: 0, passes: 38, mean_prompt_tokens: 1400 }),
    run(),
    run({ split: 'dev', passes: 49 }),
  ]
  const out = summarizeTest(runs)

  it('orders the presets baseline, optimized, cheap and ignores dev runs', () => {
    expect(out.map((s) => s.name)).toEqual(['baseline', 'optimized', 'cheap'])
  })
  it('sorts passes by replicate and averages', () => {
    expect(out[0].passes).toEqual([37, 36])
    expect(out[0].meanPasses).toBe(36.5)
    expect(out[1].tokens).toBe(1500)
  })
})

describe('helpers', () => {
  it('computes the token change against the baseline', () => {
    expect(tokenChange({ tokens: 800 }, { tokens: 1000 })).toBeCloseTo(-20)
    expect(tokenChange({ tokens: 800 }, null)).toBeNull()
  })
  it('formats numbers', () => {
    expect(pct(0.876)).toBe('88%')
    expect(pct(null)).toBe('-')
    expect(fmt(0.5)).toBe('0.50')
  })
})