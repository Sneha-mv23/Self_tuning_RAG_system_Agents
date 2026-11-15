import { describe, expect, it } from 'vitest'
import { bestCoverage, configsBeating, filterSpace, frontier, presetPoints, tokenDelta } from './space.js'

const point = (retriever, chunk_size, overlap, top_k, covered, tokens, pareto) => ({
  retriever, chunk_size, overlap, top_k, covered, prompt_tokens_estimate: tokens, pareto,
})
const space = [
  point('dense', 256, 32, 4, 36, 1204, false),
  point('dense', 256, 32, 6, 41, 1746, true),
  point('hybrid', 256, 32, 3, 39, 933, true),
  point('hybrid', 128, 16, 3, 33, 549, true),
]

describe('filterSpace', () => {
  it('filters by retriever and chunk size', () => {
    expect(filterSpace(space, {})).toHaveLength(4)
    expect(filterSpace(space, { retriever: 'hybrid' })).toHaveLength(2)
    expect(filterSpace(space, { chunkSize: '256' })).toHaveLength(3)
    expect(filterSpace(space, { retriever: 'hybrid', chunkSize: '128' })).toHaveLength(1)
  })
})

describe('frontier', () => {
  it('keeps only flagged points, sorted by prompt tokens', () => {
    expect(frontier(space).map((p) => p.x)).toEqual([549, 933, 1746])
    expect(frontier(space).map((p) => p.y)).toEqual([33, 39, 41])
  })
})

describe('presetPoints', () => {
  it('finds the baseline, optimized and cheap configs', () => {
    const found = presetPoints(space)
    expect(Object.keys(found)).toEqual(['baseline', 'optimized', 'cheap'])
    expect(found.optimized.y).toBe(41)
    expect(found.cheap.x).toBe(933)
  })
})

describe('summaries', () => {
  it('counts the configs that cover more', () => {
    expect(bestCoverage(space)).toBe(41)
    expect(configsBeating(space, { covered: 36 })).toBe(2)
    expect(configsBeating(space, { covered: 41 })).toBe(0)
  })
  it('computes the token change against a base point', () => {
    expect(tokenDelta({ x: 1746 }, { x: 1204 })).toBeCloseTo(45, 0)
    expect(tokenDelta({ x: 1746 }, null)).toBeNull()
  })
})