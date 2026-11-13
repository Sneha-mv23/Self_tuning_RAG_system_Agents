import { describe, expect, it } from 'vitest'
import { categoryCounts, categoryMap, filterQuestions, rankLabel, stageCounts } from './failures.js'

const rows = [
  { run_id: 1, question_id: 'q002', type: 'factual', question: 'How do I use Form?', category: 'pass' },
  { run_id: 1, question_id: 'q001', type: 'unanswerable', question: 'Deploy to Lambda?', category: 'failed_to_decline' },
  { run_id: 2, question_id: 'q001', type: 'unanswerable', question: 'Deploy to Lambda?', category: 'pass' },
]

describe('counts', () => {
  it('counts categories', () => {
    expect(categoryCounts(rows)).toEqual({ pass: 2, failed_to_decline: 1 })
  })
  it('groups categories into stages', () => {
    const staged = ['pass', 'ranking_failure', 'retrieval_miss', 'over_refusal', 'failed_to_decline'].map(
      (category) => ({ category }),
    )
    expect(stageCounts(staged)).toEqual({ retrieval: 2, generation: 2, none: 1, error: 0 })
  })
})

describe('filterQuestions', () => {
  it('filters by run, category, type and text, sorted by id', () => {
    expect(filterQuestions(rows, { runId: 1 }).map((r) => r.question_id)).toEqual(['q001', 'q002'])
    expect(filterQuestions(rows, { runId: 1, category: 'failed_to_decline' }).map((r) => r.question_id)).toEqual(['q001'])
    expect(filterQuestions(rows, { runId: 1, type: 'factual' }).map((r) => r.question_id)).toEqual(['q002'])
    expect(filterQuestions(rows, { runId: 1, text: 'FORM' }).map((r) => r.question_id)).toEqual(['q002'])
    expect(filterQuestions(rows, { runId: 1, text: 'q001' }).map((r) => r.question_id)).toEqual(['q001'])
  })
  it('does not reorder the input', () => {
    filterQuestions(rows, { runId: 1 })
    expect(rows[0].question_id).toBe('q002')
  })
})

describe('rankLabel', () => {
  it('describes questions with no evidence or missing evidence', () => {
    expect(rankLabel({ type: 'unanswerable', rank_needed: null }, 4)).toBe('no evidence expected')
    expect(rankLabel({ type: 'factual', rank_needed: null }, 4)).toBe('evidence not in the top 20')
  })
  it('says whether the evidence was inside top_k', () => {
    expect(rankLabel({ type: 'factual', rank_needed: 3 }, 4)).toBe('evidence at rank 3 (in the prompt)')
    expect(rankLabel({ type: 'factual', rank_needed: 5 }, 4)).toBe('evidence at rank 5 (below top_k 4)')
  })
})

describe('categoryMap', () => {
  it('maps question ids to categories for one run', () => {
    const map = categoryMap(rows, 2)
    expect(map.size).toBe(1)
    expect(map.get('q001')).toBe('pass')
  })
})