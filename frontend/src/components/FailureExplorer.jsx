import { useMemo, useState } from 'react'
import { configLabel } from '../lib/data.js'
import {
  CATEGORIES, CATEGORY_ORDER, categoryCounts, categoryMap, devRuns, filterQuestions, rankLabel, stageCounts,
} from '../lib/failures.js'

const TYPES = ['all', 'factual', 'paraphrase', 'multi_hop', 'unanswerable']
const words = (s) => s.replace(/_/g, ' ')

export default function FailureExplorer({ data }) {
  const runs = devRuns(data.runs)
  const rows = data.dev_questions ?? []
  const [runId, setRunId] = useState(runs.length ? runs[runs.length - 1].id : null)
  const [compareId, setCompareId] = useState('none')
  const [category, setCategory] = useState('all')
  const [type, setType] = useState('all')
  const [text, setText] = useState('')

  const runRows = useMemo(() => rows.filter((r) => r.run_id === runId), [rows, runId])
  const other = useMemo(
    () => (compareId === 'none' ? null : categoryMap(rows, Number(compareId))),
    [rows, compareId],
  )

  if (!runs.length) return <p className="muted">No dev runs in the exported results.</p>

  const run = runs.find((r) => r.id === runId)
  const counts = categoryCounts(runRows)
  const stages = stageCounts(runRows)
  const shown = filterQuestions(rows, { runId, category, type, text })
  const runLabel = (r) => `Run ${r.id}: ${r.generator_model} | ${configLabel(r)}`

  return (
    <div>
      <h2>Failure explorer</h2>
      <p className="muted">
        Dev questions only. The held-out test split is shown as aggregates on the Overview tab, so its individual
        failures are never exposed. Failure categories come from the rule-based classifier.
      </p>

      <div className="controls">
        <label>
          Run
          <select value={runId} onChange={(e) => setRunId(Number(e.target.value))}>
            {runs.map((r) => (
              <option key={r.id} value={r.id}>{runLabel(r)}</option>
            ))}
          </select>
        </label>
        <label>
          Compare with
          <select value={compareId} onChange={(e) => setCompareId(e.target.value)}>
            <option value="none">nothing</option>
            {runs.filter((r) => r.id !== runId).map((r) => (
              <option key={r.id} value={r.id}>{runLabel(r)}</option>
            ))}
          </select>
        </label>
        <label>
          Type
          <select value={type} onChange={(e) => setType(e.target.value)}>
            {TYPES.map((t) => (
              <option key={t} value={t}>{words(t)}</option>
            ))}
          </select>
        </label>
        <label>
          Search
          <input value={text} onChange={(e) => setText(e.target.value)} placeholder="question text or id" />
        </label>
      </div>

      <p>
        <strong>{run.passes} of {run.n_questions}</strong> pass. Failures: retrieval {stages.retrieval}, generation{' '}
        {stages.generation}, error {stages.error}.
      </p>

      <div className="chips">
        <button className={category === 'all' ? 'chip active' : 'chip'} onClick={() => setCategory('all')}>
          all ({runRows.length})
        </button>
        {CATEGORY_ORDER.filter((c) => counts[c]).map((c) => (
          <button
            key={c}
            className={category === c ? 'chip active' : 'chip'}
            title={CATEGORIES[c].text}
            onClick={() => setCategory(c)}
          >
            {words(c)} ({counts[c]})
          </button>
        ))}
      </div>

      {shown.length === 0 && <p className="muted">No questions match these filters.</p>}
      {shown.map((row) => {
        const was = other?.get(row.question_id)
        const changed = other && was !== undefined && was !== row.category
        return (
          <details key={row.question_id} className="q">
            <summary>
              <span className="qid">{row.question_id}</span>
              <span className={`tag stage-${CATEGORIES[row.category]?.stage ?? 'error'}`}>{words(row.category)}</span>
              <span className="qtype">{row.type.replace('_', '-')}</span>
              <span className="qtext">{row.question}</span>
              {changed && <span className="changed">was {words(was)} in run {compareId}</span>}
            </summary>
            <div className="q-body">
              <div className="muted">{CATEGORIES[row.category]?.text}</div>
              <div className="muted">{rankLabel(row, run.top_k)}</div>
              <pre className="answer">{row.answer || '(no answer stored)'}</pre>
              <div className="muted">Answers are stored truncated to 600 characters.</div>
            </div>
          </details>
        )
      })}
    </div>
  )
}