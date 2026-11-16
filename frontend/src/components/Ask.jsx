import { useEffect, useRef, useState } from 'react'
import { askQuestion, checkApi, EXAMPLES, presetConfig, scoreKind } from '../lib/ask.js'
import { configLabel, fmt } from '../lib/data.js'
import '../ask.css'

const TIMEOUT_MS = 60000
const PRESET_HELP = {
  optimized: "dense retrieval, top 6 chunks (the optimizer's pick)",
  baseline: 'dense retrieval, top 4 chunks (where the optimizer started)',
  cheap: 'hybrid retrieval, top 3 chunks (the shortest prompt)',
}

export default function Ask() {
  const [health, setHealth] = useState(undefined) // undefined while checking, null when unavailable
  const [question, setQuestion] = useState('')
  const [preset, setPreset] = useState('optimized')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const alive = useRef(true)

  const check = () => {
    setHealth(undefined)
    checkApi().then((h) => {
      if (alive.current) setHealth(h)
    })
  }
  useEffect(() => {
    alive.current = true
    check()
    return () => {
      alive.current = false
    }
  }, [])

  async function submit(event) {
    event?.preventDefault()
    const q = question.trim()
    if (q.length < 3 || busy) return
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
    setBusy(true)
    setError('')
    try {
      setResult(await askQuestion(q, presetConfig(preset), { signal: controller.signal }))
    } catch (err) {
      setResult(null)
      setError(err.message)
    } finally {
      clearTimeout(timer)
      setBusy(false)
    }
  }

  return (
    <div>
      <h2>Ask the RAG agent</h2>
      <p className="muted">
        Each question retrieves passages from the FastAPI tutorial and sends them to a free-tier language model. It
        makes one model call, so the demo limits how many questions one visitor can ask per hour and how many the whole
        site can answer per day. Answers can be wrong: check them against the passages shown below.
      </p>

      {health === undefined && <p className="muted">Checking whether the demo server is running...</p>}
      {health === null && (
        <div className="notice">
          <p>
            The live demo needs the API server, and it is not reachable from here: either it is not running, or this
            is the static version of the site. Everything else on this site works from saved results.
          </p>
          <p>
            To run it yourself, start the API from the repository root with{' '}
            <code>python -m uvicorn app.api.main:app --port 8000</code>, then check again.
          </p>
          <button className="btn plain" onClick={check}>Check again</button>
        </div>
      )}

      {health && (
        <form className="ask-form" onSubmit={submit}>
          <textarea
            value={question}
            maxLength={500}
            placeholder="Ask something about the FastAPI tutorial..."
            onChange={(e) => setQuestion(e.target.value)}
          />
          <div className="chips">
            {EXAMPLES.map((example) => (
              <button type="button" key={example} className="chip" onClick={() => setQuestion(example)}>
                {example}
              </button>
            ))}
          </div>
          <div className="ask-row">
            {Object.keys(PRESET_HELP).map((name) => (
              <button
                type="button" key={name} title={PRESET_HELP[name]}
                className={preset === name ? 'chip active' : 'chip'} onClick={() => setPreset(name)}
              >
                {name}
              </button>
            ))}
            <button className="btn" type="submit" disabled={busy || question.trim().length < 3}>
              {busy ? 'Thinking...' : 'Ask'}
            </button>
          </div>
          <p className="muted">{PRESET_HELP[preset]}</p>
        </form>
      )}

      {error && <p className="error">{error}</p>}

      {result && (
        <div>
          <h3>Answer</h3>
          <pre className="answer">{result.answer}</pre>
          <p className="muted">
            {configLabel(result.config)} | {result.prompt_tokens} prompt tokens | retrieval{' '}
            {fmt(result.timings?.retrieve)} s, generation {fmt(result.timings?.generate)} s
          </p>
          <h3>Passages the model saw</h3>
          <p className="muted">Scores are {scoreKind(result.config.retriever)}.</p>
          {result.sources.map((s) => (
            <details className="q" key={`${s.rank}-${s.doc_id}-${s.start}`}>
              <summary>
                <span className="qid">[{s.rank}]</span>
                <span className="qtext">{s.doc_id}</span>
                <span className="qtype">chars {s.start}-{s.end} | score {s.score}</span>
              </summary>
              <div className="source-text">{s.text}</div>
            </details>
          ))}
        </div>
      )}
    </div>
  )
}