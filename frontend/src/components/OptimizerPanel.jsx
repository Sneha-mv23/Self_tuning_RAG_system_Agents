import { useEffect, useMemo, useRef, useState } from 'react'
import { checkApi } from '../lib/ask.js'
import { configLabel } from '../lib/data.js'
import {
  api, describeProposal, isActive, startBody, streamEvents, validateForm,
} from '../lib/jobs.js'
import '../jobs.css'

const POLL_MS = 1500

export default function OptimizerPanel({ data }) {
  const runs = useMemo(() => data.runs.filter((r) => r.split === 'dev'), [data])
  const [health, setHealth] = useState(undefined)
  const [token, setToken] = useState('') // held in memory only: never stored anywhere
  const [form, setForm] = useState({
    startRun: String(runs.length ? runs[runs.length - 1].id : ''),
    budget: '150', maxExperiments: '2', dryRun: true, approve: true,
  })
  const [job, setJob] = useState(null)
  const [events, setEvents] = useState([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const stop = useRef(null)
  const alive = useRef(true)

  useEffect(() => {
    alive.current = true
    checkApi().then((h) => alive.current && setHealth(h))
    return () => {
      alive.current = false
      stop.current?.abort()
    }
  }, [])

  const set = (key) => (e) =>
    setForm({ ...form, [key]: e.target.type === 'checkbox' ? e.target.checked : e.target.value })

  async function refresh(id) {
    try {
      const view = await api(`/api/jobs/${id}`, { token })
      if (alive.current) setJob(view)
      return view
    } catch (err) {
      if (alive.current) setError(err.message)
      return null
    }
  }

  async function follow(id) {
    stop.current?.abort()
    const controller = new AbortController()
    stop.current = controller
    setEvents([])
    const seen = new Set()
    const stream = streamEvents(id, {
      token, signal: controller.signal,
      onEvent: (ev) => {
        if (seen.has(ev.i)) return
        seen.add(ev.i)
        setEvents((prev) => [...prev, ev])
      },
    }).catch((err) => {
      if (err?.name !== 'AbortError' && alive.current) setError(err.message)
    })
    // poll the job view too, so the approval box and the final result appear promptly
    for (;;) {
      const view = await refresh(id)
      if (!view || !alive.current || controller.signal.aborted || !isActive(view.status)) break
      await new Promise((resolve) => setTimeout(resolve, POLL_MS))
    }
    await Promise.race([stream, new Promise((resolve) => setTimeout(resolve, 3000))])
    await refresh(id)
  }

  async function start(event) {
    event.preventDefault()
    const problem = validateForm(form, runs.map((r) => r.id))
    if (problem) return setError(problem)
    if (!token) return setError('Enter the admin token first.')
    if (!form.dryRun && !window.confirm('This run can spend real Gemini API quota. Continue?')) return
    setBusy(true)
    setError('')
    try {
      const view = await api('/api/jobs', { token, method: 'POST', body: startBody(form) })
      setJob(view)
      await follow(view.id)
    } catch (err) {
      setError(err.message)
    } finally {
      if (alive.current) setBusy(false)
    }
  }

  async function decide(approve) {
    if (!job) return
    setError('')
    try {
      await api(`/api/jobs/${job.id}/decision`, { token, method: 'POST', body: { approve } })
      await refresh(job.id)
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div>
      <h2>Optimizer</h2>
      <p className="muted">
        Starts an optimizer session on the dev split. Each round diagnoses the current run's failures, screens the
        candidate changes for free with retrieval metrics, and runs only the promising one. A change is accepted only
        if it fixes at least two more questions than it breaks. This panel is for the person running the server.
      </p>

      {health === undefined && <p className="muted">Checking the server...</p>}
      {health === null && (
        <div className="notice"><p>The API server is not reachable, so there is nothing to control from here.</p></div>
      )}
      {health && !health.jobs_enabled && (
        <div className="notice">
          <p>
            Optimizer jobs are switched off on this server. They need an admin token set in the server's{' '}
            <code>.env</code>, and they are meant for local use.
          </p>
        </div>
      )}

      {health?.jobs_enabled && (
        <form onSubmit={start}>
          <div className="fields">
            <label>
              Admin token (kept in memory only)
              <input type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} />
            </label>
            <label>
              Start from run
              <select value={form.startRun} onChange={set('startRun')}>
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>Run {r.id}: {r.generator_model} | {configLabel(r)}</option>
                ))}
              </select>
            </label>
            <label>
              API call budget
              <input type="number" min="1" max="250" value={form.budget} onChange={set('budget')} />
            </label>
            <label>
              Max experiments
              <input type="number" min="1" max="3" value={form.maxExperiments} onChange={set('maxExperiments')} />
            </label>
          </div>
          <div className="fields">
            <label className="check">
              <input type="checkbox" checked={form.dryRun} onChange={set('dryRun')} /> Dry run (diagnose and estimate,
              call no API)
            </label>
            <label className="check">
              <input type="checkbox" checked={form.approve} onChange={set('approve')} /> Ask me before each experiment
            </label>
            <button className="btn" type="submit" disabled={busy || (job && isActive(job.status))}>
              {busy ? 'Running...' : 'Start'}
            </button>
          </div>
        </form>
      )}

      {error && <p className="error">{error}</p>}

      {job && (
        <div>
          <p>
            Job <code>{job.id}</code>: <strong>{job.status.replace('_', ' ')}</strong>
          </p>

          {job.status === 'awaiting_approval' && job.proposal && (
            <div className="decision">
              <strong>Run this experiment?</strong>
              <div>{describeProposal(job.proposal)}</div>
              <div className="muted">No answer within 15 minutes counts as a rejection.</div>
              <div className="decision-row">
                <button className="btn" onClick={() => decide(true)}>Approve</button>
                <button className="btn danger" onClick={() => decide(false)}>Reject</button>
              </div>
            </div>
          )}

          <div className="log">
            {events.length === 0 && <span className="dim">waiting for output...</span>}
            {events.map((ev) => (
              <div key={ev.i} className={ev.kind === 'status' ? 'status' : ''}>
                {ev.kind === 'status' ? `-- ${ev.text.replace('_', ' ')} --` : ev.text}
              </div>
            ))}
          </div>

          {job.status === 'finished' && job.result && (
            <div className="card">
              <div className="card-title">Result</div>
              <div className="big">
                {job.result.start_passes} to {job.result.final_passes}
                <span className="muted"> / {job.result.n_questions} pass</span>
              </div>
              <div className="muted">
                run {job.result.start_run} to run {job.result.final_run} | {job.result.spent} of {job.result.budget} API
                calls | {job.result.stop}
              </div>
            </div>
          )}
          {job.status === 'failed' && <p className="error">The job failed: {job.error}</p>}
        </div>
      )}
    </div>
  )
}