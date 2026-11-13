import { useEffect, useState } from 'react'
import FailureExplorer from './components/FailureExplorer.jsx'
import Overview from './components/Overview.jsx'
import { loadResults } from './lib/api.js'
import './styles.css'
import './explorer.css'

const TABS = [['overview', 'Overview'], ['failures', 'Failure explorer']]

export default function App() {
  const [state, setState] = useState({ status: 'loading' })
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    loadResults()
      .then((result) => setState({ status: 'ready', ...result }))
      .catch((error) => setState({ status: 'error', message: error.message }))
  }, [])

  return (
    <main>
      <header>
        <h1>Self-Tuning RAG</h1>
        <p className="muted">
          A RAG agent, and an optimizer that diagnoses its failures and tunes retrieval and generation settings.
        </p>
        {state.status === 'ready' && (
          <span className="badge">
            data: {state.source}, exported {state.data.generated}
          </span>
        )}
      </header>
      {state.status === 'loading' && <p>Loading results...</p>}
      {state.status === 'error' && <p className="error">{state.message}</p>}
      {state.status === 'ready' && (
        <>
          <nav className="tabs">
            {TABS.map(([id, label]) => (
              <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>
                {label}
              </button>
            ))}
          </nav>
          {tab === 'overview' && <Overview data={state.data} />}
          {tab === 'failures' && <FailureExplorer data={state.data} />}
        </>
      )}
    </main>
  )
}