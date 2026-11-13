import { useEffect, useState } from 'react'
import Overview from './components/Overview.jsx'
import { loadResults } from './lib/api.js'
import './styles.css'

export default function App() {
  const [state, setState] = useState({ status: 'loading' })

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
      {state.status === 'ready' && <Overview data={state.data} />}
    </main>
  )
}