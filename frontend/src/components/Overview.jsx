import { configLabel, fmt, pct, splitRuns, summarizeTest, tokenChange } from '../lib/data.js'

function signed(x) {
  return `${x > 0 ? '+' : ''}${Math.round(x)}%`
}

export default function Overview({ data }) {
  const test = summarizeTest(data.runs)
  const baseline = test.find((t) => t.name === 'baseline')
  const dev = splitRuns(data.runs, 'dev')

  return (
    <div>
      <section>
        <h2>Held-out test split</h2>
        <p className="muted">
          Each config was evaluated once per replicate on questions the optimizer never saw. A difference of one
          question is inside the run-to-run noise measured on the dev split.
        </p>
        <div className="cards">
          {test.map((t) => {
            const change = tokenChange(t, baseline)
            return (
              <div className="card" key={t.name}>
                <div className="card-title">{t.name}</div>
                <div className="big">
                  {t.passes.join(', ')}
                  <span className="muted"> / {t.nQuestions}</span>
                </div>
                <div className="muted">passes per replicate</div>
                <div className="card-foot">
                  {Math.round(t.tokens).toLocaleString()} prompt tokens
                  {t.name !== 'baseline' && change != null && <strong> ({signed(change)})</strong>}
                </div>
              </div>
            )
          })}
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Config</th><th>Replicates</th><th>Mean passes</th><th>recall@k</th>
                <th>Answerable correct</th><th>Faithful</th><th>Declined unanswerable</th><th>Mean prompt tokens</th>
              </tr>
            </thead>
            <tbody>
              {test.map((t) => (
                <tr key={t.name}>
                  <td>{t.name}</td><td>{t.replicates}</td><td>{fmt(t.meanPasses, 1)} / {t.nQuestions}</td>
                  <td>{fmt(t.recall)}</td><td>{pct(t.correct)}</td><td>{pct(t.faithful)}</td>
                  <td>{pct(t.decline)}</td><td>{Math.round(t.tokens).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2>Dev split runs</h2>
        <p className="muted">
          Runs 1 and 2 used Llama 3 8B, runs 3 and 5 used Gemini. Retrieval numbers are identical across prompts
          because the prompt only changes generation. The optimizer tuned against these questions.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Run</th><th>Generator</th><th>Config</th><th>Passes</th><th>recall@k</th>
                <th>Answerable correct</th><th>Faithful</th><th>Declined unanswerable</th><th>Median latency</th>
              </tr>
            </thead>
            <tbody>
              {dev.map((r) => (
                <tr key={r.id}>
                  <td>{r.id}</td><td>{r.generator_model}</td><td>{configLabel(r)}</td>
                  <td>{r.passes} / {r.n_questions}</td><td>{fmt(r.recall_at_k)}</td>
                  <td>{pct(r.answer_correctness)}</td><td>{pct(r.faithfulness)}</td>
                  <td>{pct(r.proper_refusal)}</td><td>{Math.round(r.median_latency_s)} s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}