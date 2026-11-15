import { useMemo, useState } from 'react'
import {
  CartesianGrid, ReferenceDot, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from 'recharts'
import { fmt } from '../lib/data.js'
import {
  bestCoverage, configsBeating, filterSpace, frontier, presetPoints, spaceLabel, toPoint, tokenDelta,
} from '../lib/space.js'
import '../space.css'

const DENSE = '#4b5bd7'
const HYBRID = '#e08a00'
const FRONTIER = '#1b7f4b'
const MARK = '#d1344a'

function PointTip({ active, payload }) {
  if (!active || !payload?.length) return null
  const p = payload[0].payload
  return (
    <div className="tip">
      <div><strong>{spaceLabel(p)}</strong></div>
      <div>covers {p.covered} of {p.n_answerable} answerable dev questions</div>
      <div>about {p.prompt_tokens_estimate} prompt tokens (estimate){p.pareto ? ' - on the frontier' : ''}</div>
    </div>
  )
}

export default function SpaceChart({ data }) {
  const space = data.retrieval_space ?? []
  const [retriever, setRetriever] = useState('all')
  const [chunkSize, setChunkSize] = useState('all')

  const shown = useMemo(() => filterSpace(space, { retriever, chunkSize }).map(toPoint), [space, retriever, chunkSize])
  const front = useMemo(() => frontier(space), [space])
  const presets = useMemo(() => presetPoints(space), [space])

  if (!space.length) return <p className="muted">No retrieval-space data in the exported results.</p>

  const dense = shown.filter((p) => p.retriever === 'dense')
  const hybrid = shown.filter((p) => p.retriever === 'hybrid')
  const chunkSizes = [...new Set(space.map((p) => p.chunk_size))].sort((a, b) => a - b)
  const nAnswerable = space[0].n_answerable
  const tick = { fill: 'currentColor', fontSize: 12 }

  return (
    <div>
      <h2>Retrieval space</h2>
      <p className="muted">
        All {space.length} valid retrieval configurations, scored for free on the dev split: how many of the{' '}
        {nAnswerable} answerable questions have all their gold evidence inside the prompt. The best any
        configuration reaches is {bestCoverage(space)}. The green line is the cost frontier: nothing covers more with
        a shorter prompt. Many dots overlap, because several settings give identical results.
      </p>

      <div className="controls">
        <label>
          Retriever
          <select value={retriever} onChange={(e) => setRetriever(e.target.value)}>
            <option value="all">all</option><option value="dense">dense</option><option value="hybrid">hybrid</option>
          </select>
        </label>
        <label>
          Chunk size
          <select value={chunkSize} onChange={(e) => setChunkSize(e.target.value)}>
            <option value="all">all</option>
            {chunkSizes.map((c) => (<option key={c} value={c}>{c}</option>))}
          </select>
        </label>
      </div>

      <div className="legend">
        <span style={{ '--dot': DENSE }}>dense</span>
        <span style={{ '--dot': HYBRID }}>hybrid</span>
        <span className="line" style={{ '--dot': FRONTIER }}>frontier of the full space</span>
        <span style={{ '--dot': MARK }}>evaluated configs</span>
      </div>

      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={380}>
          <ScatterChart margin={{ top: 24, right: 28, bottom: 30, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis
              type="number" dataKey="x" name="prompt tokens" tick={tick}
              domain={['dataMin - 100', 'dataMax + 100']}
              label={{ value: 'estimated prompt tokens (upper bound)', position: 'insideBottom', offset: -18,
                fill: 'currentColor', fontSize: 12 }}
            />
            <YAxis
              type="number" dataKey="y" name="covered" tick={tick} allowDecimals={false}
              domain={['dataMin - 1', 'dataMax + 1']}
              label={{ value: 'questions covered', angle: -90, position: 'insideLeft', fill: 'currentColor',
                fontSize: 12 }}
            />
            <Tooltip content={<PointTip />} cursor={{ strokeDasharray: '3 3' }} />
            <Scatter name="dense" data={dense} fill={DENSE} fillOpacity={0.45} isAnimationActive={false} />
            <Scatter name="hybrid" data={hybrid} fill={HYBRID} fillOpacity={0.45} isAnimationActive={false} />
            <Scatter
              name="frontier" data={front} fill={FRONTIER} line={{ stroke: FRONTIER, strokeWidth: 2 }}
              lineType="joint" isAnimationActive={false}
            />
            {Object.entries(presets).map(([name, p]) => (
              <ReferenceDot
                key={name} x={p.x} y={p.y} r={8} fill="none" stroke={MARK} strokeWidth={2.5}
                ifOverflow="extendDomain" label={{ value: name, position: 'top', fill: MARK, fontSize: 12 }}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="cards">
        {Object.entries(presets).map(([name, p]) => {
          const change = name === 'baseline' ? null : tokenDelta(p, presets.baseline)
          return (
            <div className="card" key={name}>
              <div className="card-title">{name}</div>
              <div className="big">{p.y}<span className="muted"> / {p.n_answerable}</span></div>
              <div className="muted">{spaceLabel(p)}</div>
              <div className="card-foot">
                about {p.x.toLocaleString()} prompt tokens
                {change != null && <strong> ({change > 0 ? '+' : ''}{Math.round(change)}%)</strong>}
                <br />{configsBeating(space, p)} configurations cover more questions
              </div>
            </div>
          )
        })}
      </div>

      <h3>Frontier</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Retriever</th><th>Chunk / overlap</th><th>top_k</th><th>Covered</th><th>Estimated prompt tokens</th></tr>
          </thead>
          <tbody>
            {front.map((p, i) => (
              <tr key={i}>
                <td>{p.retriever}</td><td>{p.chunk_size}/{p.overlap}</td><td>{p.top_k}</td>
                <td>{p.covered} ({fmt(p.covered / p.n_answerable)})</td><td>{p.x.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted">
        Caveats: coverage is a proxy for answer quality (on dev it predicted +5 questions for the optimizer's move,
        and the real gain was a net +2). The token axis is a formula. Measured mean prompt tokens on the test split
        were 1,028 for the baseline, 1,515 for the optimized config and 814 for the cheap one. These are dev
        questions the search was tuned on.
      </p>
    </div>
  )
}