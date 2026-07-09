import { useCallback, useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import { api } from '../api'
import type { FLServerState } from '../types'
import { FL_THEME as T } from '../theme'
import { useConfig, useClients } from '../useConfig'

const REFRESH_MS = 1000

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Trace = any

function FLTrustChart({ hist }: { hist: FLServerState['fltrust_scores_hist'] }) {
  if (!hist || hist.length === 0) return (
    <div style={{ color: T.dim, fontSize: '0.76rem', padding: '0.5rem 0' }}>Run FL simulation to see FLTrust scores.</div>
  )
  const rounds = hist.map(h => h.round)
  const clients = Object.keys(hist[0]?.scores ?? {})
  const traces: Trace[] = clients.map(c => ({
    x: rounds,
    y: hist.map(h => h.scores[c] ?? 0),
    name: c,
    mode: 'lines+markers',
    line: { width: 2 },
    marker: { size: 5 },
  }))
  return (
    <Plot
      data={traces}
      layout={{
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: T.text, size: 10 },
        legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 9 }, orientation: 'h', y: -0.3 },
        xaxis: { showgrid: false, title: { text: 'Round', standoff: 4 }, color: T.dim, tickfont: { size: 9 }, dtick: 1 },
        yaxis: { showgrid: true, gridcolor: 'rgba(48,54,61,0.5)', range: [0, 1.05], title: { text: 'Trust Score', standoff: 4 }, color: T.dim, tickfont: { size: 9 } },
        height: 200,
        margin: { l: 44, r: 10, t: 8, b: 50 },
        autosize: true,
        hoverlabel: { bgcolor: '#161b22', bordercolor: '#30363d', font: { size: 11 } },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}

export default function FLServerPage() {
  const cfg       = useConfig()
  const FL_CLIENTS = useClients(cfg)

  const [state, setState] = useState<FLServerState | null>(null)
  const [busy,  setBusy]  = useState(false)

  const refresh = useCallback(async () => {
    try { setState(await api.getFLState()) }
    catch { /* API starting */ }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, REFRESH_MS)
    return () => clearInterval(id)
  }, [refresh])

  const runFL = async () => {
    setBusy(true)
    try {
      const res = await api.runFLSimulation()
      setState(res.state)
    } finally { setBusy(false) }
  }

  if (!state) return (
    <div style={{ padding: '3rem', color: T.dim, textAlign: 'center' }}>
      <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>⚙️</div>
      <div>Loading FL Server Console…</div>
    </div>
  )

  const runColor = state.fl_running ? T.yellow : state.fl_done ? T.green : T.dim
  const readinessMap = new Map(state.orgs.map(o => [o.key, o]))
  const totalRounds  = state.total_rounds || cfg.fl_num_rounds || 3
  const roundPct     = (state.current_round / totalRounds) * 100

  return (
    <>
      {/* ── Header ───────────────────────────────────────────────── */}
      <div style={{
        background: 'linear-gradient(135deg, #0c1828 0%, #080f1e 100%)',
        border: `1px solid ${T.border}`, borderRadius: 14,
        padding: '0.85rem 1.3rem', display: 'flex',
        justifyContent: 'space-between', alignItems: 'center',
        marginBottom: '0.75rem', position: 'relative', overflow: 'hidden',
      }}>
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: 'linear-gradient(90deg, transparent, #58d1e8, #388bfd, transparent)' }} />
        <div>
          <span style={{ fontSize: '1.3em', fontWeight: 800, color: T.cyan }}>⚙️ FL Server Console</span>
          <span style={{ color: T.dim, marginLeft: '0.8em', fontSize: '0.78em' }}>
            FLTrust-Aggregated · Blockchain-Audited · {FL_CLIENTS.length} Federated Clients
          </span>
        </div>
        <div>
          <span style={{ color: runColor, fontWeight: 700 }}>
            <span style={{
              display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
              backgroundColor: runColor, marginRight: 5,
              animation: state.fl_running ? 'blink-dot 1s infinite' : 'none',
            }} />
            {state.run_state}
          </span>
          <span style={{ color: T.dim, marginLeft: '1em', fontSize: '0.74em' }}>
            Round {state.current_round}/{totalRounds}
          </span>
        </div>
      </div>

      {/* ── Metrics ──────────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: '0.75rem', marginBottom: '0.9rem' }}>
        {[
          ['Rounds Done',     `${state.current_round} / ${totalRounds}`, T.cyan],
          ['FLTrust Trusted', state.fl_done ? `${FL_CLIENTS.length} / ${FL_CLIENTS.length}` : '— / —', T.green],
          ['Global Version',  state.global_version ?? '—', T.purple],
          ['Status',          state.run_state, runColor],
        ].map(([label, val, col]) => (
          <div key={String(label)} style={{ background: T.panel, border: `1px solid ${T.border}`, borderRadius: 12, padding: '0.85rem 1rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.67rem', color: T.dim, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>{label}</div>
            <div style={{ fontSize: '1.3rem', fontWeight: 800, color: String(col), marginTop: '0.3rem' }}>{val}</div>
          </div>
        ))}
      </div>

      {/* Round progress bar */}
      {state.fl_running && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
          <span style={{ fontSize: '0.72rem', color: T.dim, flexShrink: 0 }}>Progress</span>
          <div className="progress-bar-track">
            <div className="progress-bar-fill" style={{ width: `${roundPct}%` }} />
          </div>
          <span style={{ fontSize: '0.72rem', color: T.cyan, flexShrink: 0 }}>{roundPct.toFixed(0)}%</span>
        </div>
      )}

      <hr style={{ border: 'none', borderTop: `1px solid ${T.border}`, margin: '0.75rem 0' }} />

      {/* ── Client Cards ─────────────────────────────────────────── */}
      <h4 style={{ color: T.green, marginBottom: '0.55rem', fontSize: '0.85rem', fontWeight: 700 }}>
        📡 Federation Clients — Node Readiness ({FL_CLIENTS.length} Orgs)
      </h4>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(FL_CLIENTS.length, 5)}, 1fr)`, gap: '0.65rem', marginBottom: '1rem' }}>
        {FL_CLIENTS.map((cl) => {
          const orgData       = readinessMap.get(cl.key)
          const isReady       = orgData?.ready ?? false
          const isAttacked    = orgData?.under_attack ?? false
          const isByzantine   = state.byzantine_org === cl.id
          const isQuarantined = state.quarantined_orgs?.includes(cl.id)
          const cardState     = isAttacked || isByzantine ? 'danger' : isReady ? 'ready' : 'idle'
          const borderCol     = cardState === 'danger' ? '#f85149' : cardState === 'ready' ? cl.color : T.border
          void isQuarantined

          return (
            <div key={cl.key} style={{
              background: `linear-gradient(145deg, ${T.panel} 0%, #0b1220 100%)`,
              border: `1px solid ${borderCol}`, borderRadius: 12,
              padding: '0.8rem 0.85rem', transition: 'box-shadow 0.2s',
              boxShadow: cardState !== 'idle' ? `0 0 16px ${borderCol}33` : 'none',
              position: 'relative', overflow: 'hidden',
            }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: borderCol, opacity: cardState === 'idle' ? 0.3 : 0.8 }} />
              <div style={{ fontSize: '1.3em', marginBottom: '0.25rem' }}>{cl.icon}</div>
              <div style={{ fontWeight: 700, fontSize: '0.85rem', color: T.text }}>{cl.label}</div>
              <div style={{ fontSize: '0.66rem', color: T.dim, marginTop: '0.12rem', fontFamily: 'monospace' }}>{cl.net}</div>
              <div style={{ marginTop: '0.45rem', fontSize: '0.7rem', fontWeight: 700, color: cardState === 'danger' ? '#f85149' : cardState === 'ready' ? cl.color : T.dim }}>
                {isAttacked || isByzantine ? '🚨 QUARANTINED' : isReady ? '✅ READY' : '⏸ IDLE'}
              </div>
              {isByzantine && (
                <div style={{ marginTop: '0.3rem', fontSize: '0.63rem', fontWeight: 700, color: '#f85149', background: '#f8514918', borderRadius: 4, padding: '2px 5px', display: 'inline-block' }}>
                  ⚠ BYZANTINE
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* ── Aggregation Pipeline ─────────────────────────────────── */}
      <h4 style={{ color: T.cyan, marginBottom: '0.45rem', fontSize: '0.85rem', fontWeight: 700 }}>🔄 Aggregation Pipeline</h4>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${state.pipe_steps.length}, 1fr)`, gap: '0.45rem', marginBottom: '0.9rem' }}>
        {state.pipe_steps.map((step, i) => {
          const done   = step.state === 2
          const active = step.state === 1
          return (
            <div key={i} style={{
              background: done ? '#0f2117' : active ? '#1a2a38' : T.panel,
              border: `1px solid ${done ? T.green : active ? T.cyan : T.border}`,
              borderRadius: 10, padding: '0.55rem 0.45rem', textAlign: 'center', fontSize: '0.74em',
              opacity: (!done && !active) ? 0.4 : 1,
              transition: 'all 0.3s',
              boxShadow: done ? `0 0 10px ${T.green}33` : active ? `0 0 10px ${T.cyan}33` : 'none',
            }}>
              <div style={{ fontSize: '1.25em', marginBottom: '0.18rem' }}>{step.icon}</div>
              <div style={{ whiteSpace: 'pre-line', color: done ? T.green : active ? T.cyan : T.dim, fontWeight: done || active ? 600 : 400 }}>{step.label}</div>
              {done   && <div style={{ fontSize: '0.62em', color: T.green,  marginTop: '0.18rem' }}>✓ Done</div>}
              {active && <div style={{ fontSize: '0.62em', color: T.cyan,   marginTop: '0.18rem' }}>▶ Active</div>}
            </div>
          )
        })}
      </div>

      {/* ── Run Button ───────────────────────────────────────────── */}
      <button
        style={{
          width: '100%', marginBottom: '0.9rem', padding: '0.7rem', borderRadius: 10,
          border: `1px solid ${T.cyan}`,
          background: state.fl_running ? 'rgba(88,209,232,0.04)' : 'rgba(88,209,232,0.08)',
          color: T.cyan, fontFamily: 'Inter, sans-serif', fontWeight: 700, fontSize: '0.88rem',
          cursor: busy || state.fl_running ? 'not-allowed' : 'pointer',
          opacity: busy || state.fl_running ? 0.6 : 1,
          transition: 'all 0.2s',
        }}
        disabled={busy || state.fl_running}
        onClick={runFL}
      >
        {state.fl_running ? `⏳ FL Running across ${FL_CLIENTS.length} clients…` : `🚀 Start FL Simulation (${FL_CLIENTS.length} Clients · ${totalRounds} Rounds)`}
      </button>

      {/* ── FLTrust History + Ledger + Log ───────────────────────── */}
      {state.global_hash && (
        <div style={{ background: T.panel, border: `1px solid ${T.border}`, borderRadius: 10, padding: '0.6rem 0.9rem', marginBottom: '0.9rem', fontFamily: 'monospace', fontSize: '0.74rem' }}>
          <span style={{ color: T.dim }}>Global model hash: </span>
          <span style={{ color: T.purple }}>{state.global_hash}</span>
        </div>
      )}

      {/* FLTrust per-client score history */}
      {state.fltrust_scores_hist && state.fltrust_scores_hist.length > 0 && (
        <div style={{ background: T.panel, border: `1px solid ${T.border}`, borderRadius: 12, padding: '1rem', marginBottom: '0.9rem' }}>
          <h4 style={{ color: T.cyan, fontSize: '0.83rem', marginBottom: '0.4rem', fontWeight: 700 }}>📈 FLTrust Per-Client Score History</h4>
          <FLTrustChart hist={state.fltrust_scores_hist} />
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
        <div style={{ background: T.panel, border: `1px solid ${T.border}`, borderRadius: 12, padding: '1rem' }}>
          <h4 style={{ color: T.purple, fontSize: '0.83rem', marginBottom: '0.45rem', fontWeight: 700 }}>⛓ Hash Ledger</h4>
          {state.hash_ledger.length === 0 ? (
            <span style={{ color: T.dim, fontSize: '0.76rem' }}>No hashes minted yet.</span>
          ) : state.hash_ledger.map((e, i) => (
            <div key={i} style={{ borderLeft: `3px solid ${T.purple}`, padding: '0.3rem 0.55rem', margin: '0.22rem 0', fontSize: '0.71em', fontFamily: 'monospace', color: T.text }}>
              R{e.round} {e.version}: <span style={{ color: T.purple }}>{e.hash.slice(0, 28)}…</span>
              {' @ '}<span style={{ color: T.dim }}>{e.time}</span>
            </div>
          ))}
        </div>
        <div style={{ background: T.panel, border: `1px solid ${T.border}`, borderRadius: 12, padding: '1rem' }}>
          <h4 style={{ color: T.dim, fontSize: '0.83rem', marginBottom: '0.45rem', fontWeight: 700 }}>📋 FL Log</h4>
          <div style={{ maxHeight: 260, overflowY: 'auto' }}>
            {state.fl_log.length === 0 ? (
              <span style={{ color: T.dim, fontSize: '0.76rem' }}>Waiting for FL run…</span>
            ) : state.fl_log.map((line, i) => (
              <div key={i} style={{
                color: line.includes('✅') ? T.green : line.includes('❌') ? '#f85149' : line.includes('Round') ? T.cyan : T.dim,
                fontSize: '0.72em', fontFamily: 'monospace', padding: '0.18rem 0',
                borderBottom: `1px solid ${T.border}44`,
              }}>{line}</div>
            ))}
          </div>
        </div>
      </div>
    </>
  )
}
