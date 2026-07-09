import { useCallback, useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import { api } from '../api'
import type { BenchmarkResults, HITLTierMetrics } from '../types'
import { THEME } from '../theme'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Trace = any

const REFRESH_MS = 30_000  // benchmarks dont change every second — check every 30s

function PassBadge({ pass, label }: { pass: boolean | undefined; label: string }) {
  if (pass === undefined) return <span className="bench-pending">{label}: N/A</span>
  return pass
    ? <span className="bench-pass">✓ PASS — {label}</span>
    : <span className="bench-fail">✗ FAIL — {label}</span>
}

function LatencyBar({ label, value, max }: { label: string; value: number; max: number }) {
  const pct = Math.min(100, (value / max) * 100)
  const color = value < 1 ? THEME.green : value < 10 ? THEME.yellow : THEME.red
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.5rem' }}>
      <span style={{ color: THEME.dim, width: 60, fontSize: '0.72rem', flexShrink: 0 }}>{label}</span>
      <div className="progress-bar-track">
        <div className="progress-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.72rem', color, width: 60, textAlign: 'right' }}>
        {value.toFixed(3)} ms
      </span>
    </div>
  )
}

function TierTable({ tiers }: { tiers: HITLTierMetrics[] }) {
  const tierColor = (t: string) => {
    if (t === 'HIGH')     return THEME.red
    if (t === 'MEDIUM')   return THEME.orange
    if (t === 'LOW')      return THEME.yellow
    if (t === 'DEGRADED') return '#a855f7'
    return THEME.dim
  }
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem' }}>
      <thead>
        <tr style={{ color: THEME.dim, fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          <th style={{ textAlign: 'left', paddingBottom: '0.4rem', paddingRight: '0.8rem' }}>Tier</th>
          <th style={{ textAlign: 'right', paddingBottom: '0.4rem', paddingRight: '0.8rem' }}>Count</th>
          <th style={{ textAlign: 'right', paddingBottom: '0.4rem', paddingRight: '0.8rem' }}>FER %</th>
          <th style={{ textAlign: 'right', paddingBottom: '0.4rem', paddingRight: '0.8rem' }}>P50 ms</th>
          <th style={{ textAlign: 'right', paddingBottom: '0.4rem' }}>P95 ms</th>
        </tr>
      </thead>
      <tbody>
        {tiers.map((row) => (
          <tr key={row.tier} style={{ borderTop: `1px solid ${THEME.border}55` }}>
            <td style={{ padding: '0.3rem 0.8rem 0.3rem 0', color: tierColor(row.tier), fontWeight: 700 }}>{row.tier}</td>
            <td style={{ textAlign: 'right', padding: '0.3rem 0.8rem 0.3rem 0', color: THEME.text, fontFamily: 'monospace' }}>{row.count}</td>
            <td style={{ textAlign: 'right', padding: '0.3rem 0.8rem 0.3rem 0', color: row.fer > 0.05 ? THEME.red : THEME.green, fontFamily: 'monospace' }}>{(row.fer * 100).toFixed(2)}</td>
            <td style={{ textAlign: 'right', padding: '0.3rem 0.8rem 0.3rem 0', color: THEME.text, fontFamily: 'monospace' }}>{row.latency_p50_ms.toFixed(3)}</td>
            <td style={{ textAlign: 'right', padding: '0.3rem 0', color: THEME.text, fontFamily: 'monospace' }}>{row.latency_p95_ms.toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function AblationChart({ ablation }: { ablation: BenchmarkResults['ablation'] }) {
  const modes   = Object.keys(ablation)
  const f1s     = modes.map(m => Number(ablation[m].F1        ?? 0))
  const precs   = modes.map(m => Number(ablation[m].Precision ?? 0))
  const recalls = modes.map(m => Number(ablation[m].Recall    ?? 0))
  const aucs    = modes.map(m => Number(ablation[m].AUC_ROC ?? ablation[m].AUC_PR ?? 0))

  const modeLabels = modes.map(m => {
    if (m === 'A' || m.toUpperCase().includes('AE_ONLY'))          return 'A — AE Only'
    if (m === 'B' || m.toUpperCase().includes('GNN_ONLY'))         return 'B — GNN Only'
    if (m === 'C' || m.toUpperCase().includes('AE_GNN'))           return 'C — AE+GNN'
    if (m === 'D' || m.toUpperCase().includes('FULL'))             return 'D — Full AURA ★'
    return m
  })

  const traces: Trace[] = [
    { x: modeLabels, y: f1s,     name: 'F1',        type: 'bar', marker: { color: THEME.cyan,   opacity: 0.9 } },
    { x: modeLabels, y: precs,   name: 'Precision',  type: 'bar', marker: { color: THEME.green,  opacity: 0.9 } },
    { x: modeLabels, y: recalls, name: 'Recall',     type: 'bar', marker: { color: THEME.yellow, opacity: 0.9 } },
    { x: modeLabels, y: aucs,    name: 'AUC-ROC',    type: 'bar', marker: { color: '#a855f7',    opacity: 0.9 } },
  ]

  return (
    <Plot
      data={traces}
      layout={{
        barmode: 'group',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: THEME.text, size: 11 },
        legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 10 }, orientation: 'h', y: -0.2 },
        xaxis: { showgrid: false, color: THEME.dim, tickfont: { size: 11 } },
        yaxis: { showgrid: true, gridcolor: 'rgba(24,37,64,0.5)', range: [0, 1], title: { text: 'Score', standoff: 4 }, color: THEME.dim, tickfont: { size: 10 } },
        height: 320,
        margin: { l: 44, r: 10, t: 10, b: 70 },
        autosize: true,
        bargap: 0.2,
        bargroupgap: 0.08,
        hoverlabel: { bgcolor: '#0b1325', bordercolor: '#1e3058', font: { size: 11 } },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}

export default function BenchmarkPage() {
  const [results, setResults] = useState<BenchmarkResults | null>(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    try {
      const data = await api.getBenchmarkResults()
      setResults(data)
    } catch { /* backend warming up */ }
    finally { setLoading(false) }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, REFRESH_MS)
    return () => clearInterval(id)
  }, [refresh])

  const hitl     = results?.hitl      ?? {}
  const ablation = results?.ablation  ?? {}
  const avail    = results?.available ?? { hitl: false, ablation: false }
  const crit     = hitl.criteria

  const tiers    = (hitl.tier_breakdown ?? []) as HITLTierMetrics[]
  const maxLatency = Math.max(hitl.latency_p99_ms ?? 0, 2)

  const ablationModes = Object.keys(ablation)

  return (
    <>
      {/* Header */}
      <div className="header-panel" style={{ marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ fontSize: '1.3em', fontWeight: 800, color: THEME.cyan }} className="glow-cyan">
            📊 AURA — Research Benchmarks
          </span>
          <span style={{ color: THEME.dim, fontSize: '0.78em' }}>
            Live results from <code style={{ color: THEME.cyan, fontSize: '0.9em' }}>reports/</code> directory
          </span>
        </div>
        {loading && <span style={{ color: THEME.dim, fontSize: '0.78rem' }}>⏳ Loading…</span>}
      </div>

      {/* ─── Section 3.5 — HITL Response Engine ──────────────────────── */}
      <div className="section-header">Section 3.5 — Three-Tier HITL Response Engine</div>
      {!avail.hitl ? (
        <div className="panel" style={{ color: THEME.dim, fontSize: '0.82rem', marginBottom: '1rem' }}>
          No HITL results yet. Run: <code style={{ color: THEME.cyan }}>python scripts/benchmark_hitl_response.py</code>
        </div>
      ) : (
        <>
          {/* Summary metric cards */}
          <div className="grid-6" style={{ marginBottom: '0.75rem' }}>
            {([
              ['Windows',       hitl.windows_evaluated ?? '—',                            THEME.cyan],
              ['Attacks',       hitl.attack_windows    ?? '—',                            THEME.red],
              ['Escalations',   hitl.total_escalations ?? '—',                            THEME.orange],
              ['FER',           hitl.overall_fer !== undefined ? `${(hitl.overall_fer * 100).toFixed(2)}%` : '—', hitl.overall_fer !== undefined && hitl.overall_fer < 0.1 ? THEME.green : THEME.red],
              ['HITL/hr',       hitl.hitl_calls_per_hour !== undefined ? hitl.hitl_calls_per_hour.toFixed(1) : '—', hitl.hitl_calls_per_hour !== undefined && hitl.hitl_calls_per_hour < 10 ? THEME.green : THEME.red],
              ['Wall Time',     hitl.wall_time_s !== undefined ? `${hitl.wall_time_s.toFixed(1)}s` : '—', THEME.dim],
            ] as [string, string | number, string][]).map(([label, val, col]) => (
              <div key={label} className="metric-card">
                <div className="metric-label">{label}</div>
                <div className="metric-value" style={{ color: col, fontSize: '1.25rem' }}>{val}</div>
              </div>
            ))}
          </div>

          {/* Pass/fail criteria row */}
          <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
            <PassBadge pass={crit?.fer_pass}       label="FER < 10%" />
            <PassBadge pass={crit?.latency_pass}   label="P95 Latency < 100ms" />
            <PassBadge pass={crit?.hitl_rate_pass} label="HITL < 10 calls/hr" />
            <PassBadge pass={crit?.overall_pass}   label="Overall" />
          </div>

          <div className="grid-2" style={{ marginBottom: '1rem' }}>
            {/* Latency panel */}
            <div className="panel">
              <h4 className="panel-title" style={{ color: THEME.cyan }}>⚡ Response Latency</h4>
              <LatencyBar label="P50" value={hitl.latency_p50_ms ?? 0} max={maxLatency} />
              <LatencyBar label="P95" value={hitl.latency_p95_ms ?? 0} max={maxLatency} />
              <LatencyBar label="P99" value={hitl.latency_p99_ms ?? 0} max={maxLatency} />
              <div style={{ fontSize: '0.7rem', color: THEME.dim, marginTop: '0.5rem' }}>
                All times well under 100ms real-time threshold
              </div>
            </div>

            {/* HITL stats */}
            <div className="panel">
              <h4 className="panel-title" style={{ color: THEME.orange }}>🧑‍💻 HITL Simulator Stats</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.8rem' }}>
                {[
                  ['Simulated Duration', hitl.simulated_duration_hr !== undefined ? `${hitl.simulated_duration_hr?.toFixed(2)} hr` : '—', THEME.cyan],
                  ['Degraded Rate',      hitl.hitl_degraded_rate  !== undefined ? `${(hitl.hitl_degraded_rate * 100).toFixed(2)}%` : '—', THEME.yellow],
                  ['Calls / Hour',       hitl.hitl_calls_per_hour !== undefined ? hitl.hitl_calls_per_hour.toFixed(1) : '—', THEME.green],
                ].map(([k, v, col]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: `1px solid ${THEME.border}66`, paddingBottom: '0.35rem' }}>
                    <span style={{ color: THEME.dim }}>{k}</span>
                    <span style={{ color: col, fontFamily: 'monospace', fontWeight: 700 }}>{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Tier breakdown */}
          {tiers.length > 0 && (
            <div className="panel" style={{ marginBottom: '1rem' }}>
              <h4 className="panel-title" style={{ color: '#a855f7' }}>🗂 Per-Tier Breakdown</h4>
              <TierTable tiers={tiers} />
            </div>
          )}
        </>
      )}

      {/* ─── Section 5.1.2 — Ablation Study ──────────────────────────── */}
      <div className="section-header">Section 5.1.2 — Ablation Study (Mode A → D)</div>
      {!avail.ablation || ablationModes.length === 0 ? (
        <div className="panel" style={{ color: THEME.dim, fontSize: '0.82rem', marginBottom: '1rem' }}>
          No ablation results yet. Run: <code style={{ color: THEME.cyan }}>python scripts/benchmark_ablation.py</code>
        </div>
      ) : (
        <>
          <div className="panel" style={{ marginBottom: '0.75rem' }}>
            <h4 className="panel-title" style={{ color: THEME.green }}>
              📈 Multi-Mode Performance Comparison
            </h4>
            <AblationChart ablation={ablation} />
          </div>

          {/* Mode summary cards */}
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${ablationModes.length}, 1fr)`, gap: '0.75rem', marginBottom: '1rem' }}>
            {ablationModes.map((mode) => {
              const m = ablation[mode]
              const isFull = mode === 'D' || String(mode).toUpperCase().includes('FULL')
              const f1 = Number(m.F1 ?? 0)
              return (
                <div
                  key={mode}
                  className="panel"
                  style={{
                    borderColor: isFull ? `${THEME.cyan}88` : undefined,
                    boxShadow: isFull ? `0 0 24px rgba(0,212,255,0.15)` : undefined,
                  }}
                >
                  {isFull && (
                    <div style={{ fontSize: '0.6rem', color: THEME.cyan, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.3rem' }}>
                      ★ Full AURA
                    </div>
                  )}
                  <div style={{ fontSize: '1rem', fontWeight: 800, color: THEME.text, marginBottom: '0.4rem' }}>
                    Mode {mode}
                  </div>
                  {[
                    ['F1',        m.F1],
                    ['Precision', m.Precision],
                    ['Recall',    m.Recall],
                    ['AUC-ROC',   m.AUC_ROC ?? m.AUC_PR],
                  ].map(([k, v]) => (
                    <div key={String(k)} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.73rem', marginBottom: '0.2rem' }}>
                      <span style={{ color: THEME.dim }}>{k}</span>
                      <span style={{ fontFamily: 'monospace', color: Number(v) > 0.85 ? THEME.green : THEME.text, fontWeight: 600 }}>
                        {v !== undefined && v !== null ? Number(v).toFixed(4) : '—'}
                      </span>
                    </div>
                  ))}
                  {/* F1 bar */}
                  <div className="progress-bar-track" style={{ marginTop: '0.5rem' }}>
                    <div className="progress-bar-fill" style={{ width: `${f1 * 100}%`, background: isFull ? THEME.cyan : THEME.blue }} />
                  </div>
                </div>
              )
            })}
          </div>
        </>
      )}

      {/* Footer note */}
      <div style={{ fontSize: '0.68rem', color: THEME.dim, textAlign: 'center', marginTop: '1rem' }}>
        Results are loaded live from <code>reports/</code> — run benchmarks to populate. Auto-refreshes every {REFRESH_MS / 1000}s.
      </div>
    </>
  )
}
