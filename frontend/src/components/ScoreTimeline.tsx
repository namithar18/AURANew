import Plot from 'react-plotly.js'
import { THEME } from '../theme'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Trace = any

interface Props {
  scores: number[]
  thresholds: number[]
  mseThresholdMedium?: number  // driven by /api/config
  mseThresholdHigh?: number    // driven by /api/config
}

export default function ScoreTimeline({ scores, thresholds, mseThresholdMedium, mseThresholdHigh }: Props) {
  if (!scores.length) {
    return (
      <Plot
        data={[]}
        layout={{
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          height: 210,
          font: { color: THEME.dim },
          annotations: [{ text: 'Awaiting data…', showarrow: false, font: { color: THEME.dim, size: 13 } }],
          xaxis: { showgrid: false, zeroline: false, showticklabels: false },
          yaxis: { showgrid: false, zeroline: false, showticklabels: false },
          margin: { l: 40, r: 10, t: 10, b: 30 },
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
    )
  }

  const t = scores.map((_, i) => i)

  // Identify attack zones (score exceeds EMA threshold)
  const shapes: object[] = []
  let inZone = false
  let zoneStart = 0
  for (let i = 0; i < scores.length; i++) {
    const over = thresholds[i] > 0 && scores[i] > thresholds[i]
    if (over && !inZone) { inZone = true; zoneStart = i }
    if (!over && inZone) {
      shapes.push({
        type: 'rect', x0: zoneStart, x1: i, y0: 0, y1: 1,
        xref: 'x', yref: 'paper',
        fillcolor: 'rgba(255,58,58,0.08)',
        line: { width: 0 },
        layer: 'below',
      })
      inZone = false
    }
  }
  if (inZone) {
    shapes.push({
      type: 'rect', x0: zoneStart, x1: scores.length - 1, y0: 0, y1: 1,
      xref: 'x', yref: 'paper',
      fillcolor: 'rgba(255,58,58,0.08)',
      line: { width: 0 },
      layer: 'below',
    })
  }

  const traces: Trace[] = [
    {
      x: t, y: scores,
      mode: 'lines',
      name: 'AE Score (MSE)',
      fill: 'tozeroy',
      fillcolor: 'rgba(0,212,255,0.05)',
      line: { color: THEME.cyan, width: 2 },
    },
  ]

  if (thresholds.some(v => v > 0)) {
    traces.push({
      x: t, y: thresholds,
      mode: 'lines',
      name: 'EMA Threshold (3σ)',
      line: { color: THEME.red, width: 1.5, dash: 'dash' },
    })
  }

  // Static threshold lines from config (if provided)
  const annotations: object[] = []
  if (mseThresholdMedium) {
    shapes.push({
      type: 'line', x0: 0, x1: scores.length - 1, y0: mseThresholdMedium, y1: mseThresholdMedium,
      xref: 'x', yref: 'y',
      line: { color: 'rgba(255,140,0,0.5)', width: 1, dash: 'dot' },
      layer: 'above',
    })
    annotations.push({ x: 0, y: mseThresholdMedium, xref: 'x', yref: 'y', text: 'P90', showarrow: false, font: { size: 9, color: THEME.orange }, xanchor: 'left', yanchor: 'bottom' })
  }
  if (mseThresholdHigh) {
    shapes.push({
      type: 'line', x0: 0, x1: scores.length - 1, y0: mseThresholdHigh, y1: mseThresholdHigh,
      xref: 'x', yref: 'y',
      line: { color: 'rgba(255,58,58,0.4)', width: 1, dash: 'dot' },
      layer: 'above',
    })
    annotations.push({ x: 0, y: mseThresholdHigh, xref: 'x', yref: 'y', text: 'P99', showarrow: false, font: { size: 9, color: THEME.red }, xanchor: 'left', yanchor: 'bottom' })
  }

  return (
    <Plot
      data={traces}
      layout={{
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: THEME.text, size: 11 },
        legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 9 } },
        xaxis: { showgrid: false, title: { text: 'Window', standoff: 4 }, color: THEME.dim, tickfont: { size: 9 } },
        yaxis: { showgrid: true, gridcolor: 'rgba(24,37,64,0.6)', title: { text: 'MSE', standoff: 4 }, color: THEME.dim, tickfont: { size: 9 } },
        height: 210,
        margin: { l: 44, r: 10, t: 8, b: 36 },
        autosize: true,
        shapes,
        annotations,
        hoverlabel: { bgcolor: '#0b1325', bordercolor: '#1e3058', font: { size: 11 } },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}
