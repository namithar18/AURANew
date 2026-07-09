import Plot from 'react-plotly.js'
import { THEME } from '../theme'

interface Props {
  groupResiduals: Record<string, number> | null
  underAttack: boolean
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Trace = any

export default function ThreatRadar({ groupResiduals, underAttack }: Props) {
  const empty = !groupResiduals || Object.keys(groupResiduals).length === 0

  const categories = empty
    ? ['Throughput', 'Packet Rate', 'TTL', 'TCP Flags', 'IAT', 'DNS', 'Retransmit']
    : Object.keys(groupResiduals)

  const values = empty
    ? [0, 0, 0, 0, 0, 0, 0]
    : categories.map(k => groupResiduals[k] ?? 0)

  const maxVal = Math.max(...values, 0.001)
  const normValues = values.map(v => v / maxVal)

  const fillColor = underAttack
    ? 'rgba(255,58,58,0.15)'
    : 'rgba(0,212,255,0.08)'
  const lineColor = underAttack ? THEME.red : THEME.cyan

  const trace: Trace = {
    type: 'scatterpolar',
    r: [...normValues, normValues[0]],
    theta: [...categories, categories[0]],
    fill: 'toself',
    fillcolor: fillColor,
    line: { color: lineColor, width: 2 },
    mode: 'lines+markers',
    marker: { size: 5, color: lineColor },
    hovertemplate: '<b>%{theta}</b><br>Residual: %{r:.4f}<extra></extra>',
    name: 'Feature Group Residuals',
  }

  return (
    <Plot
      data={[trace]}
      layout={{
        polar: {
          bgcolor: 'rgba(0,0,0,0)',
          radialaxis: {
            visible: true,
            range: [0, 1],
            showticklabels: false,
            gridcolor: 'rgba(24,37,64,0.6)',
            linecolor: 'rgba(24,37,64,0.4)',
          },
          angularaxis: {
            gridcolor: 'rgba(24,37,64,0.5)',
            linecolor: 'rgba(24,37,64,0.5)',
            tickfont: { size: 9, color: THEME.dim },
          },
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor:  'rgba(0,0,0,0)',
        font: { color: THEME.text, size: 10 },
        showlegend: false,
        margin: { l: 40, r: 40, t: 20, b: 20 },
        height: 240,
        autosize: true,
        hoverlabel: { bgcolor: '#0b1325', bordercolor: '#1e3058', font: { size: 11 } },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}
