import { useMemo } from 'react'
import Plot from 'react-plotly.js'
import type { NodeInfo } from '../types'
import { THEME } from '../theme'

interface Props {
  nodes: NodeInfo[]
  edgeIndex: number[][] | null
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Trace = any

export default function NetworkGraph({ nodes, edgeIndex }: Props) {
  const n = nodes.length || 1
  const angles = useMemo(
    () => Array.from({ length: n }, (_, i) => (2 * Math.PI * i) / n),
    [n]
  )
  const posX = angles.map(a => Math.cos(a))
  const posY = angles.map(a => Math.sin(a))

  const edgeTraces: Trace[] = useMemo(() => {
    if (!edgeIndex || edgeIndex.length !== 2) return []
    const [src, dst] = edgeIndex
    return Array.from({ length: Math.min(src.length, 80) }, (_, idx) => {
      const s = src[idx], d = dst[idx]
      const isHot = nodes[s]?.state === 'threat' || nodes[d]?.state === 'threat'
      return {
        x: [posX[s], posX[d], null],
        y: [posY[s], posY[d], null],
        mode: 'lines',
        line: { width: isHot ? 2 : 1, color: isHot ? 'rgba(255,58,58,0.5)' : 'rgba(24,37,64,0.7)' },
        hoverinfo: 'none',
        showlegend: false,
      }
    })
  }, [edgeIndex, posX, posY, nodes])

  const nodeColors = nodes.map(nd => {
    if (nd.state === 'threat')     return THEME.red
    if (nd.state === 'evaluating') return THEME.yellow
    return nd.color || THEME.green
  })

  const nodeSizes = nodes.map(nd => {
    if (nd.critical) return nd.state === 'threat' ? 26 : 22
    return nd.state === 'threat' ? 18 : 14
  })

  const nodeTrace: Trace = {
    x: posX,
    y: posY,
    mode: 'markers+text',
    marker: {
      size: nodeSizes,
      color: nodeColors,
      symbol: nodes.map(nd => (nd.critical ? 'diamond' : 'circle')),
      line: {
        width: nodes.map(nd => (nd.state === 'threat' ? 2.5 : 1.5)),
        color: nodes.map(nd => (nd.state === 'threat' ? THEME.red : 'rgba(255,255,255,0.35)')),
      },
      opacity: 1,
    },
    text: nodes.map(nd => `N${nd.index}`),
    textposition: 'top center',
    textfont: { size: 8, color: THEME.text },
    hovertext: nodes.map(nd => `<b>${nd.label}</b><br>State: ${nd.state}${nd.critical ? '<br>⚠ Critical Asset' : ''}`),
    hoverinfo: 'text',
    showlegend: false,
  }

  return (
    <Plot
      data={[...edgeTraces, nodeTrace]}
      layout={{
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 12, r: 12, t: 12, b: 12 },
        xaxis: { showgrid: false, zeroline: false, showticklabels: false, range: [-1.25, 1.25] },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false, scaleanchor: 'x', range: [-1.25, 1.25] },
        height: 300,
        autosize: true,
        hoverlabel: { bgcolor: '#0b1325', bordercolor: '#1e3058', font: { size: 11, color: '#d4e4f4' } },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}
