import type { AppConfig, BenchmarkResults, DashboardState, FLServerState } from './types'

const BASE = '/api'

async function req<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...opts?.headers },
    ...opts,
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || 'Request failed')
  return data
}

export const api = {
  // Config -- single source of truth for all frontend constants
  getConfig:            ()             => req<AppConfig>('/config'),
  getBenchmarkResults:  ()             => req<BenchmarkResults>('/benchmark/results'),

  // Dashboard state
  getState:             ()             => req<DashboardState>('/state'),
  getClientState:       (c: string)    => req<DashboardState>(`/client-state?client=${c}`),
  getClientsSummary:    ()             => req<{ key: string; label: string; system_status: string; ae_score: number; attack_active: boolean }[]>('/clients/summary'),

  // Attack / normal injection
  injectClientAttack:   (t: string, c: string) => req<{ state: DashboardState }>(`/client-attack/${t}?client=${c}`, { method: 'POST' }),
  injectClientNormal:   (c: string)            => req<{ state: DashboardState }>(`/client-normal?client=${c}`,    { method: 'POST' }),
  injectAttack:         (t: string)            => req<{ state: DashboardState }>(`/attack/${t}`,                  { method: 'POST' }),
  injectNormal:         ()                     => req<{ state: DashboardState }>('/normal',                       { method: 'POST' }),
  injectCustom:         (script: string, target_node: string, attack_type = 'custom') =>
    req<{ state: DashboardState; mse: number }>('/inject_custom', { method: 'POST', body: JSON.stringify({ script, target_node, attack_type }) }),

  // Logs
  clearLogs:            ()             => req<{ state: DashboardState }>('/logs/clear',   { method: 'POST' }),
  clearClientLogs:      (c?: string)   => req<{ state: DashboardState }>(`/client-clear${c ? `?client=${c}` : ''}`, { method: 'POST' }),

  // Federation
  runFederation:        ()             => req<{ state: DashboardState }>('/federation/run', { method: 'POST' }),

  // Blockchain
  registerHash:         ()             => req<{ state: DashboardState }>('/blockchain/register', { method: 'POST' }),
  verifyChain:          ()             => req<{ ok: boolean; message: string; entries?: unknown[] }>('/blockchain/verify'),

  // FL readiness signals
  setFlReady:           (r: boolean)   => req<{ state: DashboardState }>('/fl/ready',        { method: 'POST', body: JSON.stringify({ ready: r }) }),
  setUnderAttack:       ()             => req<{ state: DashboardState }>('/fl/under-attack',  { method: 'POST' }),
  resolveAttack:        ()             => req<{ state: DashboardState }>('/fl/resolved',      { method: 'POST' }),

  // Node registry
  getNodes:             ()             => req<{ id: string; label: string; critical: boolean }[]>('/nodes'),

  // FL Server console
  getFLState:           ()             => req<FLServerState>('/fl-server/state'),
  runFLSimulation:      ()             => req<{ state: FLServerState }>('/fl-server/run', { method: 'POST' }),
}
