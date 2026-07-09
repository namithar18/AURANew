/**
 * useConfig — singleton hook that fetches /api/config once on app mount.
 * All frontend components MUST use this hook for org profiles, attack types,
 * refresh rate, thresholds, etc. Nothing is hardcoded in TSX/TSX files.
 */
import { useEffect, useState } from 'react'
import { api } from './api'
import type { AppConfig } from './types'

// Fallback config so UI never breaks if API is temporarily unreachable
const FALLBACK_CONFIG: AppConfig = {
  theme: {},
  org_profiles: {},
  attack_types: [],
  all_attack_types: [],
  critical_allowlist: {},
  num_nodes: 20,
  refresh_ms: 1500,
  mse_threshold_medium: 0.02,
  mse_threshold_high:   0.12,
  hitl_low_to_medium:   5,
  hitl_low_to_high:     10,
  hitl_medium_to_high:  10,
  fl_num_rounds:        3,
  ema_alpha:            0.1,
  ema_sigma_multiplier: 3.0,
}

let _cached: AppConfig | null = null
const _listeners: Array<(c: AppConfig) => void> = []

async function _load() {
  try {
    const cfg = await api.getConfig()
    _cached = cfg
    _listeners.forEach(fn => fn(cfg))
  } catch {
    // API not yet ready — use fallback
    _cached = FALLBACK_CONFIG
    _listeners.forEach(fn => fn(FALLBACK_CONFIG))
  }
}

// Kick off the load immediately when this module is imported
_load()

export function useConfig(): AppConfig {
  const [cfg, setCfg] = useState<AppConfig>(_cached ?? FALLBACK_CONFIG)
  useEffect(() => {
    if (_cached) { setCfg(_cached); return }
    const handler = (c: AppConfig) => setCfg(c)
    _listeners.push(handler)
    return () => { const i = _listeners.indexOf(handler); if (i >= 0) _listeners.splice(i, 1) }
  }, [])
  return cfg
}

/** Returns org profiles as an ordered array matching backend insertion order */
export function useClients(cfg: AppConfig) {
  return Object.entries(cfg.org_profiles).map(([key, p]) => ({ key, ...p }))
}
