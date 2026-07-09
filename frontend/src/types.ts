export interface OrgProfile {
  label: string
  id: string
  net: string
  icon: string
  role: string
  color: string
}

export interface AttackType {
  type: string
  label: string
  icon: string
}

export interface NodeInfo {
  id: string
  label: string
  index: number
  critical: boolean
  color: string
  state: string
}

export interface DashboardState {
  system_status: string
  model_status: string
  blockchain_mode: string
  org: OrgProfile | null
  metrics: {
    window_counter: number
    total_attacks: number
    total_blocked: number
    fl_rounds_done: number
    chain_entries: number
    current_ae_score: number
  }
  nodes: NodeInfo[]
  edge_index: number[][] | null
  timeline: {
    scores: number[]
    thresholds: number[]
    timestamps: number[]
  }
  last_explanation: Record<string, unknown> | null
  alerts: Record<string, unknown>[]
  incidents: Record<string, unknown>[]
  fed_log: string[]
  chain_log: { version: string; hash: string; round: string | number; time: string }[]
  fl_client_status: Record<string, unknown>[]
  fl_ready: boolean
  under_attack: boolean
  fl_running: boolean
  ema: {
    warmup_left: number
    mean: number
    std: number
    threshold: number
  }
  critical_allowlist: Record<string, string>
}

export interface FLServerState {
  run_state: string
  fl_running: boolean
  fl_done: boolean
  current_round: number
  total_rounds: number
  pipe_steps: { icon: string; label: string; state: number }[]
  orgs: {
    id: string
    key: string
    label: string
    net: string
    icon: string
    ready: boolean
    under_attack: boolean
    net_live: string
  }[]
  client_cards: Record<string, { status: string; round: number; selected: boolean | null; verified: boolean | null }>
  round_results: Record<string, unknown>[]
  hash_ledger: { version: string; hash: string; round: number; time: string }[]
  fl_log: string[]
  fltrust_scores_hist: { round: number; scores: Record<string, number> }[]
  global_hash: string | null
  global_version: string | null
  byzantine_org: string | null
  quarantined_orgs: string[]
}

/** Returned by /api/config — single source of truth for all frontend config */
export interface AppConfig {
  theme: Record<string, string>
  org_profiles: Record<string, OrgProfile>
  attack_types: AttackType[]
  all_attack_types: { type: string; label: string }[]
  critical_allowlist: Record<string, string>
  num_nodes: number
  refresh_ms: number
  mse_threshold_medium: number
  mse_threshold_high: number
  hitl_low_to_medium: number
  hitl_low_to_high: number
  hitl_medium_to_high: number
  fl_num_rounds: number
  ema_alpha: number
  ema_sigma_multiplier: number
}

export interface HITLTierMetrics {
  tier: string
  count: number
  fer: number
  latency_p50_ms: number
  latency_p95_ms: number
  latency_p99_ms: number
}

export interface HITLBenchmarkResults {
  windows_evaluated?: number
  attack_windows?: number
  benign_windows?: number
  simulated_duration_hr?: number
  total_escalations?: number
  overall_fer?: number
  latency_p50_ms?: number
  latency_p95_ms?: number
  latency_p99_ms?: number
  hitl_calls_per_hour?: number
  hitl_degraded_rate?: number
  wall_time_s?: number
  tier_breakdown?: HITLTierMetrics[]
  criteria?: { fer_pass: boolean; latency_pass: boolean; hitl_rate_pass: boolean; overall_pass: boolean }
}

export interface AblationModeMetrics {
  Mode?: string
  Precision?: number
  Recall?: number
  F1?: number
  AUC_ROC?: number
  AUC_PR?: number
  AUC_Approximate?: boolean
  Time_s?: number
  [key: string]: unknown
}

export interface BenchmarkResults {
  hitl: HITLBenchmarkResults
  ablation: Record<string, AblationModeMetrics>
  available: { hitl: boolean; ablation: boolean }
}


