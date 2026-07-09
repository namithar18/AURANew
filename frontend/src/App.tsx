import { Routes, Route, NavLink } from 'react-router-dom'
import DashboardPage from './pages/DashboardPage'
import FLServerPage from './pages/FLServerPage'
import BenchmarkPage from './pages/BenchmarkPage'
import { useConfig } from './useConfig'

export default function App() {
  const cfg = useConfig()
  const numClients = Object.keys(cfg.org_profiles).length || 5

  return (
    <div className="app-layout">
      <aside className="sidebar">
        {/* Brand */}
        <div className="sidebar-brand">🛡 AURA</div>
        <div className="sidebar-subtitle">Autonomous Unified Resilience Architecture</div>

        {/* Navigation */}
        <nav className="sidebar-nav">
          <NavLink to="/" end className={({ isActive }) => (isActive ? 'active' : '')}>
            🖥️&nbsp; Operations Dashboard
          </NavLink>
          <NavLink to="/fl-server" className={({ isActive }) => (isActive ? 'active' : '')}>
            ⚙️&nbsp; FL Server Console
          </NavLink>
          <NavLink to="/benchmark" className={({ isActive }) => (isActive ? 'active' : '')}>
            📊&nbsp; Research Benchmarks
          </NavLink>
        </nav>

        <hr className="divider" style={{ margin: '0.75rem 0' }} />

        {/* Architecture legend */}
        <div style={{ fontSize: '0.7rem', lineHeight: 1.9 }}>
          <div style={{ color: '#3a8bff', fontWeight: 700 }}>Layer 1</div>
          <div style={{ color: '#3d5070', marginBottom: '0.3rem' }}>Flow Autoencoder (MSE)</div>
          <div style={{ color: '#3a8bff', fontWeight: 700 }}>Layer 2</div>
          <div style={{ color: '#3d5070', marginBottom: '0.3rem' }}>GraphSAGE ST-GNN</div>
          <div style={{ color: '#a855f7', fontWeight: 700 }}>Federation</div>
          <div style={{ color: '#3d5070', marginBottom: '0.3rem' }}>FLTrust + Flower</div>
          <div style={{ color: '#a855f7', fontWeight: 700 }}>Audit</div>
          <div style={{ color: '#3d5070' }}>SHA-256 Blockchain Ledger</div>
        </div>

        {/* Live status */}
        <div className="sidebar-status" style={{ marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid #182540' }}>
          <div className="live-badge">
            <div className="live-dot" />
            LIVE
          </div>
          <div style={{ fontSize: '0.65rem', color: '#3d5070', marginTop: '0.4rem', lineHeight: 1.6 }}>
            {numClients} Federated Clients Active
          </div>
          <div style={{ fontSize: '0.6rem', color: '#253050', marginTop: '0.3rem', fontFamily: 'JetBrains Mono, monospace' }}>
            Refresh: {cfg.refresh_ms}ms · Nodes: {cfg.num_nodes}
          </div>
        </div>
      </aside>

      <main className="main-content">
        <Routes>
          <Route path="/"          element={<DashboardPage />} />
          <Route path="/fl-server" element={<FLServerPage />} />
          <Route path="/benchmark" element={<BenchmarkPage />} />
        </Routes>
      </main>
    </div>
  )
}
