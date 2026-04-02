import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Cpu, Play, CheckCircle, XCircle, TrendingDown, AlignLeft } from 'lucide-react';

const MODELS = [
  { m: 'XGBoost', b: 0.183, a: 0.681, f: 0.512, w: '38%' },
  { m: 'LightGBM', b: 0.179, a: 0.693, f: 0.528, w: '39%' },
  { m: 'CatBoost', b: 0.191, a: 0.667, f: 0.501, w: '23%' },
];

export default function Models() {
  const [btRun, setBtRun] = useState(false);
  const [btPct, setBtPct] = useState(0);
  const [btLog, setBtLog] = useState([]);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/backtest/health')
      .then(r => r.json())
      .then(d => setHealth(d));
  }, []);

  const runBT = async () => {
    if (btRun) return;
    setBtRun(true); setBtPct(0); setBtLog([]);
    try {
        await fetch(`http://localhost:8000/api/backtest/run`, { method: 'POST' });
        const poll = setInterval(async () => {
            const res = await fetch(`http://localhost:8000/api/backtest/status`);
            const data = await res.json();
            setBtPct(data.progress);
            setBtLog(data.logs);
            if (!data.running && data.progress === 100) {
                clearInterval(poll);
                setBtRun(false);
            }
        }, 500);
    } catch(e) {
        setBtRun(false);
    }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full flex flex-col">
      <div style={{ marginBottom: '32px' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 700, letterSpacing: '-0.02em', marginBottom: '8px' }}>Model Lab (Quant View)</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Ensemble performance, active gating (Rule 39), and alpha decay tracking (Rule 30).</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 500px) 1fr', gap: '24px', marginBottom: '24px' }}>
        
        {/* Edge Tracker & Sharpe */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            
            {/* Edge Tracker */}
            {health && (
              <div className="glass-card" style={{ padding: '24px', background: health.live_edge.win_rate < 40 ? 'var(--danger-bg)' : 'var(--bg-elevated)', border: health.live_edge.win_rate < 40 ? '1px solid rgba(239, 68, 68, 0.3)' : '1px solid var(--border)' }}>
                 <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', color: health.live_edge.win_rate < 40 ? 'var(--danger)' : 'var(--text-primary)' }}>
                   <TrendingDown size={18} /> Live Edge Monitor (Rule 30)
                 </h2>
                 <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                   Tracking system logic vs actual outcomes over last {health.live_edge.rolling_trades} trades.
                 </p>
                 <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                     <div>
                       <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>Win Rate (&gt;40%)</div>
                       <div style={{ fontSize: '1.5rem', fontWeight: 700, color: health.live_edge.win_rate < 40 ? 'var(--danger)' : 'var(--success)' }}>{health.live_edge.win_rate}%</div>
                     </div>
                     <div>
                       <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>Profit Factor (&gt;1.2)</div>
                       <div style={{ fontSize: '1.5rem', fontWeight: 700, color: health.live_edge.profit_factor < 1.2 ? 'var(--warning)' : 'var(--success)' }}>{health.live_edge.profit_factor}</div>
                     </div>
                     <div>
                       <span style={{ padding: '4px 12px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600, background: health.live_edge.status === 'DEGRADED' ? 'var(--danger)' : 'var(--success)', color: '#fff' }}>
                         {health.live_edge.status}
                       </span>
                     </div>
                 </div>
                 {health.live_edge.win_rate < 40 && (
                     <div style={{ marginTop: '16px', fontSize: '0.875rem', color: 'var(--danger)', fontWeight: 500 }}>
                         ⚠️ Automated override active: New position sizing halved until edge recovers.
                     </div>
                 )}
              </div>
            )}

            {/* Sharpe Gating Table */}
            {health && (
                <div className="glass-panel" style={{ padding: '24px' }}>
                    <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <AlignLeft size={18} color="var(--warning)" /> Universal Gating (Rule 39)
                    </h2>
                    <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                      Stocks with 20-session rolling Sharpe &lt; 0.4 are excluded from new entries.
                    </p>
                    <table style={{ width: '100%', fontSize: '0.875rem' }}>
                        <thead>
                            <tr>
                                <th style={{ padding: '8px', borderBottom: '1px solid var(--border)', textAlign: 'left', color: 'var(--text-secondary)' }}>Ticker</th>
                                <th style={{ padding: '8px', borderBottom: '1px solid var(--border)', textAlign: 'right', color: 'var(--text-secondary)' }}>20d Sharpe</th>
                                <th style={{ padding: '8px', borderBottom: '1px solid var(--border)', textAlign: 'center', color: 'var(--text-secondary)' }}>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {health.sharpe_gating.map(s => (
                                <tr key={s.ticker}>
                                    <td className="mono" style={{ padding: '8px', opacity: s.gated ? 0.5 : 1 }}>{s.ticker}</td>
                                    <td className="mono" style={{ padding: '8px', textAlign: 'right', color: s.sharpe < 0.4 ? 'var(--danger)' : 'var(--success)', opacity: s.gated ? 0.5 : 1 }}>{s.sharpe.toFixed(2)}</td>
                                    <td style={{ padding: '8px', textAlign: 'center' }}>
                                        {s.gated ? 
                                            <span style={{ background: 'var(--border)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.7rem' }}>GATED</span> : 
                                            <span style={{ background: 'var(--success-bg)', color: 'var(--success)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.7rem' }}>ACTIVE</span>
                                        }
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>

        {/* Existing Ensemble & Backtest Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
            <div className="glass-panel" style={{ padding: '24px', height: '100%' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Cpu color="var(--accent-primary)" size={20} /> Production Ensemble
                </h2>
                
                {MODELS.map(m => (
                    <div key={m.m} style={{ marginBottom: '24px', padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid var(--border)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
                        <span style={{ fontWeight: 600, fontSize: '1rem' }}>{m.m}</span>
                        <span style={{ fontSize: '0.875rem', color: 'var(--accent-primary)', fontWeight: 600, background: 'var(--accent-primary-glow)', padding: '2px 8px', borderRadius: '4px' }}>
                        Weight: {m.w}
                        </span>
                    </div>
                    {[
                        { l: 'Brier Score (Lower is better)', v: m.b, max: 0.25, type: 'lower' },
                        { l: 'AUC-ROC (Higher is better)', v: m.a, max: 1, type: 'higher' },
                    ].map(stat => {
                        let color = 'var(--text-secondary)';
                        if (stat.type === 'lower') color = stat.v < 0.18 ? 'var(--success)' : stat.v < 0.22 ? 'var(--warning)' : 'var(--danger)';
                        if (stat.type === 'higher') color = stat.v > 0.68 ? 'var(--success)' : stat.v > 0.60 ? 'var(--warning)' : 'var(--danger)';
                        return (
                        <div key={stat.l} style={{ marginBottom: '12px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '6px' }}>
                            <span style={{ color: 'var(--text-secondary)' }}>{stat.l}</span>
                            <span className="mono" style={{ color, fontWeight: 600 }}>{stat.v.toFixed(3)}</span>
                            </div>
                            <div style={{ height: '6px', background: 'var(--border)', borderRadius: '3px', overflow: 'hidden' }}>
                            <div style={{ height: '100%', width: `${(stat.v / stat.max) * 100}%`, background: color, borderRadius: '3px' }} />
                            </div>
                        </div>
                        );
                    })}
                    </div>
                ))}
            </div>

            <div className="glass-panel" style={{ padding: '24px', height: '100%', display: 'flex', flexDirection: 'column' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Parallel Backtest</h2>
                <button 
                    onClick={runBT} disabled={btRun}
                    style={{ 
                    display: 'flex', alignItems: 'center', gap: '8px',
                    background: btRun ? 'transparent' : 'var(--success)', color: btRun ? 'var(--text-secondary)' : '#fff', 
                    border: btRun ? '1px solid var(--border)' : 'none', padding: '8px 16px', borderRadius: '8px', 
                    fontWeight: 600, cursor: btRun ? 'not-allowed' : 'pointer' 
                    }}
                >
                    <Play size={16} fill="currentColor" /> {btRun ? 'Running...' : 'Execute'}
                </button>
                </div>

                {btRun && (
                <div style={{ marginBottom: '24px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '8px' }}>
                    <span style={{ color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Progress</span>
                    <span className="mono">{btPct}%</span>
                    </div>
                    <div style={{ width: '100%', height: '8px', background: 'var(--border)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ width: `${btPct}%`, height: '100%', background: 'var(--success)', transition: 'width 0.3s ease', boxShadow: '0 0 10px var(--success)' }} />
                    </div>
                </div>
                )}

                <div style={{ flex: 1, background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border)', borderRadius: '8px', padding: '16px', overflowY: 'auto' }}>
                {btLog.length === 0 ? (
                    <div style={{ color: 'var(--text-tertiary)', fontSize: '0.875rem', fontStyle: 'italic' }}>Awaiting execution sequence...</div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {btLog.map((log, i) => (
                        <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} key={i} style={{ display: 'flex', gap: '12px', alignItems: 'center', fontSize: '0.875rem', fontFamily: 'var(--font-mono)' }}>
                        {log.pass ? <CheckCircle size={16} color="var(--success)" /> : <XCircle size={16} color="var(--danger)" />}
                        <span style={{ width: '120px', color: 'var(--text-primary)' }}>{log.ticker}</span>
                        <span style={{ color: 'var(--text-secondary)' }}>Sharpe:</span>
                        <span style={{ color: log.pass ? 'var(--success)' : 'var(--danger)', fontWeight: 600 }}>{log.sh.toFixed(2)}</span>
                        </motion.div>
                    ))}
                    </div>
                )}
                </div>
            </div>
        </div>

      </div>
    </motion.div>
  );
}
