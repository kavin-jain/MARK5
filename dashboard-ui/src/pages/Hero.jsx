import React, { useState, useEffect } from 'react';
import { AreaChart, Area, BarChart, Bar, Cell, XAxis, YAxis, ResponsiveContainer, Tooltip as RTC } from 'recharts';
import { TrendingUp, Activity, DollarSign, PieChart, ShieldAlert, CheckCircle, Database, Server, Key, PowerOff, Zap } from 'lucide-react';
import { motion } from 'framer-motion';

function StatCard({ title, value, icon: Icon, color, trend }) {
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} whileHover={{ y: -2 }} className="glass-card" style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h3 style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '8px' }}>{title}</h3>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.03em' }}>{value}</div>
          {trend && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginTop: '8px' }}>
              <TrendingUp size={14} color="var(--success)" />
              <span className="glow-text-success" style={{ fontSize: '0.8rem', fontWeight: 600 }}>{trend}</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>vs last month</span>
            </div>
          )}
        </div>
        <div style={{ padding: '12px', borderRadius: '12px', background: `rgba(${color}, 0.1)`, color: `rgb(${color})`, boxShadow: `0 0 20px rgba(${color}, 0.15)` }}>
          <Icon size={24} />
        </div>
      </div>
    </motion.div>
  );
}

export default function Hero() {
  const [data, setData] = useState({ equity: [], health: null, fii: null });
  const [killing, setKilling] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch('http://localhost:8000/api/data/pnl').then(r => r.json()),
      fetch('http://localhost:8000/api/system/health').then(r => r.json()),
      fetch('http://localhost:8000/api/data/fii-flow').then(r => r.json())
    ]).then(([pnl, health, fii]) => {
      setData({ equity: pnl.equity, health, fii });
    });
  }, []);

  const triggerKill = async (action) => {
    if (!window.confirm(`CRITICAL OVERRIDE: Are you sure you want to ${action} the system?`)) return;
    setKilling(true);
    await fetch('http://localhost:8000/api/system/kill-switch', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action })
    });
    alert(`System action ${action} executed.`);
    setKilling(false);
  };

  const lastEq = data.equity.length ? data.equity[data.equity.length - 1].eq : 100000;
  const pnlPct = ((lastEq - 100000) / 1000).toFixed(2);

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
      
      {/* Morning Checklist Traffic Light */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '32px' }}>
        {[
          { icon: Server, label: 'Redis', state: data.health?.checklist.redis.status === 'OK' },
          { icon: Database, label: 'DB Sync', state: data.health?.checklist.db_sync.status === 'OK' },
          { icon: Key, label: 'Kite API', state: data.health?.checklist.kite_token.status === 'VALID' },
          { icon: Activity, label: 'Market', state: data.health?.checklist.market.status === 'OPEN' },
        ].map((item, i) => (
          <div key={i} style={{ flex: 1, padding: '12px 16px', borderRadius: '8px', background: 'var(--bg-elevated)', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ width: 10, height: 10, borderRadius: '50%', background: item.state ? 'var(--success)' : 'var(--warning)', boxShadow: `0 0 10px ${item.state ? 'var(--success)' : 'var(--warning)'}` }} />
            <item.icon size={16} color="var(--text-secondary)" />
            <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>{item.label}</span>
          </div>
        ))}
      </div>

      <div style={{ marginBottom: '32px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 style={{ fontSize: '2rem', fontWeight: 700, letterSpacing: '-0.02em', marginBottom: '8px' }}>Command Center</h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            System overview and high-level performance metrics. <strong style={{ color: 'var(--accent-primary)' }}>NIFTY Regime: {data.health?.checklist.market.regime || '—'}</strong>
          </p>
        </div>
        <div style={{ display: 'flex', gap: '16px' }}>
          <button onClick={() => triggerKill('HALT')} disabled={killing} className="glass-card" style={{ padding: '12px 24px', background: 'var(--warning-bg)', color: 'var(--warning)', border: '1px solid rgba(245,158,11,0.3)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <ShieldAlert size={18} /> HALT ENTRIES
          </button>
          <button onClick={() => triggerKill('LIQUIDATE')} disabled={killing} className="glass-card" style={{ padding: '12px 24px', background: 'var(--danger-bg)', color: 'var(--danger)', border: '1px solid rgba(239, 68, 68,0.3)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', boxShadow: '0 0 15px var(--danger-glow)' }}>
            <PowerOff size={18} /> LIQUIDATE ALL
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '24px', marginBottom: '32px' }}>
        <StatCard title="Total Capital" value={`₹${(lastEq / 1000).toFixed(1)}k`} icon={DollarSign} color="99, 102, 241" trend={`+${pnlPct}%`} />
        <StatCard title="Win Rate" value="61.4%" icon={PieChart} color="16, 185, 129" trend="+2.1%" />
        <StatCard title="Live Positions" value="4 / 10" icon={Activity} color="245, 158, 11" />
        <StatCard title="Current Drawdown" value="-2.3%" icon={TrendingUp} color="239, 68, 68" />
      </div>

      {/* Market Regime Block */}
      {data.health && data.health.market_context && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} delay={0.1} className="glass-panel" style={{ padding: '24px', marginBottom: '32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderLeft: '4px solid var(--accent-secondary)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
             <div>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '4px', color: 'var(--accent-secondary)' }}>Regime Gate (Rule 23)</h2>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Dynamically regulates sizing and strategy selection.</div>
             </div>
          </div>
          <div style={{ display: 'flex', gap: '40px', alignItems: 'center' }}>
             <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '4px' }}>India VIX (Risk)</div>
                <div className="mono" style={{ fontSize: '1.25rem', fontWeight: 600, color: data.health.market_context.india_vix > 22 ? 'var(--warning)' : 'var(--text-primary)' }}>{data.health.market_context.india_vix}</div>
             </div>
             <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '4px' }}>NIFTY ADX (Trend)</div>
                <div className="mono" style={{ fontSize: '1.25rem', fontWeight: 600, color: data.health.market_context.nifty_adx > 25 ? 'var(--success)' : 'var(--text-primary)' }}>{data.health.market_context.nifty_adx}</div>
             </div>
             <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '4px' }}>NIFTY 50d EMA</div>
                <div className="mono" style={{ fontSize: '1.25rem', fontWeight: 600, color: data.health.market_context.nifty_above_50ema ? 'var(--success)' : 'var(--danger)' }}>{data.health.market_context.nifty_above_50ema ? 'ABOVE' : 'BELOW'}</div>
             </div>
             <div style={{ height: '40px', width: '1px', background: 'var(--border)' }}></div>
             <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '6px' }}>Current Multiplier</div>
                <span style={{ background: 'rgba(99, 102, 241, 0.1)', color: 'var(--accent-primary)', border: '1px solid rgba(99, 102, 241, 0.3)', padding: '6px 12px', borderRadius: '6px', fontSize: '1rem', fontWeight: 700, letterSpacing: '0.05em' }}>{data.health.market_context.regime_status}</span>
             </div>
          </div>
        </motion.div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
        <motion.div className="glass-panel" style={{ padding: '24px', height: '400px' }} initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }}>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent-primary)', boxShadow: '0 0 10px var(--accent-primary)' }} />
            Equity Curve (Live)
          </h2>
          <ResponsiveContainer width="100%" height="85%">
            <AreaChart data={data.equity} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colorEq" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--accent-primary)" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="var(--accent-primary)" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <XAxis dataKey="d" stroke="var(--border-light)" tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }} tickLine={false} axisLine={false} minTickGap={40} />
              <YAxis stroke="var(--border-light)" tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }} tickLine={false} axisLine={false} tickFormatter={v => `₹${v/1000}k`} domain={['auto', 'auto']} />
              <RTC contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.5)', color: '#fff' }} />
              <Area type="monotone" dataKey="eq" stroke="var(--accent-primary)" strokeWidth={3} fillOpacity={1} fill="url(#colorEq)" />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>

        {/* FII Flow Mini Chart */}
        <motion.div className="glass-panel" style={{ padding: '24px', height: '400px', display: 'flex', flexDirection: 'column' }} initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }}>
           <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Zap size={18} color="var(--success)" /> FII Net Flow (3-Day)
          </h2>
          
          <div style={{ flex: 1 }}>
            {data.fii && (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.fii.days}>
                  <XAxis dataKey="date" stroke="var(--text-tertiary)" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
                  <RTC cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: '8px', color: '#fff' }} formatter={v => `${v} Cr`} />
                  <Bar dataKey="net_cr" radius={[4, 4, 4, 4]}>
                    {data.fii.days.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.net_cr > 0 ? "var(--success)" : "var(--danger)"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
          
          {data.fii && data.fii.regime_gate_active && (
            <div style={{ marginTop: '16px', padding: '16px', background: 'var(--warning-bg)', border: '1px solid rgba(245,158,11,0.2)', borderRadius: '8px', color: 'var(--warning)', fontSize: '0.875rem' }}>
              <strong>Gate Triggered:</strong> {data.fii.action}
            </div>
          )}
        </motion.div>
      </div>
    </motion.div>
  );
}
