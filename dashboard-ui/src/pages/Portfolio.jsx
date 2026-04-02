import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RTC } from 'recharts';
import { Shield, AlertTriangle, Briefcase, Calendar } from 'lucide-react';

const COLORS = ['var(--accent-primary)', 'var(--success)', 'var(--warning)', 'var(--danger)'];

export default function Portfolio() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/data/exposure')
      .then(r => r.json())
      .then(d => setData(d));
  }, []);

  if (!data) return <div style={{ color: 'var(--text-secondary)' }}>Loading portfolio exposure...</div>;

  const deployPct = (data.deployed_capital / data.total_capital) * 100;
  const pieData = data.sectors.map(s => ({ name: s.name, value: s.deployed }));
  pieData.push({ name: 'Cash', value: data.cash_balance });

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full flex flex-col">
      <div style={{ marginBottom: '32px' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 700, letterSpacing: '-0.02em', marginBottom: '8px' }}>Portfolio & Risk</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Visualizing capital allocation limits and hard cap thresholds (Rules 12, 14, 25).</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 500px) 1fr', gap: '24px' }}>
        
        {/* Left Col: Capital Deployment and Sectors */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          {/* Rule 12 Gauge */}
          <div className="glass-card" style={{ padding: '32px' }}>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Shield size={20} color={deployPct > 55 ? 'var(--warning)' : 'var(--success)'} /> Capital Deployment (Rule 12)
            </h2>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Deployed Limit (60%)</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 600, color: 'var(--text-primary)' }}>₹{(data.deployed_capital/1000).toFixed(1)}k / ₹{(data.max_allowed_deployed/1000).toFixed(1)}k</div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Total Cash</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 600, color: 'var(--success)' }}>₹{(data.cash_balance/1000).toFixed(1)}k</div>
              </div>
            </div>

            <div style={{ width: '100%', height: '12px', background: 'var(--border)', borderRadius: '6px', overflow: 'hidden' }}>
              <div style={{ 
                width: `${deployPct}%`, height: '100%', 
                background: deployPct > 55 ? 'var(--warning)' : 'var(--success)', 
                transition: 'width 0.5s ease' 
              }} />
            </div>
            {deployPct > 55 && (
              <div style={{ marginTop: '12px', fontSize: '0.875rem', color: 'var(--warning)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <AlertTriangle size={14} /> Approaching max 60% deployment threshold.
              </div>
            )}
          </div>

          {/* Corporate Actions Block */}
          <div className="glass-panel" style={{ padding: '24px' }}>
             <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--warning)' }}>
              <Calendar size={18} /> Blocked Corporate Actions (Rule 25)
            </h2>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              The following tickers have earnings or splits within 3 days. No new long entries are permitted.
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
              {data.blocked_corp_actions.map(ticker => (
                <div key={ticker} style={{ padding: '8px 16px', borderRadius: '4px', background: 'var(--warning-bg)', border: '1px solid rgba(245,158,11,0.3)', color: 'var(--warning)', fontSize: '0.875rem', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                  {ticker}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Col: Sector Exposure Map (Rule 14) */}
        <div className="glass-panel" style={{ padding: '32px', display: 'flex', flexDirection: 'column' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Briefcase size={20} color="var(--accent-primary)" /> Sector Concentration Map
          </h2>
          
          <div style={{ flex: 1, position: 'relative' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={80} outerRadius={120} paddingAngle={2} dataKey="value" stroke="none">
                  {pieData.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.name === 'Cash' ? 'var(--border-light)' : COLORS[index % COLORS.length]} />)}
                </Pie>
                <RTC formatter={(value) => `₹${value}`} contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: '8px', color: '#fff' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div style={{ marginTop: '24px' }}>
             <h3 style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Allocation Breakdown</h3>
             <table style={{ width: '100%', fontSize: '0.875rem' }}>
                <tbody>
                  {data.sectors.map((s, i) => (
                    <tr key={s.name} style={{ borderBottom: '1px solid var(--border-light)' }}>
                      <td style={{ padding: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div style={{ width: '12px', height: '12px', borderRadius: '3px', background: COLORS[i % COLORS.length] }} />
                        {s.name}
                      </td>
                      <td style={{ padding: '12px', textAlign: 'right' }}>
                        <span style={{ color: s.positions >= 2 ? 'var(--danger)' : 'var(--text-primary)', fontWeight: s.positions >= 2 ? 600 : 400 }}>{s.positions} Active Pos</span>
                      </td>
                      <td className="mono" style={{ padding: '12px', textAlign: 'right' }}>₹{(s.deployed/1000).toFixed(1)}k</td>
                    </tr>
                  ))}
                  <tr style={{ background: 'rgba(255,255,255,0.02)' }}>
                      <td style={{ padding: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div style={{ width: '12px', height: '12px', borderRadius: '3px', background: 'var(--border-light)' }} />
                        CASH
                      </td>
                      <td style={{ padding: '12px', textAlign: 'right' }} />
                      <td className="mono" style={{ padding: '12px', textAlign: 'right' }}>₹{(data.cash_balance/1000).toFixed(1)}k</td>
                  </tr>
                </tbody>
             </table>
          </div>
        </div>

      </div>
    </motion.div>
  );
}
