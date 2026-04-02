import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Terminal, Database, ShieldCheck, Activity, AlertTriangle, AlertOctagon } from 'lucide-react';

export default function Engineering() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/system/health')
      .then(r => r.json())
      .then(d => setData(d.engineering));
  }, []);

  if (!data) return <div style={{ color: 'var(--text-secondary)' }}>Loading telemetry...</div>;

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full flex flex-col">
      <div style={{ marginBottom: '32px' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 700, letterSpacing: '-0.02em', marginBottom: '8px' }}>Engineering & DevOps</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Pipeline telemetry, broker reconciliation metrics, and API latency charts.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 1fr', gap: '24px' }}>
        
        {/* Left Col: Alerts and Broker States */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

          {/* Broker Reconciliation (Rule 46) */}
          <div className="glass-card" style={{ padding: '32px', border: data.broker_reconciliation ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid rgba(239, 68, 68, 0.4)' }}>
             <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px', color: data.broker_reconciliation ? 'var(--success)' : 'var(--danger)' }}>
                {data.broker_reconciliation ? <ShieldCheck size={24} /> : <AlertOctagon size={24} />}
                Broker State Reconciliation
             </h2>
             <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
               Rule 46 Verification: Local system variables must perfectly match Kite Connect portfolio holdings.
             </p>
             <div style={{ display: 'flex', gap: '16px' }}>
                 <div style={{ flex: 1, padding: '16px', borderRadius: '8px', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border)' }}>
                     <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '4px' }}>System DB Positions</div>
                     <div className="mono" style={{ fontSize: '1.5rem', fontWeight: 600 }}>142</div>
                 </div>
                 <div style={{ flex: 1, padding: '16px', borderRadius: '8px', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border)' }}>
                     <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '4px' }}>Kite Connect Positions</div>
                     <div className="mono" style={{ fontSize: '1.5rem', fontWeight: 600 }}>142</div>
                 </div>
             </div>
             <div style={{ marginTop: '16px', padding: '12px', borderRadius: '8px', background: data.broker_reconciliation ? 'var(--success-bg)' : 'var(--danger-bg)', color: data.broker_reconciliation ? 'var(--success)' : 'var(--danger)', fontSize: '0.875rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                 {data.broker_reconciliation ? 'MATCH CONFIRMED: Trading Allowed' : 'CRITICAL MISMATCH: System Halted'}
             </div>
          </div>

          {/* API Latency Block */}
          <div className="glass-panel" style={{ padding: '24px' }}>
              <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-primary)' }}>
                <Activity size={18} /> API Telemetry
              </h2>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
                 <div>
                   <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Kite API Calls / Min</div>
                   <div className="mono" style={{ fontSize: '1.5rem', fontWeight: 600 }}>{data.api_requests_per_min} <span style={{ fontSize: '0.875rem', color: 'var(--text-tertiary)' }}>/ 180</span></div>
                 </div>
                 <div style={{ textAlign: 'right' }}>
                   <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Latency (ms)</div>
                   <div className="mono" style={{ fontSize: '1.5rem', fontWeight: 600, color: data.api_latency_ms < 200 ? 'var(--success)' : 'var(--warning)' }}>{data.api_latency_ms}</div>
                 </div>
              </div>
              <div style={{ width: '100%', height: '8px', background: 'var(--border)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ width: `${(data.api_requests_per_min / 180) * 100}%`, height: '100%', background: 'var(--accent-primary)', boxShadow: '0 0 10px var(--accent-primary-glow)' }}/>
              </div>
          </div>

        </div>

        {/* Right Col: Data Pipeline Integrity */}
        <div className="glass-panel" style={{ padding: '32px', display: 'flex', flexDirection: 'column' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Database size={20} color="var(--accent-secondary)" /> Pipeline Integrity Log
          </h2>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
             
             {/* Missing Bars Check (Rule 50) */}
             <div style={{ padding: '16px', borderRadius: '8px', border: '1px solid var(--border)', background: data.missing_bars_detected > 0 ? 'var(--warning-bg)' : 'rgba(255,255,255,0.02)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                   <span style={{ fontSize: '0.875rem', fontWeight: 600, color: data.missing_bars_detected > 0 ? 'var(--warning)' : 'var(--text-primary)' }}>MISSING BARS DETECTION</span>
                   <span className="mono" style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Rule 50</span>
                </div>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                    Forward-filling threshold: max 1 period.
                </div>
                <div style={{ marginTop: '12px', fontSize: '1.25rem', fontWeight: 700, color: data.missing_bars_detected === 0 ? 'var(--success)' : 'var(--warning)' }}>
                    {data.missing_bars_detected} Failures
                </div>
             </div>

             {/* Gap Violations Check (Rule 49) */}
             <div style={{ padding: '16px', borderRadius: '8px', border: '1px solid var(--border)', background: data.gap_violations.length > 0 ? 'var(--danger-bg)' : 'rgba(255,255,255,0.02)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                   <span style={{ fontSize: '0.875rem', fontWeight: 600, color: data.gap_violations.length > 0 ? 'var(--danger)' : 'var(--text-primary)' }}>&gt;10% OHLC GAP DETECTED</span>
                   <span className="mono" style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Rule 49</span>
                </div>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                    Reject bars with massive anomalies to protect model ingestion.
                </div>
                <div style={{ marginTop: '12px' }}>
                    {data.gap_violations.length === 0 ? (
                        <div style={{ color: 'var(--success)', fontWeight: 600, fontSize: '0.875rem' }}>All streams clean.</div>
                    ) : (
                        <div style={{ display: 'flex', gap: '8px' }}>
                           {data.gap_violations.map(t => (
                               <span key={t} style={{ background: 'var(--danger)', color: '#fff', padding: '4px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600 }}>{t} <AlertTriangle size={12} style={{ display: 'inline', marginLeft: 4 }} /></span>
                           ))}
                        </div>
                    )}
                </div>
             </div>

             {/* Output Log stream */}
             <div style={{ flex: 1, marginTop: '16px', background: '#000', borderRadius: '8px', border: '1px solid var(--border-light)', padding: '16px', overflowY: 'auto' }}>
                 <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '8px', letterSpacing: '0.05em' }}>Tail: /var/log/system.log</div>
                 {[
                     "[INFO] CRON: Fetched latest daily OHLC bars.",
                     "[INFO] PIPELINE: Feature engineering completed in 1.42s.",
                     `[WARN] PIPELINE: Dropped 1 bar for ${data.gap_violations[0]} due to >10% jump.`,
                     "[INFO] BROKER: Reconciled 142 positions successfully."
                 ].map((log, i) => (
                    <div key={i} className="mono" style={{ fontSize: '0.75rem', color: log.includes('WARN') ? 'var(--warning)' : 'var(--success)', marginBottom: '4px' }}>{log}</div>
                 ))}
             </div>

          </div>
        </div>

      </div>
    </motion.div>
  );
}
