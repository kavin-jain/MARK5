import React from 'react';
import { motion } from 'framer-motion';

const SIGNALS = [
  { t: 'RELIANCE.NS', a: 'BUY', c: 0.721, rank: 1, qty: 34, atr: 38.5, reg: 'BULL', stat: 'ACTV' },
  { t: 'HDFCBANK.NS', a: 'BUY', c: 0.683, rank: 2, qty: 51, atr: 22.1, reg: 'BULL', stat: 'ACTV' },
  { t: 'BHARTIARTL.NS', a: 'BUY', c: 0.698, rank: 3, qty: 62, atr: 15.7, reg: 'BULL', stat: 'QUED' },
  { t: 'TCS.NS', a: 'BUY', c: 0.641, rank: 4, qty: 28, atr: 31.2, reg: 'BULL', stat: 'QUED' },
  { t: 'INFY.NS', a: 'HOLD', c: 0.498, rank: 7, qty: 0, atr: 18.9, reg: 'NEUTRAL', stat: 'HOLD' },
  { t: 'SBIN.NS', a: 'HOLD', c: 0.512, rank: 9, qty: 0, atr: 11.4, reg: 'CHOPPY', stat: 'HOLD' },
  { t: 'ICICIBANK.NS', a: 'HOLD', c: 0.531, rank: 8, qty: 0, atr: 19.8, reg: 'NEUTRAL', stat: 'HOLD' },
];

export default function OrderBook() {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full flex flex-col">
      <div style={{ marginBottom: '32px' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 700, letterSpacing: '-0.02em', marginBottom: '8px' }}>Order Ledger</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Live view of queued, active, and historical positions.</p>
      </div>

      <div className="glass-panel" style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div style={{ overflowX: 'auto', flex: 1 }}>
          <table style={{ minWidth: '800px' }}>
            <thead>
              <tr style={{ background: 'rgba(0,0,0,0.2)' }}>
                <th>Ticker</th>
                <th>Operation</th>
                <th>Confidence</th>
                <th>Model Rank</th>
                <th>Capital Alloted</th>
                <th>ATR Target</th>
                <th>Mkt Regime</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {SIGNALS.map((s, i) => {
                const isB = s.a === 'BUY';
                return (
                  <tr key={i}>
                    <td style={{ fontWeight: 600 }}>{s.t}</td>
                    <td>
                      <span style={{ 
                        padding: '4px 12px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600,
                        background: isB ? 'var(--success-bg)' : 'transparent',
                        color: isB ? 'var(--success)' : 'var(--text-secondary)',
                      }}>
                        {s.a}
                      </span>
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div style={{ flex: 1, maxWidth: '100px', height: '6px', background: 'var(--border)', borderRadius: '3px' }}>
                          <div style={{ 
                            height: '100%', borderRadius: '3px',
                            width: `${s.c * 100}%`, 
                            background: s.c > 0.65 ? 'var(--success)' : s.c > 0.5 ? 'var(--warning)' : 'var(--danger)',
                            boxShadow: s.c > 0.65 ? '0 0 10px var(--success)' : 'none'
                          }} />
                        </div>
                        <span className="mono" style={{ fontSize: '0.875rem' }}>{(s.c * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="mono">#{s.rank}</td>
                    <td className="mono">{isB ? s.qty + ' shares' : '—'}</td>
                    <td className="mono">₹{s.atr}</td>
                    <td>
                      <span style={{ fontSize: '0.75rem', color: s.reg === 'BULL' ? 'var(--success)' : s.reg === 'CHOPPY' ? 'var(--danger)' : 'var(--warning)' }}>
                        {s.reg}
                      </span>
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <div style={{ 
                          width: '8px', height: '8px', borderRadius: '50%',
                          background: s.stat === 'ACTV' ? 'var(--success)' : s.stat === 'QUED' ? 'var(--accent-primary)' : 'var(--text-tertiary)',
                          boxShadow: s.stat === 'ACTV' ? '0 0 8px var(--success-glow)' : 'none'
                        }} />
                        <span style={{ fontSize: '0.75rem', fontWeight: 500, color: s.stat === 'ACTV' ? '#fff' : 'var(--text-tertiary)' }}>{s.stat}</span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </motion.div>
  );
}
