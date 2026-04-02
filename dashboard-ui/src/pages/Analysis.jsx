import React, { useState } from 'react';
import { Search, Hash, TrendingUp, AlertCircle, HelpCircle } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Analysis() {
  const [query, setQuery] = useState('');
  const [stock, setStock] = useState(null);
  const [fetching, setFetching] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query) return;
    setFetching(true);
    try {
      const res = await fetch(`http://localhost:8000/api/data/stock/${query}`);
      const data = await res.json();
      setStock(data);
    } catch (e) {
      setStock({ err: true });
    }
    setFetching(false);
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full flex flex-col">
      <div style={{ marginBottom: '32px' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 700, letterSpacing: '-0.02em', marginBottom: '8px' }}>Analysis Engine</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Deep dive fundamental scanning and AI-assisted synopsis.</p>
      </div>

      <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
        <form onSubmit={handleSearch} style={{ display: 'flex', gap: '16px' }}>
          <div style={{ position: 'relative', flex: 1 }}>
            <Search size={20} color="var(--text-secondary)" style={{ position: 'absolute', left: 16, top: 12 }} />
            <input 
              value={query} onChange={e => setQuery(e.target.value)}
              placeholder="Search ticker (e.g. RELIANCE.NS)..."
              style={{
                width: '100%',
                padding: '12px 16px 12px 48px',
                background: 'rgba(0, 0, 0, 0.2)',
                border: '1px solid var(--border)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontSize: '1rem',
                fontFamily: 'var(--font-mono)'
              }}
            />
          </div>
          <button type="submit" style={{
            background: 'var(--accent-primary)',
            color: '#fff',
            border: 'none',
            padding: '0 24px',
            borderRadius: '8px',
            fontWeight: 600,
            cursor: fetching ? 'wait' : 'pointer',
            opacity: fetching ? 0.7 : 1,
            boxShadow: '0 4px 14px var(--accent-primary-glow)'
          }}>
            {fetching ? 'Scanning...' : 'Analyze'}
          </button>
        </form>
      </div>

      {stock && !stock.err && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '24px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div className="glass-card" style={{ padding: '32px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: '24px', marginBottom: '24px' }}>
                <div>
                  <h2 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-primary)', letterSpacing: '-0.02em' }}>{stock.name}</h2>
                  <div style={{ display: 'flex', gap: '12px', marginTop: '8px', color: 'var(--text-secondary)' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><Hash size={14}/> {stock.sector}</span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><TrendingUp size={14}/> {stock.marketCap} Cap</span>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Current Signal</div>
                  <div className="glow-text-success" style={{ fontSize: '1.5rem', fontWeight: 700 }}>STRONG BUY</div>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '32px' }}>
                {[
                  ['P/E Ratio', stock.pe], ['P/BV', stock.pbv],
                  ['ROE', stock.roe], ['Debt/Eq', stock.debtEquity]
                ].map(([k, v]) => (
                  <div key={k} style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '8px', border: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>{k}</div>
                    <div className="mono" style={{ fontSize: '1.25rem', fontWeight: 600 }}>{v}</div>
                  </div>
                ))}
              </div>

              <div>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <HelpCircle size={18} color="var(--accent-primary)" /> AI Synopsis
                </h3>
                <p style={{ lineHeight: 1.6, color: 'var(--text-secondary)', background: 'rgba(0,0,0,0.2)', padding: '20px', borderRadius: '8px', borderLeft: '3px solid var(--accent-primary)' }}>
                  {stock.synopsis}
                </p>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div className="glass-card" style={{ padding: '24px' }}>
              <h3 style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '16px', letterSpacing: '0.05em' }}>Technical Alerts</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {stock.signals?.map((s, i) => (
                  <div key={i} style={{ display: 'flex', gap: '12px', alignItems: 'center', background: 'var(--success-bg)', border: '1px solid rgba(16,185,129,0.2)', padding: '12px', borderRadius: '6px' }}>
                    <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--success)' }} />
                    <span style={{ fontSize: '0.875rem', color: 'var(--success)', fontWeight: 500 }}>{s}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="glass-card" style={{ padding: '24px' }}>
              <h3 style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '16px', letterSpacing: '0.05em' }}>Recent News</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {stock.news?.map((n, i) => (
                  <div key={i} style={{ paddingBottom: '16px', borderBottom: i < stock.news.length - 1 ? '1px solid var(--border)' : 'none' }}>
                    <div style={{ fontSize: '0.875rem', color: 'var(--text-primary)', marginBottom: '4px', lineHeight: 1.4 }}>{n}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>{Math.floor(Math.random() * 24) + 1} hours ago</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {stock?.err && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', background: 'var(--danger-bg)', border: '1px solid rgba(239, 68, 68, 0.2)', padding: '24px', borderRadius: '12px', color: 'var(--danger)' }}>
          <AlertCircle size={24} />
          <span style={{ fontWeight: 500 }}>Failed to locate ticker data or perform fundamental analysis. Verify symbol.</span>
        </div>
      )}
    </motion.div>
  );
}
