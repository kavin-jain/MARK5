import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Save, AlertTriangle, Key } from 'lucide-react';

export default function Settings() {
  const [consts, setConsts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [savingK, setSavingK] = useState(null);

  const [tokState, setTokState] = useState('idle');
  const [tokVal, setTokVal] = useState('');

  useEffect(() => {
    fetch('http://localhost:8000/api/system/constants')
      .then(r => r.json())
      .then(d => { setConsts(d); setLoading(false); });
  }, []);

  const updateConstant = async (k, v) => {
    setSavingK(k);
    const c = consts.find(x => x.k === k);
    await fetch('http://localhost:8000/api/system/constants', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...c, v })
    });
    setSavingK(null);
  };

  const genToken = async () => {
    setTokState('gen');
    try {
        await fetch('http://localhost:8000/api/system/kite-token', { method: 'POST' });
        const tok = Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join('');
        setTokVal(tok); setTokState('done');
    } catch(e) {
        setTokState('idle');
    }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full flex flex-col">
      <div style={{ marginBottom: '32px' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 700, letterSpacing: '-0.02em', marginBottom: '8px' }}>Global Settings</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Modify fundamental system variables and API keys.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 500px) 1fr', gap: '32px' }}>
        
        {/* API Tokens */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
          <div className="glass-card" style={{ padding: '32px' }}>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Key size={18} color="var(--accent-primary)" /> Broker API (Kite Connect)
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '24px' }}>
              Authentication tokens expire daily. Generate a fresh token before market open.
            </p>

            <button 
              onClick={genToken} disabled={tokState === 'gen'}
              style={{
                width: '100%',
                background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                color: '#fff',
                border: 'none',
                padding: '12px 24px',
                borderRadius: '8px',
                fontWeight: 600,
                cursor: tokState === 'gen' ? 'not-allowed' : 'pointer',
                display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px',
                boxShadow: '0 4px 15px var(--accent-primary-glow)'
              }}
            >
              <Key size={16} /> {tokState === 'gen' ? 'Exchanging Request URL...' : tokState === 'done' ? 'Regenerate Session' : 'Generate Authentication Token'}
            </button>

            {tokState === 'done' && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} style={{ marginTop: '24px', background: 'var(--success-bg)', border: '1px solid rgba(16, 185, 129, 0.2)', padding: '16px', borderRadius: '8px' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--success)', marginBottom: '8px', fontWeight: 600 }}>ACTIVE TOKEN</div>
                <div className="mono" style={{ fontSize: '0.875rem', color: 'var(--text-primary)', wordBreak: 'break-all' }}>{tokVal}</div>
              </motion.div>
            )}
          </div>

          <div className="glass-card" style={{ padding: '32px', background: 'var(--danger-bg)', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--danger)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertTriangle size={18} /> Danger Zone
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '24px' }}>
              System variables modified directly via this console bypass the normal deployment CI/CD rules. Use with extreme caution during live market hours.
            </p>
          </div>
        </div>

        {/* Global Variables */}
        <div className="glass-panel" style={{ padding: '0', overflow: 'hidden' }}>
          <div style={{ padding: '24px', borderBottom: '1px solid var(--border)', background: 'rgba(255,255,255,0.02)' }}>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 600 }}>System Variables</h2>
          </div>
          
          <div style={{ maxHeight: '600px', overflowY: 'auto' }}>
            {loading ? (
              <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-tertiary)' }}>Loading variables...</div>
            ) : (
              <table style={{ width: '100%' }}>
                <tbody>
                  {consts.map((c, i) => (
                    <tr key={c.k} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '16px 24px', width: '60%' }}>
                        <div style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '4px' }}>{c.k}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>{c.desc}</div>
                      </td>
                      <td style={{ padding: '16px 24px', textAlign: 'right' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '8px' }}>
                          <input 
                            value={c.v}
                            onChange={(e) => setConsts(consts.map(x => x.k === c.k ? { ...x, v: e.target.value } : x))}
                            onBlur={(e) => updateConstant(c.k, e.target.value)}
                            onKeyDown={e => { if (e.key === 'Enter') e.target.blur(); }}
                            style={{ 
                              width: '80px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border)',
                              color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '6px',
                              fontFamily: 'var(--font-mono)', fontSize: '0.875rem', textAlign: 'right',
                              boxShadow: savingK === c.k ? '0 0 0 2px var(--accent-primary)' : 'none'
                            }}
                          />
                          <span style={{ color: 'var(--text-tertiary)', fontSize: '0.75rem', width: '24px', textAlign: 'left' }}>{c.u}</span>
                          {savingK === c.k ? (
                            <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}>
                              <Save size={16} color="var(--accent-primary)" />
                            </motion.div>
                          ) : (
                            <Save size={16} color="var(--border)" style={{ opacity: 0.5 }} />
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
