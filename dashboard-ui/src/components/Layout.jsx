import React from 'react';
import { NavLink } from 'react-router-dom';
import { LineChart, PieChart, Search, BookOpen, Cpu, Terminal, Settings, Activity } from 'lucide-react';
import './../index.css';

const navItems = [
  { path: '/', name: 'Command Center', icon: LineChart },
  { path: '/portfolio', name: 'Portfolio & Risk', icon: PieChart },
  { path: '/analysis', name: 'Analysis Engine', icon: Search },
  { path: '/orderbook', name: 'Order Book', icon: BookOpen },
  { path: '/models', name: 'Model Lab', icon: Cpu },
  { path: '/engineering', name: 'DevOps & Health', icon: Terminal },
  { path: '/settings', name: 'System Settings', icon: Settings },
];

export default function Layout({ children }) {
  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: 'var(--bg)' }}>
      {/* Sidebar Navigation */}
      <aside style={{
        width: '260px',
        borderRight: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        padding: '24px 0',
        background: 'rgba(11, 14, 20, 0.4)',
        backdropFilter: 'blur(20px)',
        position: 'fixed',
        top: 0, bottom: 0, left: 0,
        zIndex: 10
      }}>
        {/* Logo Section */}
        <div style={{ padding: '0 24px', marginBottom: '40px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: '32px', height: '32px', borderRadius: '8px',
              background: 'linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 4px 12px var(--accent-primary-glow)'
            }}>
              <Activity size={18} color="#fff" />
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: '1.1rem', letterSpacing: '-0.02em', color: '#fff' }}>MARK5</div>
              <div style={{ fontSize: '0.65rem', color: 'var(--accent-primary)', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 600 }}>Pro Suite v5.0</div>
            </div>
          </div>
        </div>

        {/* Navigation Links */}
        <nav style={{ display: 'flex', flexDirection: 'column', gap: '4px', padding: '0 12px' }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                style={({ isActive }) => ({
                  display: 'flex', alignItems: 'center', gap: '12px',
                  padding: '12px 16px', borderRadius: '8px',
                  textDecoration: 'none', transition: 'all 0.2s ease',
                  color: isActive ? '#fff' : 'var(--text-secondary)',
                  background: isActive ? 'var(--bg-elevated)' : 'transparent',
                  border: isActive ? '1px solid rgba(255, 255, 255, 0.08)' : '1px solid transparent',
                  boxShadow: isActive ? '0 4px 12px rgba(0, 0, 0, 0.2)' : 'none',
                  fontWeight: isActive ? 500 : 400
                })}
              >
                {({ isActive }) => (
                  <>
                    <Icon size={18} color={isActive ? 'var(--accent-primary)' : 'currentColor'} />
                    <span style={{ fontSize: '0.9rem' }}>{item.name}</span>
                    {isActive && (
                      <div style={{
                        marginLeft: 'auto', width: '4px', height: '4px', borderRadius: '50%',
                        background: 'var(--accent-primary)', boxShadow: '0 0 8px var(--accent-primary)'
                      }} />
                    )}
                  </>
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* System Status Foot */}
        <div style={{ marginTop: 'auto', padding: '0 24px' }}>
          <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>System Engine</span>
              <span className="glow-text-success" style={{ fontSize: '0.75rem', fontWeight: 600 }}>ONLINE</span>
            </div>
            <div style={{ display: 'flex', gap: '6px' }}>
              {['DB', 'REDIS', 'KITE'].map(sys => (
                <div key={sys} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--success)' }} />
                  <span style={{ fontSize: '0.6rem', color: 'var(--text-tertiary)' }}>{sys}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main style={{ marginLeft: '260px', flex: 1, padding: '32px', overflowY: 'auto', height: '100vh', paddingBottom: '64px' }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
          {children}
        </div>
      </main>
    </div>
  );
}
