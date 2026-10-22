import React, { useState, useEffect } from 'react';
import { 
  Search, Video, ShieldAlert, Sparkles, Clock, Play, 
  X, Filter, RefreshCw, Settings, Sliders, ExternalLink, Download
} from 'lucide-react';
import './App.css';

const QUICK_SUGGESTIONS = [
  "a person walking",
  "a person with a bottle",
  "someone holding a phone",
  "dog or cat moving around",
  "vehicle pulling into driveway",
  "delivery driver with a package"
];

export default function App() {
  const [query, setQuery] = useState('');
  const [limit, setLimit] = useState(12);
  const [clips, setClips] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeFilter, setActiveFilter] = useState('ALL');
  const [selectedClip, setSelectedClip] = useState(null);
  const [systemStatus, setSystemStatus] = useState({ online: true, modelLoaded: true });
  const [captureStatus, setCaptureStatus] = useState({ watching: false, status: 'standby' });
  const [isSearchResult, setIsSearchResult] = useState(false);
  const [lastQuery, setLastQuery] = useState('');
  const [rulesModalOpen, setRulesModalOpen] = useState(false);
  const [rules, setRules] = useState([]);
  const [actions, setActions] = useState([]);
  const [newRuleDesc, setNewRuleDesc] = useState('');
  const [newRuleActionId, setNewRuleActionId] = useState(2);

  // Initial load: fetch clips, health, capture status & start polling
  useEffect(() => {
    fetchRecentClips();
    checkHealth();
    checkCaptureStatus();
    fetchRulesAndActions();

    const interval = setInterval(() => {
      checkHealth();
      checkCaptureStatus();
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  // Keyboard shortcut: Escape to close modal
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setSelectedClip(null);
        setRulesModalOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        setSystemStatus({
          online: true,
          modelLoaded: data.model_loaded
        });
      }
    } catch (e) {
      setSystemStatus({
        online: false,
        modelLoaded: false
      });
    }
  };

  const checkCaptureStatus = async () => {
    try {
      const res = await fetch('/api/capture/status');
      if (res.ok) {
        const data = await res.json();
        setCaptureStatus({
          watching: Boolean(data.watching),
          status: data.status || 'standby'
        });
      }
    } catch (e) {
      setCaptureStatus({
        watching: false,
        status: 'standby'
      });
    }
  };

  const fetchRecentClips = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/clips/recent?limit=${limit}`);
      if (res.ok) {
        const data = await res.json();
        setClips(data);
        setIsSearchResult(false);
      }
    } catch (e) {
      console.error("Failed to fetch recent clips", e);
    } finally {
      setLoading(false);
    }
  };

  const fetchRulesAndActions = async () => {
    try {
      const [rulesRes, actionsRes] = await Promise.all([
        fetch('/rules'),
        fetch('/actions')
      ]);
      if (rulesRes.ok) setRules(await rulesRes.json());
      if (actionsRes.ok) setActions(await actionsRes.json());
    } catch (e) {
      console.error("Failed to fetch rules", e);
    }
  };

  const handleSearch = async (searchQuery = query) => {
    const q = searchQuery.trim();
    if (!q) {
      fetchRecentClips();
      return;
    }

    setLoading(true);
    try {
      const res = await fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, limit: parseInt(limit, 10) })
      });
      if (res.ok) {
        const data = await res.json();
        setClips(data.results || []);
        setIsSearchResult(true);
        setLastQuery(q);
      }
    } catch (e) {
      console.error("Search failed", e);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDownOnTextarea = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  };

  const handleAddRule = async (e) => {
    e.preventDefault();
    if (!newRuleDesc.trim()) return;
    try {
      const res = await fetch('/rules', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: newRuleDesc.trim(), action_id: newRuleActionId })
      });
      if (res.ok) {
        setNewRuleDesc('');
        fetchRulesAndActions();
      }
    } catch (e) {
      console.error("Failed to add rule", e);
    }
  };

  const handleDeleteRule = async (ruleId) => {
    try {
      const res = await fetch(`/rules/${ruleId}`, { method: 'DELETE' });
      if (res.ok) {
        fetchRulesAndActions();
      }
    } catch (e) {
      console.error("Failed to delete rule", e);
    }
  };

  // Filter clips
  const filteredClips = clips.filter(clip => {
    if (activeFilter === 'ALERT') return clip.action === 'ALERTED';
    if (activeFilter === 'SAVE') return clip.action === 'SAVED';
    return true;
  });

  const formatTimestamp = (ts) => {
    if (!ts) return '';
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) + 
        ' · ' + d.toLocaleDateString([], { month: 'short', day: 'numeric' });
    } catch (e) {
      return ts;
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header glass">
        <div className="logo-area">
          <div className="logo-icon-wrapper">
            <Video size={24} />
          </div>
          <div>
            <h1 className="logo-title">CheckCam</h1>
            <p className="logo-subtitle">Local AI Security Studio</p>
          </div>
        </div>

        <div className="header-actions">
          {/* AI Model Status */}
          <div className={`status-pill ${systemStatus.online && systemStatus.modelLoaded ? 'status-pill-active' : 'status-pill-inactive'}`}>
            <span className="status-dot"></span>
            <span>{systemStatus.online ? 'Model Resident & Live' : 'Connecting...'}</span>
          </div>

          {/* Capture Daemon Status: Live Watching vs Disabled/Standby */}
          <div 
            className={`status-pill ${captureStatus.watching ? 'status-pill-active' : 'status-pill-standby'}`}
            title={`Capture Daemon (${captureStatus.watching ? 'Active & Watching' : 'Disabled / Standby'})`}
          >
            <span className="status-dot"></span>
            <span>{captureStatus.watching ? 'Capture Active' : 'Capture Disabled'}</span>
          </div>

          <button 
            className="btn-secondary"
            onClick={() => setRulesModalOpen(true)}
            title="Classification Rules"
          >
            <Sliders size={16} />
            <span>Rules ({rules.length})</span>
          </button>

          <button 
            className="btn-secondary"
            onClick={fetchRecentClips}
            title="Refresh latest camera events"
          >
            <RefreshCw size={16} className={loading && !isSearchResult ? 'spinner' : ''} />
          </button>
        </div>
      </header>

      {/* Main Search Section */}
      <section className="search-card glass-card">
        <div className="search-header-row">
          <div className="search-label">
            <Sparkles size={16} color="var(--accent-cyan)" />
            <span>Natural Language Semantic Video Search</span>
          </div>
          <span className="search-hint">Press Enter ↵ to search</span>
        </div>

        <div className="search-input-wrapper">
          <textarea
            className="search-textarea"
            placeholder="Describe what you are looking for in natural language... (e.g. 'a person walking past', 'someone holding a bottle', 'car entering driveway')"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDownOnTextarea}
            rows={2}
          />
        </div>

        <div className="search-bottom-bar">
          <div className="suggestions-row">
            <span className="suggestions-label">Try:</span>
            {QUICK_SUGGESTIONS.map((sug, i) => (
              <button 
                key={i} 
                className="suggestion-chip"
                onClick={() => {
                  setQuery(sug);
                  handleSearch(sug);
                }}
              >
                {sug}
              </button>
            ))}
          </div>

          <div className="search-controls">
            <select 
              className="limit-select"
              value={limit}
              onChange={(e) => setLimit(e.target.value)}
            >
              <option value="6">Top 6</option>
              <option value="12">Top 12</option>
              <option value="24">Top 24</option>
              <option value="48">Top 48</option>
            </select>

            <button 
              className="btn-primary"
              onClick={() => handleSearch()}
              disabled={loading}
            >
              <Search size={18} className={loading ? 'spinner' : ''} />
              <span>{loading ? 'Searching...' : 'Search Clips'}</span>
            </button>
          </div>
        </div>
      </section>

      {/* Results Header */}
      <div className="results-header-bar">
        <div className="results-title-group">
          <h2 className="results-heading">
            {isSearchResult ? `Results for "${lastQuery}"` : 'Recent Camera Events'}
          </h2>
          <span className="results-count">
            {filteredClips.length} {filteredClips.length === 1 ? 'clip' : 'clips'}
          </span>
        </div>

        <div className="filter-tabs">
          <button 
            className={`tab-btn ${activeFilter === 'ALL' ? 'active' : ''}`}
            onClick={() => setActiveFilter('ALL')}
          >
            All ({clips.length})
          </button>
          <button 
            className={`tab-btn ${activeFilter === 'ALERT' ? 'active' : ''}`}
            onClick={() => setActiveFilter('ALERT')}
          >
            Alerts ({clips.filter(c => c.action === 'ALERTED').length})
          </button>
          <button 
            className={`tab-btn ${activeFilter === 'SAVE' ? 'active' : ''}`}
            onClick={() => setActiveFilter('SAVE')}
          >
            Saved ({clips.filter(c => c.action === 'SAVED').length})
          </button>
        </div>
      </div>

      {/* Grid of Clips */}
      {filteredClips.length === 0 ? (
        <div className="empty-state glass-card fade-in">
          <div className="empty-icon">
            <Video size={32} />
          </div>
          <h3 className="empty-title">
            {loading ? 'Analyzing multimodal embeddings...' : 'No matching clips found'}
          </h3>
          <p className="empty-desc">
            {isSearchResult 
              ? 'Try adjusting your search query, selecting different words, or browsing recent events.'
              : 'Move in front of the camera to trigger recording, or verify your camera daemon is running.'}
          </p>
        </div>
      ) : (
        <div className="clips-grid">
          {filteredClips.map((clip) => (
            <div 
              key={clip.clip_id} 
              className="clip-card glass-card fade-in"
              onClick={() => setSelectedClip(clip)}
            >
              <div className="thumbnail-container">
                {clip.thumbnail_url ? (
                  <img 
                    src={clip.thumbnail_url} 
                    alt={`Clip ${clip.clip_id}`}
                    className="clip-thumbnail"
                    loading="lazy"
                    onError={(e) => {
                      e.target.style.display = 'none';
                      e.target.nextSibling.style.display = 'flex';
                    }}
                  />
                ) : null}
                <div 
                  className="thumbnail-fallback"
                  style={{ display: clip.thumbnail_url ? 'none' : 'flex' }}
                >
                  <Video size={28} />
                  <span>Clip #{clip.clip_id}</span>
                </div>

                <div className="thumbnail-overlay">
                  <div className="play-circle">
                    <Play size={24} fill="#051026" />
                  </div>
                </div>

                <div className="badge-top-left">
                  <span className={`action-badge ${clip.action === 'ALERTED' ? 'alerted' : 'saved'}`}>
                    {clip.action === 'ALERTED' ? <ShieldAlert size={12} /> : null}
                    {clip.action || 'SAVED'}
                  </span>
                </div>

                {clip.distance !== undefined && (
                  <div className="similarity-badge">
                    {Math.max(0, Math.min(100, Math.round((2 - clip.distance) * 50)))}% match
                  </div>
                )}

                <div className="duration-tag">
                  {clip.duration_sec ? `${clip.duration_sec.toFixed(1)}s` : '5.0s'}
                </div>
              </div>

              <div className="clip-info">
                <div className="clip-title-row">
                  <span className="clip-id">Clip #{clip.clip_id}</span>
                </div>
                <div className="clip-timestamp">
                  <Clock size={13} />
                  <span>{formatTimestamp(clip.timestamp)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Video Modal Player */}
      {selectedClip && (
        <div className="modal-overlay" onClick={() => setSelectedClip(null)}>
          <div className="modal-content glass-card fade-in" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span className={`action-badge ${selectedClip.action === 'ALERTED' ? 'alerted' : 'saved'}`}>
                  {selectedClip.action}
                </span>
                <h3 className="modal-title">Clip #{selectedClip.clip_id}</h3>
              </div>
              <button 
                className="modal-close-btn"
                onClick={() => setSelectedClip(null)}
              >
                <X size={20} />
              </button>
            </div>

            <div className="modal-video-container">
              <video 
                src={`/api/clips/${selectedClip.clip_id}/video`}
                controls 
                autoPlay 
                loop 
                className="modal-video"
              />
            </div>

            <div className="modal-footer">
              <div className="clip-timestamp">
                <Clock size={14} />
                <span>{formatTimestamp(selectedClip.timestamp)}</span>
                <span style={{ margin: '0 8px' }}>•</span>
                <span>{selectedClip.file_path}</span>
              </div>
              <a 
                href={`/api/clips/${selectedClip.clip_id}/video`}
                download={`clip_${selectedClip.clip_id}.mp4`}
                className="btn-secondary"
                style={{ textDecoration: 'none' }}
              >
                <Download size={15} />
                <span>Download MP4</span>
              </a>
            </div>
          </div>
        </div>
      )}

      {/* Rules Manager Modal */}
      {rulesModalOpen && (
        <div className="modal-overlay" onClick={() => setRulesModalOpen(false)}>
          <div className="modal-content glass-card fade-in" style={{ maxWidth: '640px' }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">Zero-Shot Classification Rules</h3>
              <button className="modal-close-btn" onClick={() => setRulesModalOpen(false)}>
                <X size={20} />
              </button>
            </div>

            <div style={{ padding: '20px', maxHeight: '420px', overflowY: 'auto' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '20px' }}>
                {rules.map((rule) => (
                  <div 
                    key={rule.id}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '10px 14px',
                      background: 'rgba(255, 255, 255, 0.03)',
                      borderRadius: '8px',
                      border: '1px solid var(--border-subtle)'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <span className={`action-badge ${rule.action_name === 'alert' ? 'alerted' : 'saved'}`} style={{ fontSize: '0.68rem' }}>
                        {rule.action_name}
                      </span>
                      <span style={{ fontSize: '0.9rem' }}>"{rule.description}"</span>
                    </div>
                    <button 
                      onClick={() => handleDeleteRule(rule.id)}
                      style={{ color: '#ef4444', padding: '4px' }}
                      title="Delete rule"
                    >
                      <X size={16} />
                    </button>
                  </div>
                ))}
              </div>

              {/* Add Rule Form */}
              <form onSubmit={handleAddRule} style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '16px' }}>
                <h4 style={{ fontSize: '0.85rem', marginBottom: '10px', color: 'var(--text-secondary)' }}>Add New Classification Rule</h4>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                  <input
                    type="text"
                    placeholder="Rule prompt (e.g. 'person on bicycle')"
                    value={newRuleDesc}
                    onChange={(e) => setNewRuleDesc(e.target.value)}
                    style={{
                      flex: 1,
                      padding: '10px 14px',
                      background: 'var(--bg-input)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: '8px',
                      fontSize: '0.85rem'
                    }}
                  />
                  <select
                    value={newRuleActionId}
                    onChange={(e) => setNewRuleActionId(parseInt(e.target.value, 10))}
                    style={{
                      padding: '10px 12px',
                      background: 'var(--bg-input)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: '8px',
                      color: 'var(--text-secondary)',
                      fontSize: '0.85rem'
                    }}
                  >
                    {actions.map(a => (
                      <option key={a.id} value={a.id}>{a.name.toUpperCase()}</option>
                    ))}
                  </select>
                  <button type="submit" className="btn-primary" style={{ padding: '10px 18px', fontSize: '0.85rem' }}>
                    Add
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
