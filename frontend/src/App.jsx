import React, { useState, useEffect } from 'react';
import { 
  Search, Video, ShieldAlert, Sparkles, Clock, Play, 
  X, Filter, RefreshCw, Settings, Sliders, ExternalLink, Download,
  Check, Save, Plus, Trash2, Camera, Bell, HardDrive, Cpu, AlertTriangle, Send, Film
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
  // Navigation
  const [activeTab, setActiveTab] = useState('studio'); // 'studio' | 'settings'

  // Studio state
  const [query, setQuery] = useState('');
  const [limit, setLimit] = useState(12);
  const [clips, setClips] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeFilter, setActiveFilter] = useState('ALL');
  const [selectedClip, setSelectedClip] = useState(null);
  const [isSearchResult, setIsSearchResult] = useState(false);
  const [lastQuery, setLastQuery] = useState('');

  // System & Capture Health state
  const [systemStatus, setSystemStatus] = useState({ online: true, modelLoaded: true });
  const [captureStatus, setCaptureStatus] = useState({ watching: false, status: 'standby' });

  // Rules & Actions state
  const [rules, setRules] = useState([]);
  const [actions, setActions] = useState([]);
  const [newRuleDesc, setNewRuleDesc] = useState('');
  const [newRuleActionId, setNewRuleActionId] = useState(2);
  const [rulesReloading, setRulesReloading] = useState(false);

  // Dynamic Settings state
  const [settings, setSettings] = useState({
    camera: { source: 0, fps: 15, resolution: [640, 480] },
    recording: { clip_duration_sec: 5, motion_threshold: 25, min_motion_area_pct: 1.5 },
    storage: { clip_dir: './data/clips' },
    model: { keyframes_per_clip: 3 },
    rules: { confidence_threshold: 0.05 },
    telegram: { enabled: false, bot_token: '', chat_id: '' }
  });
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [telegramTesting, setTelegramTesting] = useState(false);
  const [telegramResult, setTelegramResult] = useState(null);

  // Initial load
  useEffect(() => {
    fetchRecentClips();
    checkHealth();
    checkCaptureStatus();
    fetchRulesAndActions();
    fetchSettings();

    // Periodically update health and capture status every 10s only when browser tab is active
    const interval = setInterval(() => {
      if (!document.hidden) {
        checkHealth();
        checkCaptureStatus();
      }
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  // Keyboard shortcut: Escape to close video player modal
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setSelectedClip(null);
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
      setSystemStatus({ online: false, modelLoaded: false });
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
      setCaptureStatus({ watching: false, status: 'standby' });
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

  const fetchSettings = async () => {
    try {
      const res = await fetch('/api/settings');
      if (res.ok) {
        const data = await res.json();
        setSettings(data);
      }
    } catch (e) {
      console.error("Failed to fetch settings", e);
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
        body: JSON.stringify({
          description: newRuleDesc.trim(),
          action_id: parseInt(newRuleActionId, 10)
        })
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

  const handleReloadRules = async () => {
    setRulesReloading(true);
    try {
      await fetch('/rules/reload', { method: 'POST' });
    } catch (e) {
      console.error("Failed to reload rules", e);
    } finally {
      setRulesReloading(false);
    }
  };

  const handleSaveSettings = async (e) => {
    if (e) e.preventDefault();
    setSettingsSaving(true);
    setSaveSuccess(false);

    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
      if (res.ok) {
        const data = await res.json();
        setSettings(data.settings);
        setSaveSuccess(true);
        setTimeout(() => setSaveSuccess(false), 3000);
      }
    } catch (e) {
      console.error("Failed to save settings", e);
    } finally {
      setSettingsSaving(false);
    }
  };

  const handleTestTelegram = async () => {
    setTelegramTesting(true);
    setTelegramResult(null);

    try {
      const res = await fetch('/api/telegram/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bot_token: settings.telegram.bot_token,
          chat_id: settings.telegram.chat_id
        })
      });
      const data = await res.json();
      if (res.ok) {
        setTelegramResult({ success: true, message: data.message || "Test alert delivered!" });
      } else {
        setTelegramResult({ success: false, message: data.detail || "Failed to deliver test alert." });
      }
    } catch (e) {
      setTelegramResult({ success: false, message: "Network connection error" });
    } finally {
      setTelegramTesting(false);
    }
  };

  // Filter clips in Studio
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

        {/* Tab Switcher */}
        <div className="nav-tabs">
          <button 
            className={`nav-tab-btn ${activeTab === 'studio' ? 'active' : ''}`}
            onClick={() => setActiveTab('studio')}
          >
            <Film size={16} />
            <span>Events Studio</span>
          </button>
          <button 
            className={`nav-tab-btn ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
          >
            <Sliders size={16} />
            <span>Settings & Rules</span>
          </button>
        </div>

        {/* Health Status & Refresh */}
        <div className="header-actions">
          {/* AI Model Status */}
          <div className={`status-pill ${systemStatus.online && systemStatus.modelLoaded ? 'status-pill-active' : 'status-pill-inactive'}`}>
            <span className="status-dot"></span>
            <span>{systemStatus.online ? 'Model Resident & Live' : 'Connecting...'}</span>
          </div>

          {/* Capture Daemon Status */}
          <div 
            className={`status-pill ${captureStatus.watching ? 'status-pill-active' : 'status-pill-standby'}`}
            title={`Capture Daemon (${captureStatus.watching ? 'Active & Watching' : 'Disabled / Standby'})`}
          >
            <span className="status-dot"></span>
            <span>{captureStatus.watching ? 'Capture Active' : 'Capture Disabled'}</span>
          </div>

          <button 
            className="btn-secondary"
            onClick={() => {
              if (activeTab === 'studio') fetchRecentClips();
              else {
                fetchSettings();
                fetchRulesAndActions();
              }
            }}
            title="Refresh"
          >
            <RefreshCw size={16} className={loading ? 'spinner' : ''} />
          </button>
        </div>
      </header>

      {/* ===================== TAB 1: STUDIO ===================== */}
      {activeTab === 'studio' && (
        <main className="fade-in">
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

          {/* Filter Bar & Feed Controls */}
          <section className="feed-header-bar">
            <div className="filter-group">
              <button 
                className={`filter-btn ${activeFilter === 'ALL' ? 'active' : ''}`}
                onClick={() => setActiveFilter('ALL')}
              >
                All Clips
              </button>
              <button 
                className={`filter-btn ${activeFilter === 'SAVED' ? 'active' : ''}`}
                onClick={() => setActiveFilter('SAVED')}
              >
                Saved
              </button>
              <button 
                className={`filter-btn ${activeFilter === 'ALERT' ? 'active' : ''}`}
                onClick={() => setActiveFilter('ALERT')}
              >
                Alerted
              </button>
            </div>

            <div className="feed-status-info">
              {isSearchResult ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span className="search-query-tag">Query: "{lastQuery}"</span>
                  <button 
                    className="clear-search-btn"
                    onClick={() => {
                      setQuery('');
                      fetchRecentClips();
                    }}
                  >
                    Clear Search
                  </button>
                </div>
              ) : (
                <span className="feed-count-text">
                  Showing <strong>{filteredClips.length}</strong> recent activity clips
                </span>
              )}
            </div>
          </section>

          {/* Video Clips Grid */}
          <section className="clips-grid">
            {filteredClips.map((clip) => {
              const isAlert = clip.action === 'ALERTED';
              return (
                <article 
                  key={clip.clip_id}
                  className={`clip-card glass-card ${isAlert ? 'clip-alert-card' : ''}`}
                  onClick={() => setSelectedClip(clip)}
                >
                  <div className="thumbnail-container">
                    {clip.thumbnail_url ? (
                      <img 
                        src={clip.thumbnail_url} 
                        alt={`Clip ${clip.clip_id} keyframe`}
                        className="clip-thumb-img"
                        loading="lazy"
                        onError={(e) => {
                          e.target.style.display = 'none';
                        }}
                      />
                    ) : (
                      <div className="thumb-placeholder">
                        <Video size={36} color="var(--text-muted)" />
                        <span style={{ fontSize: '0.75rem', marginTop: '6px' }}>Clip #{clip.clip_id}</span>
                      </div>
                    )}

                    <div className="thumbnail-overlay">
                      <div className="play-circle">
                        <Play size={20} fill="white" style={{ marginLeft: '2px' }} />
                      </div>
                    </div>

                    <div className="badge-row-top">
                      <span className={`action-badge ${isAlert ? 'alerted' : 'saved'}`}>
                        {isAlert ? <ShieldAlert size={12} style={{ marginRight: '4px' }} /> : null}
                        {clip.action || 'SAVED'}
                      </span>
                    </div>

                    <div className="badge-row-bottom">
                      <span className="duration-pill">
                        {clip.duration_sec ? `${clip.duration_sec.toFixed(1)}s` : '5.0s'}
                      </span>
                    </div>
                  </div>

                  <div className="clip-meta-content">
                    <div className="clip-time-row">
                      <div className="clip-timestamp">
                        <Clock size={13} />
                        <span>{formatTimestamp(clip.timestamp)}</span>
                      </div>
                      <span className="clip-id-tag">#{clip.clip_id}</span>
                    </div>

                    {clip.distance !== undefined && (
                      <div className="distance-indicator">
                        <div className="distance-bar-wrapper">
                          <div 
                            className="distance-bar-fill" 
                            style={{ width: `${Math.max(5, Math.min(100, (2 - clip.distance) * 50))}%` }}
                          />
                        </div>
                        <span className="distance-score">
                          {((2 - clip.distance) / 2).toFixed(2)} match
                        </span>
                      </div>
                    )}
                  </div>
                </article>
              );
            })}
          </section>

          {/* Empty State */}
          {!loading && filteredClips.length === 0 && (
            <div className="empty-state glass-card">
              <div className="empty-icon">
                <Video size={32} />
              </div>
              <h3 className="empty-title">No Video Clips Found</h3>
              <p className="empty-desc">
                {isSearchResult 
                  ? "No clips closely match your query. Try broadening your terms or clear search."
                  : "No events recorded yet. CheckCam records clips automatically when motion is detected."}
              </p>
              {isSearchResult && (
                <button 
                  className="btn-primary"
                  onClick={() => {
                    setQuery('');
                    fetchRecentClips();
                  }}
                  style={{ marginTop: '12px' }}
                >
                  Show All Recent Clips
                </button>
              )}
            </div>
          )}
        </main>
      )}

      {/* ===================== TAB 2: SETTINGS & RULES ===================== */}
      {activeTab === 'settings' && (
        <main className="settings-container fade-in">
          <div className="settings-grid">

            {/* SECTION 1: Zero-Shot Classification Rules (Full Width) */}
            <div className="settings-card glass-card full-width">
              <div className="settings-card-header">
                <div>
                  <div className="settings-card-title">
                    <Sliders size={20} color="var(--accent-cyan)" />
                    <span>Zero-Shot AI Classification Rules</span>
                  </div>
                  <p className="settings-card-desc">
                    Clips are classified against these prompts using SigLIP INT8 zero-shot inference. Events automatically route to Discard, Save, or Telegram Alert.
                  </p>
                </div>
                <button 
                  className="btn-secondary"
                  onClick={handleReloadRules}
                  disabled={rulesReloading}
                  title="Re-encode all rule prompts into in-memory vectors"
                >
                  <RefreshCw size={14} className={rulesReloading ? 'spinner' : ''} />
                  <span>{rulesReloading ? 'Re-encoding...' : 'Reload In-Memory'}</span>
                </button>
              </div>

              {/* Rules List */}
              <div className="rules-list-grid">
                {rules.map((rule) => {
                  const isAlert = rule.action_name === 'alert';
                  const isDiscard = rule.action_name === 'discard';
                  return (
                    <div key={rule.id} className="rule-card-item">
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <span className={`action-badge ${isAlert ? 'alerted' : isDiscard ? 'discarded' : 'saved'}`} style={{ fontSize: '0.68rem' }}>
                          {rule.action_name.toUpperCase()}
                        </span>
                        <span style={{ fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                          "{rule.description}"
                        </span>
                      </div>
                      <button 
                        onClick={() => handleDeleteRule(rule.id)}
                        style={{ color: '#ef4444', padding: '4px', background: 'transparent', border: 'none', cursor: 'pointer' }}
                        title="Delete rule"
                      >
                        <Trash2 size={15} />
                      </button>
                    </div>
                  );
                })}
              </div>

              {/* Add Rule Form */}
              <form onSubmit={handleAddRule} className="add-rule-box">
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  Add Classification Rule
                </span>
                <div className="add-rule-form-row">
                  <input
                    type="text"
                    className="setting-input"
                    placeholder="Enter visual prompt (e.g. 'a person holding a package', 'a cat on the sofa')"
                    value={newRuleDesc}
                    onChange={(e) => setNewRuleDesc(e.target.value)}
                    style={{ flex: 1 }}
                  />
                  <select
                    className="setting-select"
                    value={newRuleActionId}
                    onChange={(e) => setNewRuleActionId(e.target.value)}
                    style={{ width: '140px' }}
                  >
                    {actions.map((act) => (
                      <option key={act.id} value={act.id}>
                        {act.command}
                      </option>
                    ))}
                  </select>
                  <button type="submit" className="btn-primary" style={{ padding: '10px 18px' }}>
                    <Plus size={16} />
                    <span>Add</span>
                  </button>
                </div>
              </form>
            </div>

            {/* SECTION 2: Camera Hardware */}
            <div className="settings-card glass-card">
              <div className="settings-card-header">
                <div className="settings-card-title">
                  <Camera size={19} color="var(--accent-cyan)" />
                  <span>Camera Hardware</span>
                </div>
              </div>

              <div className="setting-row">
                <div className="setting-label-row">
                  <span className="setting-label">Device Source</span>
                  <span className="setting-val-badge">{String(settings.camera.source)}</span>
                </div>
                <input 
                  type="text"
                  className="setting-input"
                  value={settings.camera.source}
                  onChange={(e) => {
                    const val = e.target.value;
                    const parsed = /^\d+$/.test(val) ? parseInt(val, 10) : val;
                    setSettings({ ...settings, camera: { ...settings.camera, source: parsed } });
                  }}
                  placeholder="0, 'picam', or rtsp://..."
                />
                <span className="setting-hint">
                  Use <code>0</code> or <code>1</code> for USB webcam, <code>picam</code> for Raspberry Pi Camera Module, or an RTSP stream URL.
                </span>
              </div>

              <div className="setting-row">
                <div className="setting-label-row">
                  <span className="setting-label">Capture Framerate</span>
                  <span className="setting-val-badge">{settings.camera.fps} FPS</span>
                </div>
                <input 
                  type="range"
                  min="5"
                  max="30"
                  step="1"
                  className="setting-range"
                  value={settings.camera.fps}
                  onChange={(e) => setSettings({
                    ...settings, 
                    camera: { ...settings.camera, fps: parseInt(e.target.value, 10) }
                  })}
                />
                <span className="setting-hint">Lower FPS (15) reduces CPU and heat on Raspberry Pi 4.</span>
              </div>

              <div className="setting-row">
                <div className="setting-label-row">
                  <span className="setting-label">Resolution</span>
                  <span className="setting-val-badge">
                    {settings.camera.resolution ? `${settings.camera.resolution[0]}x${settings.camera.resolution[1]}` : '640x480'}
                  </span>
                </div>
                <select 
                  className="setting-select"
                  value={`${settings.camera.resolution[0]}x${settings.camera.resolution[1]}`}
                  onChange={(e) => {
                    const parts = e.target.value.split('x').map(Number);
                    setSettings({ ...settings, camera: { ...settings.camera, resolution: parts } });
                  }}
                >
                  <option value="640x480">640x480 (SD — Recommended for Pi 4)</option>
                  <option value="1280x720">1280x720 (HD 720p)</option>
                  <option value="1920x1080">1920x1080 (Full HD 1080p)</option>
                </select>
              </div>
            </div>

            {/* SECTION 3: Motion & Clip Recording */}
            <div className="settings-card glass-card">
              <div className="settings-card-header">
                <div className="settings-card-title">
                  <Film size={19} color="var(--accent-emerald)" />
                  <span>Motion & Clip Recording</span>
                </div>
              </div>

              <div className="setting-row">
                <div className="setting-label-row">
                  <span className="setting-label">Clip Duration</span>
                  <span className="setting-val-badge">{settings.recording.clip_duration_sec}s (Max 10s)</span>
                </div>
                <input 
                  type="range"
                  min="1"
                  max="10"
                  step="1"
                  className="setting-range"
                  value={settings.recording.clip_duration_sec}
                  onChange={(e) => setSettings({
                    ...settings, 
                    recording: { ...settings.recording, clip_duration_sec: Math.min(10, Math.max(1, parseInt(e.target.value, 10))) }
                  })}
                />
                <span className="setting-hint">Duration of each event recording. Strictly capped at 10 seconds.</span>
              </div>

              <div className="setting-row">
                <div className="setting-label-row">
                  <span className="setting-label">Motion Pixel Threshold</span>
                  <span className="setting-val-badge">{settings.recording.motion_threshold}</span>
                </div>
                <input 
                  type="range"
                  min="5"
                  max="80"
                  step="1"
                  className="setting-range"
                  value={settings.recording.motion_threshold}
                  onChange={(e) => setSettings({
                    ...settings, 
                    recording: { ...settings.recording, motion_threshold: parseInt(e.target.value, 10) }
                  })}
                />
                <span className="setting-hint">Intensity delta (0-255) to trigger motion differencing. Default 25.</span>
              </div>

              <div className="setting-row">
                <div className="setting-label-row">
                  <span className="setting-label">Min Motion Area %</span>
                  <span className="setting-val-badge">{settings.recording.min_motion_area_pct}%</span>
                </div>
                <input 
                  type="range"
                  min="0.2"
                  max="10.0"
                  step="0.1"
                  className="setting-range"
                  value={settings.recording.min_motion_area_pct}
                  onChange={(e) => setSettings({
                    ...settings, 
                    recording: { ...settings.recording, min_motion_area_pct: parseFloat(e.target.value) }
                  })}
                />
                <span className="setting-hint">Min percentage of total frame area that must change to start recording.</span>
              </div>
            </div>

            {/* SECTION 4: AI Inference & Sampling */}
            <div className="settings-card glass-card">
              <div className="settings-card-header">
                <div className="settings-card-title">
                  <Cpu size={19} color="var(--accent-purple)" />
                  <span>AI Inference & Thresholds</span>
                </div>
              </div>

              <div className="setting-row">
                <div className="setting-label-row">
                  <span className="setting-label">Keyframes Sampled Per Clip</span>
                  <span className="setting-val-badge">{settings.model.keyframes_per_clip} Frames</span>
                </div>
                <input 
                  type="range"
                  min="1"
                  max="7"
                  step="1"
                  className="setting-range"
                  value={settings.model.keyframes_per_clip}
                  onChange={(e) => setSettings({
                    ...settings, 
                    model: { ...settings.model, keyframes_per_clip: parseInt(e.target.value, 10) }
                  })}
                />
                <span className="setting-hint">Uniformly sampled keyframes fed into SigLIP vision model. 3 is optimal.</span>
              </div>

              <div className="setting-row">
                <div className="setting-label-row">
                  <span className="setting-label">Rule Confidence Threshold</span>
                  <span className="setting-val-badge">{settings.rules.confidence_threshold.toFixed(2)}</span>
                </div>
                <input 
                  type="range"
                  min="0.01"
                  max="0.30"
                  step="0.01"
                  className="setting-range"
                  value={settings.rules.confidence_threshold}
                  onChange={(e) => setSettings({
                    ...settings, 
                    rules: { ...settings.rules, confidence_threshold: parseFloat(e.target.value) }
                  })}
                />
                <span className="setting-hint">Below this cosine similarity score, clips are automatically DISCARDED.</span>
              </div>
            </div>

            {/* SECTION 5: Storage Directory */}
            <div className="settings-card glass-card">
              <div className="settings-card-header">
                <div className="settings-card-title">
                  <HardDrive size={19} color="var(--accent-cyan)" />
                  <span>Storage Location</span>
                </div>
              </div>

              <div className="setting-row">
                <div className="setting-label-row">
                  <span className="setting-label">Video Clips Directory</span>
                  <span className="setting-val-badge">{settings.storage.clip_dir}</span>
                </div>
                <input 
                  type="text"
                  className="setting-input"
                  value={settings.storage.clip_dir}
                  onChange={(e) => setSettings({
                    ...settings,
                    storage: { ...settings.storage, clip_dir: e.target.value }
                  })}
                  placeholder="./data/clips"
                />
                <span className="setting-hint">Local filesystem directory where recorded MP4 video clips are stored.</span>
              </div>
            </div>

            {/* SECTION 6: Telegram Notifications (Full Width) */}
            <div className="settings-card glass-card full-width">
              <div className="settings-card-header">
                <div className="settings-card-title">
                  <Bell size={19} color="var(--accent-rose)" />
                  <span>Telegram Alert Notifications</span>
                </div>

                <div className="toggle-wrapper" style={{ gap: '12px' }}>
                  <span style={{ fontSize: '0.85rem', color: settings.telegram.enabled ? 'var(--accent-emerald)' : 'var(--text-muted)' }}>
                    {settings.telegram.enabled ? 'Enabled' : 'Disabled'}
                  </span>
                  <label className="switch">
                    <input 
                      type="checkbox"
                      checked={settings.telegram.enabled}
                      onChange={(e) => setSettings({
                        ...settings,
                        telegram: { ...settings.telegram, enabled: e.target.checked }
                      })}
                    />
                    <span className="slider-toggle"></span>
                  </label>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                <div className="setting-row">
                  <span className="setting-label">Bot Token</span>
                  <input 
                    type="password"
                    className="setting-input"
                    value={settings.telegram.bot_token}
                    onChange={(e) => setSettings({
                      ...settings,
                      telegram: { ...settings.telegram, bot_token: e.target.value }
                    })}
                    placeholder="123456789:ABCdefGHIjklMNOpqrSTUvwxYZ"
                  />
                  <span className="setting-hint">Token obtained from @BotFather.</span>
                </div>

                <div className="setting-row">
                  <span className="setting-label">Chat ID</span>
                  <input 
                    type="text"
                    className="setting-input"
                    value={settings.telegram.chat_id}
                    onChange={(e) => setSettings({
                      ...settings,
                      telegram: { ...settings.telegram, chat_id: e.target.value }
                    })}
                    placeholder="e.g. 987654321"
                  />
                  <span className="setting-hint">Your Telegram user ID or alert group ID.</span>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '16px', paddingTop: '8px' }}>
                <button 
                  type="button"
                  className="btn-secondary"
                  onClick={handleTestTelegram}
                  disabled={telegramTesting || !settings.telegram.bot_token || !settings.telegram.chat_id}
                >
                  <Send size={14} className={telegramTesting ? 'spinner' : ''} />
                  <span>{telegramTesting ? 'Sending test...' : 'Send Test Notification'}</span>
                </button>

                {telegramResult && (
                  <span style={{ 
                    fontSize: '0.85rem', 
                    color: telegramResult.success ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}>
                    {telegramResult.success ? <Check size={16} /> : <AlertTriangle size={16} />}
                    {telegramResult.message}
                  </span>
                )}
              </div>
            </div>

          </div>

          {/* Sticky Floating Save Bar */}
          <div className="save-bar">
            <div className="save-bar-info">
              <span>Settings are kept in SQLite and applied live to all processes.</span>
              {saveSuccess && (
                <div className="save-feedback-badge">
                  <Check size={16} />
                  <span>Settings Saved Successfully!</span>
                </div>
              )}
            </div>

            <button 
              className="btn-primary"
              onClick={handleSaveSettings}
              disabled={settingsSaving}
              style={{ minWidth: '150px', padding: '10px 22px' }}
            >
              <Save size={16} className={settingsSaving ? 'spinner' : ''} />
              <span>{settingsSaving ? 'Saving...' : 'Save Settings'}</span>
            </button>
          </div>
        </main>
      )}

      {/* Video Player Modal */}
      {selectedClip && (
        <div className="modal-overlay" onClick={() => setSelectedClip(null)}>
          <div className="modal-content glass-card fade-in" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span className={`action-badge ${selectedClip.action === 'ALERTED' ? 'alerted' : 'saved'}`}>
                  {selectedClip.action || 'SAVED'}
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
    </div>
  );
}
