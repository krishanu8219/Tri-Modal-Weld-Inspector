"use client";

import React, { useState, useEffect, useCallback } from 'react';
import styles from './page.module.css';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, ReferenceArea, BarChart, Bar, Cell, PieChart, Pie, Legend
} from 'recharts';
import {
  Activity, AlertTriangle, CheckCircle, Clock, Volume2, Image as ImageIcon,
  Database, TrendingUp, Download, FileText, BarChart3, Shield, Zap, Eye,
  Search, Filter, Cpu, GitBranch, Layers, Target
} from 'lucide-react';
import { useNavigation } from '@/components/NavigationContext';

const API = 'http://127.0.0.1:8000';

const LABEL_MAP: Record<string, string> = {
  '00': 'Good Weld', '01': 'Excessive Penetration', '02': 'Burn Through',
  '06': 'Overlap', '07': 'Lack of Fusion', '08': 'Excessive Convexity', '11': 'Crater Cracks'
};

const LABEL_COLORS: Record<string, string> = {
  '00': '#10b981', '01': '#ef4444', '02': '#f59e0b', '06': '#8b5cf6',
  '07': '#3b82f6', '08': '#ec4899', '11': '#f97316'
};

export default function Dashboard() {
  const { activePage, setActivePage } = useNavigation();
  const [logSearch, setLogSearch] = useState('');
  const [logFilter, setLogFilter] = useState<string>('all');
  const [runs, setRuns] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [diagnostics, setDiagnostics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  // Inspector state
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [inferData, setInferData] = useState<any>(null);
  const [inferLoading, setInferLoading] = useState(false);

  // Audio waveform state
  const [audioWaveform, setAudioWaveform] = useState<any>(null);

  // SHAP explanation state
  const [explanationData, setExplanationData] = useState<any>(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/test-runs`).then(r => r.json()).catch(() => ({ runs: [] })),
      fetch(`${API}/stats`).then(r => r.json()).catch(() => null),
      fetch(`${API}/metrics`).then(r => r.json()).catch(() => null),
      fetch(`${API}/diagnostics`).then(r => r.json()).catch(() => null),
    ]).then(([runsData, statsData, metricsData, diagData]) => {
      setRuns(runsData.runs || []);
      setStats(statsData);
      setMetrics(metricsData);
      setDiagnostics(diagData);
      setLoading(false);
    });
  }, []);

  const handleSelectRun = (run_id: string) => {
    // run_id may be sample_id from test data — extract it
    const sampleId = run_id.includes('sample_') ? run_id : `sample_${run_id}`;
    setSelectedRun(sampleId);
    setInferLoading(true);
    setInferData(null);
    setAudioWaveform(null);
    setExplanationData(null);
    fetch(`${API}/infer-test/${run_id}`)
      .then(res => res.json())
      .then(data => {
        setInferData(data);
        setInferLoading(false);
        // Fetch audio waveform if available
        if (data.audio) {
          fetch(`${API}/audio-waveform-test/${run_id}`)
            .then(r => r.json())
            .then(wf => setAudioWaveform(wf))
            .catch(() => { });
        }
      })
      .catch(() => setInferLoading(false));
  };

  const totalRuns = stats?.total_runs || runs.length;
  const anomalies = runs.filter((r: any) => (r.pred_label_code ?? r.label_code) !== '00').length;
  const passRate = totalRuns > 0 ? (((totalRuns - anomalies) / totalRuns) * 100).toFixed(1) : '0';

  // ——— NAVIGATION ———
  const navItems = [
    { key: 'overview', label: 'Dataset Overview', icon: <Database size={18} /> },
    { key: 'inspector', label: 'Run Inspector', icon: <Eye size={18} /> },
    { key: 'logs', label: 'Inspection Logs', icon: <FileText size={18} /> },
    { key: 'models', label: 'Data Models', icon: <Cpu size={18} /> },
    { key: 'evaluation', label: 'Evaluation Report', icon: <BarChart3 size={18} /> },
    { key: 'export', label: 'Export & Data Card', icon: <Download size={18} /> },
  ];

  // ——— Filtered runs for Inspection Logs ———
  const filteredRuns = runs.filter((r: any) => {
    const sid = r.sample_id || r.run_id || '';
    const matchesSearch = logSearch === '' || sid.toLowerCase().includes(logSearch.toLowerCase());
    const predCode = String(r.pred_label_code ?? r.label_code ?? '00').padStart(2, '0');
    const matchesFilter = logFilter === 'all'
      || (logFilter === 'pass' && predCode === '00')
      || (logFilter === 'fail' && predCode !== '00');
    return matchesSearch && matchesFilter;
  });

  return (
    <div className={`${styles.dashboard} animate-fade-in`}>
      {/* Header */}
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>🔬 Tri-Modal Weld Inspector</h1>
          <p className={styles.subtitle}>Automated Quality Assurance — Sensor · Acoustic · Vision</p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className={styles.tabs} style={{ flexWrap: 'wrap' }}>
        {navItems.map(item => (
          <button
            key={item.key}
            className={`${styles.tab} ${activePage === item.key ? styles.activeTab : ''}`}
            onClick={() => setActivePage(item.key as any)}
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
          >
            {item.icon} {item.label}
          </button>
        ))}
      </div>

      {/* ===== PAGE: DATASET OVERVIEW ===== */}
      {activePage === 'overview' && (
        <div className="animate-fade-in">
          {/* Top Stats Cards */}
          <div className={styles.statsGrid} style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
            <div className={`${styles.statCard} glass delay-100`}>
              <div className={styles.statIconWrapper} style={{ backgroundColor: 'rgba(59, 130, 246, 0.1)', color: 'var(--accent-primary)' }}>
                <Database size={24} />
              </div>
              <div className={styles.statContent}>
                <p className={styles.statLabel}>Total Runs</p>
                <h3 className={styles.statValue}>{loading ? '...' : totalRuns}</h3>
                <p className={styles.statTrend} style={{ color: 'var(--text-muted)' }}>All modalities</p>
              </div>
            </div>

            <div className={`${styles.statCard} glass delay-200`}>
              <div className={styles.statIconWrapper} style={{ backgroundColor: 'rgba(16, 185, 129, 0.1)', color: 'var(--success)' }}>
                <Shield size={24} />
              </div>
              <div className={styles.statContent}>
                <p className={styles.statLabel}>Training Pool</p>
                <h3 className={styles.statValue}>{loading ? '...' : stats?.training_pool || 0}</h3>
                <p className={styles.statTrend} style={{ color: 'var(--success)' }}>Labeled samples</p>
              </div>
            </div>

            <div className={`${styles.statCard} glass delay-300`}>
              <div className={styles.statIconWrapper} style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', color: 'var(--danger)' }}>
                <AlertTriangle size={24} />
              </div>
              <div className={styles.statContent}>
                <p className={styles.statLabel}>Ground Truth Defects</p>
                <h3 className={styles.statValue}>{loading ? '...' : anomalies}</h3>
                <p className={styles.statTrend} style={{ color: 'var(--danger)' }}>In training pool</p>
              </div>
            </div>

            <div className={`${styles.statCard} glass delay-300`}>
              <div className={styles.statIconWrapper} style={{ backgroundColor: 'rgba(245, 158, 11, 0.1)', color: '#f59e0b' }}>
                <Zap size={24} />
              </div>
              <div className={styles.statContent}>
                <p className={styles.statLabel}>Test Holdout</p>
                <h3 className={styles.statValue}>{loading ? '...' : stats?.test_samples || 0}</h3>
                <p className={styles.statTrend} style={{ color: 'var(--text-muted)' }}>Anonymized samples</p>
              </div>
            </div>
          </div>

          {/* Label Distribution + Data Quality Row */}
          <div className={styles.chartsGrid} style={{ marginTop: '1.5rem' }}>
            {/* Label Distribution Chart */}
            <div className={`${styles.chartCard} glass`}>
              <div className={styles.chartHeader}>
                <h3>📊 Label Distribution</h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Training Pool</span>
              </div>
              <div className={styles.chartBody} style={{ height: '280px' }}>
                {stats?.label_counts && (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={Object.entries(stats.label_counts).map(([code, count]) => ({
                      code, count, name: LABEL_MAP[code] || code, fill: LABEL_COLORS[code] || '#6b7280'
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                      <XAxis dataKey="code" stroke="var(--text-muted)" fontSize={12} />
                      <YAxis stroke="var(--text-muted)" fontSize={12} />
                      <Tooltip
                        contentStyle={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border-color)', borderRadius: '8px' }}
                        formatter={(value: any, name: any, props: any) => [value, props.payload.name]}
                        labelFormatter={(label: any) => `Code ${label}`}
                      />
                      <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                        {Object.entries(stats.label_counts).map(([code]) => (
                          <Cell key={code} fill={LABEL_COLORS[code] || '#6b7280'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                )}
                {!stats?.label_counts && (
                  <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                    No label data available
                  </div>
                )}
              </div>
            </div>

            {/* Data Quality Card */}
            <div className={`${styles.chartCard} glass`}>
              <div className={styles.chartHeader}>
                <h3>🔍 Data Quality Indicators</h3>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', padding: '0.5rem 0' }}>
                {/* Completeness */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
                  <div>
                    <p style={{ fontWeight: 600, marginBottom: '0.25rem' }}>Complete Runs</p>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>All 3 modalities present</p>
                  </div>
                  <span style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--success)' }}>{stats?.complete_runs || 0}</span>
                </div>

                {/* Missing Modalities */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem' }}>
                  <div style={{ textAlign: 'center', padding: '1rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
                    <p style={{ fontSize: '1.25rem', fontWeight: 700, color: stats?.missing_csv > 0 ? 'var(--danger)' : 'var(--success)' }}>{stats?.missing_csv || 0}</p>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Missing CSV</p>
                  </div>
                  <div style={{ textAlign: 'center', padding: '1rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
                    <p style={{ fontSize: '1.25rem', fontWeight: 700, color: stats?.missing_flac > 0 ? 'var(--danger)' : 'var(--success)' }}>{stats?.missing_flac || 0}</p>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Missing FLAC</p>
                  </div>
                  <div style={{ textAlign: 'center', padding: '1rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
                    <p style={{ fontSize: '1.25rem', fontWeight: 700, color: stats?.missing_avi > 0 ? 'var(--danger)' : 'var(--success)' }}>{stats?.missing_avi || 0}</p>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Missing AVI</p>
                  </div>
                </div>

                {/* Duration Stats */}
                {stats?.audio_durations && stats.audio_durations.length > 0 && (
                  <div style={{ padding: '1rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
                    <p style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Duration Statistics</p>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.85rem' }}>
                      <div>
                        <span style={{ color: 'var(--text-muted)' }}>Audio Mean: </span>
                        <span style={{ fontWeight: 600 }}>{(stats.audio_durations.reduce((a: number, b: number) => a + b, 0) / stats.audio_durations.length).toFixed(1)}s</span>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-muted)' }}>Video Mean: </span>
                        <span style={{ fontWeight: 600 }}>{stats.video_durations?.length > 0 ? (stats.video_durations.reduce((a: number, b: number) => a + b, 0) / stats.video_durations.length).toFixed(1) : 'N/A'}s</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Imbalance Indicator */}
                {stats?.label_counts && (
                  <div style={{ padding: '1rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
                    <p style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Class Imbalance</p>
                    {(() => {
                      const counts = Object.values(stats.label_counts) as number[];
                      const max = Math.max(...counts);
                      const min = Math.min(...counts);
                      const ratio = min > 0 ? (max / min).toFixed(1) : '∞';
                      return (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <span style={{ fontSize: '1.25rem', fontWeight: 700, color: Number(ratio) > 5 ? '#f59e0b' : 'var(--success)' }}>{ratio}:1</span>
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>majority / minority ratio</span>
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Label Legend Table */}
          <div className={`${styles.chartCard} glass`} style={{ marginTop: '1.5rem' }}>
            <div className={styles.chartHeader}>
              <h3>📋 Defect Code Reference</h3>
            </div>
            <div className={styles.tableContainer}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>CODE</th>
                    <th>LABEL</th>
                    <th>COUNT</th>
                    <th>DISTRIBUTION</th>
                  </tr>
                </thead>
                <tbody>
                  {stats?.label_counts && Object.entries(stats.label_counts).map(([code, count]) => {
                    const total = Object.values(stats.label_counts as Record<string, number>).reduce((a: number, b: number) => a + b, 0);
                    const pct = total > 0 ? ((count as number) / total * 100).toFixed(1) : '0';
                    return (
                      <tr key={code}>
                        <td><span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', backgroundColor: LABEL_COLORS[code] || '#6b7280', marginRight: '0.5rem' }}></span>{code}</td>
                        <td>{LABEL_MAP[code] || 'Unknown'}</td>
                        <td>{count as number}</td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <div style={{ flex: 1, height: '6px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '3px', overflow: 'hidden' }}>
                              <div style={{ width: `${pct}%`, height: '100%', backgroundColor: LABEL_COLORS[code] || '#6b7280', borderRadius: '3px', transition: 'width 0.5s ease' }}></div>
                            </div>
                            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', minWidth: '40px' }}>{pct}%</span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ===== PAGE: RUN INSPECTOR ===== */}
      {activePage === 'inspector' && (
        <div className="animate-fade-in">
          <div className={styles.chartsGrid}>
            {/* Runs Table */}
            <div className={`${styles.tableCard} glass`} style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
              <div className={styles.chartHeader}>
                <h3>Test Sample Inspector</h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{runs.length} samples</span>
              </div>
              <div className={styles.tableContainer} style={{ flex: 1, maxHeight: '500px', overflowY: 'auto' }}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>SAMPLE ID</th>
                      <th>PREDICTION</th>
                      <th>P(DEFECT)</th>
                      <th>ACTION</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((r: any) => {
                      const sid = r.sample_id || r.run_id;
                      const predCode = String(r.pred_label_code ?? '00').padStart(2, '0');
                      const pDefect = r.p_defect ?? 0;
                      return (
                        <tr key={sid} onClick={() => handleSelectRun(sid)} style={{ cursor: 'pointer', backgroundColor: selectedRun === sid ? 'rgba(59, 130, 246, 0.1)' : 'transparent' }}>
                          <td style={{ fontFamily: 'monospace' }}>{sid}</td>
                          <td>
                            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', padding: '0.2rem 0.6rem', borderRadius: '9999px', fontSize: '0.75rem', fontWeight: 600, backgroundColor: predCode === '00' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)', color: predCode === '00' ? 'var(--success)' : 'var(--danger)' }}>
                              {predCode} — {LABEL_MAP[predCode] || 'Unknown'}
                            </span>
                          </td>
                          <td style={{ fontFamily: 'monospace', fontWeight: 600, color: pDefect > 0.5 ? 'var(--danger)' : 'var(--success)' }}>{(pDefect * 100).toFixed(1)}%</td>
                          <td>
                            <button className={styles.actionBtn} style={{ padding: '4px 12px', fontSize: '0.7rem' }}>Analyze</button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Inference View */}
            <div className={`${styles.chartCard} glass`}>
              <div className={styles.chartHeader}>
                <h3>Tri-Modal Analysis Engine</h3>
                {selectedRun && <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>{selectedRun}</span>}
              </div>

              <div className={styles.chartBody} style={{ height: 'auto', minHeight: '300px' }}>
                {!selectedRun && !inferLoading && (
                  <div style={{ display: 'flex', height: '300px', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', flexDirection: 'column', gap: '1rem' }}>
                    <Activity size={48} opacity={0.5} />
                    <p>Select a run from the logs to analyze</p>
                  </div>
                )}

                {inferLoading && (
                  <div style={{ display: 'flex', height: '300px', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-primary)', flexDirection: 'column', gap: '1rem' }}>
                    <Activity size={48} className={styles.spin} />
                    <p>Running ML Pipeline...</p>
                  </div>
                )}

                {inferData && inferData.detail && (
                  <div style={{ display: 'flex', height: '200px', alignItems: 'center', justifyContent: 'center', color: 'var(--danger)', flexDirection: 'column', gap: '1rem', padding: '2rem' }}>
                    <AlertTriangle size={48} />
                    <p>Backend Error: {inferData.detail}</p>
                  </div>
                )}

                {inferData && inferData.inference && (
                  (() => {
                    const code = String(inferData.inference.pred_label_code ?? '').padStart(2, '0');
                    const isPass = code === '00';
                    const pDefect = inferData.inference.p_defect ?? 0;
                    const typeConf = inferData.inference.type_confidence;
                    return (
                      <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                        {/* Verdict Banner */}
                        <div style={{ padding: '1.25rem', borderRadius: 'var(--radius-lg)', backgroundColor: isPass ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)', border: `1px solid ${isPass ? 'var(--success)' : 'var(--danger)'}` }}>
                          <h2 style={{ color: isPass ? 'var(--success)' : 'var(--danger)', marginBottom: '0.5rem', fontSize: '1.25rem' }}>
                            {isPass ? '✅ PASSED (Code 00)' : `⚠️ FAILED — ${LABEL_MAP[code] || 'Defect'} (${code})`}
                          </h2>
                          <div style={{ display: 'flex', gap: '2rem' }}>
                            {!isPass && (
                              <>
                                <div>
                                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>P(Defect)</span>
                                  <p style={{ fontWeight: 700, fontSize: '1.1rem' }}>{(pDefect * 100).toFixed(1)}%</p>
                                </div>
                                {typeConf != null && (
                                  <div>
                                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Confidence</span>
                                    <p style={{ fontWeight: 700, fontSize: '1.1rem' }}>{(typeConf * 100).toFixed(1)}%</p>
                                  </div>
                                )}
                              </>
                            )}
                          </div>
                        </div>

                        {/* SHAP Explanation — Why Flagged? */}
                        {!isPass && explanationData && explanationData.modality_contributions && (
                          <div style={{ padding: '1.25rem', borderRadius: 'var(--radius-lg)', backgroundColor: 'rgba(139, 92, 246, 0.06)', border: '1px solid rgba(139, 92, 246, 0.2)' }}>
                            <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', color: '#a78bfa' }}>🧠 Why Flagged?</h3>
                            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>SHAP analysis showing which modality contributed most to this prediction</p>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                              {(['sensor', 'audio', 'image'] as const).map(mod => {
                                const data = explanationData.modality_contributions[mod];
                                if (!data) return null;
                                const colors: Record<string, string> = { sensor: '#3b82f6', audio: '#f59e0b', image: '#10b981' };
                                const icons: Record<string, string> = { sensor: '📊', audio: '🔊', image: '📷' };
                                return (
                                  <div key={mod}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                                      <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>{icons[mod]} {mod.charAt(0).toUpperCase() + mod.slice(1)}</span>
                                      <span style={{ fontSize: '0.8rem', fontWeight: 700, color: colors[mod] }}>{data.percentage}%</span>
                                    </div>
                                    <div style={{ height: '8px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', overflow: 'hidden' }}>
                                      <div style={{ height: '100%', width: `${data.percentage}%`, backgroundColor: colors[mod], borderRadius: '4px', transition: 'width 0.5s ease' }} />
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                            {/* Top influential features */}
                            {explanationData.top_sensor_features && explanationData.top_sensor_features.length > 0 && (
                              <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid rgba(139, 92, 246, 0.15)' }}>
                                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Top Influential Sensor Features</p>
                                {explanationData.top_sensor_features.slice(0, 3).map((feat: any, i: number) => (
                                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', padding: '0.3rem 0' }}>
                                    <span style={{ color: 'var(--text-secondary)' }}>{feat.name}</span>
                                    <span style={{ fontWeight: 600, color: feat.direction === 'defect' ? 'var(--danger)' : 'var(--success)', fontSize: '0.7rem' }}>
                                      {feat.direction === 'defect' ? '↑ Defect' : '↓ Good'}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}

                        {/* Sensor Telemetry */}
                        {inferData.sensor_telemetry && Array.isArray(inferData.sensor_telemetry.values) && inferData.sensor_telemetry.values.length > 0 && (
                          <div style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1.25rem', borderRadius: 'var(--radius-lg)' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                              <Activity size={16} color="var(--accent-primary)" />
                              <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>{inferData.sensor_telemetry.metric_name} Trace</span>
                            </div>
                            <div style={{ height: '180px', width: '100%' }}>
                              <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={inferData.sensor_telemetry.values.map((v: number, i: number) => ({ index: i, value: v }))}>
                                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                                  <XAxis dataKey="index" hide />
                                  <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                                  <Tooltip
                                    contentStyle={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border-color)', borderRadius: '8px' }}
                                    labelStyle={{ display: 'none' }}
                                    formatter={(value: any) => [Number(value).toFixed(2), 'Amplitude']}
                                  />
                                  {inferData.sensor_telemetry.hotspot && (
                                    <ReferenceArea x1={inferData.sensor_telemetry.hotspot[0]} x2={inferData.sensor_telemetry.hotspot[1]} fill="var(--danger)" fillOpacity={0.15} />
                                  )}
                                  <Line type="monotone" dataKey="value" stroke="var(--accent-primary)" strokeWidth={2} dot={false} />
                                </LineChart>
                              </ResponsiveContainer>
                            </div>
                          </div>
                        )}

                        {/* Visual Frames — Full Width, Large */}
                        <div style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1.25rem', borderRadius: 'var(--radius-lg)' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                            <ImageIcon size={18} color="var(--accent-primary)" />
                            <span style={{ fontSize: '0.95rem', fontWeight: 600 }}>Visual Frames</span>
                            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>{inferData.images?.length || 0} keyframes</span>
                          </div>
                          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(inferData.images?.length || 1, 3)}, 1fr)`, gap: '1rem' }}>
                            {inferData.images?.map((img: string, i: number) => (
                              <div key={i} style={{ position: 'relative', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border-color)' }}>
                                <img
                                  src={`${API}${img}`}
                                  alt={`Keyframe ${i + 1}`}
                                  style={{ width: '100%', height: '220px', objectFit: 'cover', display: 'block' }}
                                />
                                <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '0.5rem 0.75rem', background: 'linear-gradient(transparent, rgba(0,0,0,0.8))', fontSize: '0.7rem', color: '#fff', fontWeight: 500 }}>
                                  Frame {i + 1}
                                </div>
                              </div>
                            ))}
                            {(!inferData.images || inferData.images.length === 0) && (
                              <div style={{ display: 'flex', height: '160px', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', gridColumn: '1 / -1' }}>
                                <ImageIcon size={32} style={{ marginRight: '0.75rem', opacity: 0.4 }} />
                                No frames available for this run
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Acoustic Analysis — Waveform + Player */}
                        <div style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1.25rem', borderRadius: 'var(--radius-lg)' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                            <Volume2 size={18} color="var(--accent-secondary)" />
                            <span style={{ fontSize: '0.95rem', fontWeight: 600 }}>Acoustic Signature Analysis</span>
                          </div>

                          {/* Audio Waveform Chart */}
                          {audioWaveform && audioWaveform.waveform && (
                            <div style={{ height: '160px', width: '100%', marginBottom: '1rem' }}>
                              <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={audioWaveform.waveform.map((v: number, i: number) => ({ index: i, amplitude: v }))}>
                                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                                  <XAxis dataKey="index" hide />
                                  <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                                  <Tooltip
                                    contentStyle={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border-color)', borderRadius: '8px', fontSize: '0.8rem' }}
                                    labelStyle={{ display: 'none' }}
                                    formatter={(value: any) => [Number(value).toFixed(4), 'Amplitude']}
                                  />
                                  {audioWaveform.error_region && (
                                    <ReferenceArea
                                      x1={audioWaveform.error_region[0]}
                                      x2={audioWaveform.error_region[1]}
                                      fill="var(--danger)"
                                      fillOpacity={0.2}
                                      stroke="var(--danger)"
                                      strokeOpacity={0.6}
                                      strokeDasharray="4 4"
                                      label={{ value: '⚠ Anomaly Region', position: 'top', fill: 'var(--danger)', fontSize: 11 }}
                                    />
                                  )}
                                  <defs>
                                    <linearGradient id="audioGrad" x1="0" y1="0" x2="0" y2="1">
                                      <stop offset="5%" stopColor="var(--accent-secondary)" stopOpacity={0.4} />
                                      <stop offset="95%" stopColor="var(--accent-secondary)" stopOpacity={0.05} />
                                    </linearGradient>
                                  </defs>
                                  <Area type="monotone" dataKey="amplitude" stroke="var(--accent-secondary)" strokeWidth={1.5} fill="url(#audioGrad)" dot={false} />
                                </AreaChart>
                              </ResponsiveContainer>
                            </div>
                          )}

                          {/* Fallback: use sensor-style trace if no dedicated waveform */}
                          {!audioWaveform && inferData.sensor_telemetry && (
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.75rem', padding: '0.5rem', backgroundColor: 'rgba(59, 130, 246, 0.05)', borderRadius: 'var(--radius-md)' }}>
                              ℹ Audio waveform rendering from acoustic data
                            </div>
                          )}

                          {/* Audio Player */}
                          {inferData.audio && (
                            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                              <audio controls src={`${API}${inferData.audio}`} style={{ flex: 1, height: '40px' }} />
                            </div>
                          )}
                          {!inferData.audio && (
                            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>No audio file available</p>
                          )}
                        </div>
                      </div>
                    )
                  })())
                }
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ===== PAGE: EVALUATION REPORT ===== */}
      {activePage === 'evaluation' && (
        <div className="animate-fade-in">
          {/* Top-level binary stats row */}
          <div className={styles.statsGrid} style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginBottom: '1.5rem' }}>
            {[
              { label: 'Binary F1', value: metrics?.binary?.f1, color: '#3b82f6' },
              { label: 'Precision', value: metrics?.binary?.precision, color: '#10b981' },
              { label: 'Recall', value: metrics?.binary?.recall, color: '#f59e0b' },
              { label: 'Accuracy', value: metrics?.binary?.accuracy, color: '#8b5cf6' },
            ].map((m, i) => (
              <div key={i} className={`${styles.statCard} glass`}>
                <div className={styles.statContent} style={{ textAlign: 'center' }}>
                  <p className={styles.statLabel}>{m.label}</p>
                  <h3 className={styles.statValue} style={{ color: m.color }}>{m.value != null ? m.value.toFixed(4) : 'N/A'}</h3>
                </div>
              </div>
            ))}
          </div>

          {/* ROC AUC + Binary Confusion Matrix side by side */}
          <div className={styles.chartsGrid} style={{ marginBottom: '1.5rem' }}>
            {/* ROC Curve Info */}
            <div className={`${styles.chartCard} glass`}>
              <div className={styles.chartHeader}>
                <h3>ROC Curve (AUC: {metrics?.binary?.roc_auc?.toFixed(4) || 'N/A'})</h3>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.5rem', padding: '1rem 0' }}>
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: '3.5rem', fontWeight: 800, background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)', WebkitBackgroundClip: 'text', color: 'transparent' }}>
                    {metrics?.binary?.roc_auc?.toFixed(4) || 'N/A'}
                  </p>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Area Under ROC Curve</p>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', width: '100%' }}>
                  <div style={{ padding: '1rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Threshold</p>
                    <p style={{ fontSize: '1.25rem', fontWeight: 700 }}>{metrics?.binary?.best_threshold?.toFixed(2) || 'N/A'}</p>
                  </div>
                  <div style={{ padding: '1rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Test Samples</p>
                    <p style={{ fontSize: '1.25rem', fontWeight: 700 }}>{metrics?.pipeline?.total_test_samples || 115}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Binary Confusion Matrix */}
            <div className={`${styles.chartCard} glass`}>
              <div className={styles.chartHeader}>
                <h3>Binary Confusion Matrix</h3>
              </div>
              {metrics?.binary?.confusion_matrix && (() => {
                const cm = metrics.binary.confusion_matrix;
                return (
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem', marginTop: '0.5rem' }}>
                    <thead>
                      <tr>
                        <th style={{ padding: '0.75rem', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 500 }}></th>
                        <th style={{ padding: '0.75rem', textAlign: 'center', color: 'var(--text-muted)', fontWeight: 500 }}>Pred: No Defect</th>
                        <th style={{ padding: '0.75rem', textAlign: 'center', color: 'var(--text-muted)', fontWeight: 500 }}>Pred: Defect</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td style={{ padding: '0.75rem', fontWeight: 600 }}>True: No Defect</td>
                        <td style={{ padding: '0.75rem', textAlign: 'center', backgroundColor: 'rgba(16,185,129,0.15)', fontWeight: 700, fontSize: '1.5rem', borderRadius: 'var(--radius-sm)' }}>{cm.tn}</td>
                        <td style={{ padding: '0.75rem', textAlign: 'center', backgroundColor: 'rgba(239,68,68,0.1)', fontWeight: 700, fontSize: '1.5rem', borderRadius: 'var(--radius-sm)' }}>{cm.fp}</td>
                      </tr>
                      <tr>
                        <td style={{ padding: '0.75rem', fontWeight: 600 }}>True: Defect</td>
                        <td style={{ padding: '0.75rem', textAlign: 'center', backgroundColor: 'rgba(239,68,68,0.1)', fontWeight: 700, fontSize: '1.5rem', borderRadius: 'var(--radius-sm)' }}>{cm.fn}</td>
                        <td style={{ padding: '0.75rem', textAlign: 'center', backgroundColor: 'rgba(16,185,129,0.15)', fontWeight: 700, fontSize: '1.5rem', borderRadius: 'var(--radius-sm)' }}>{cm.tp}</td>
                      </tr>
                    </tbody>
                  </table>
                );
              })()}
            </div>
          </div>

          {/* Per-Class F1 Score Bar Chart */}
          {metrics?.multiclass?.per_class && (
            <div className={`${styles.chartCard} glass`} style={{ marginBottom: '1.5rem' }}>
              <div className={styles.chartHeader}>
                <h3>Per-Class F1 Score</h3>
              </div>
              <div style={{ padding: '0.5rem 0' }}>
                {(() => {
                  const classColors: Record<string, string> = {
                    '11': '#3b82f6', '00': '#10b981', '01': '#f97316', '02': '#ef4444',
                    '06': '#8b5cf6', '07': '#ec4899', '08': '#06b6d4'
                  };
                  const classNames: Record<string, string> = {
                    '00': 'good weld', '01': 'excessive penetration', '02': 'burn through',
                    '06': 'overlap', '07': 'lack of fusion', '08': 'excessive convexity', '11': 'crater cracks'
                  };
                  const order = ['11', '00', '01', '02', '06', '07', '08'];
                  return order.map((cls) => {
                    const data = metrics.multiclass.per_class[cls];
                    if (!data) return null;
                    return (
                      <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.75rem' }}>
                        <span style={{ minWidth: '140px', textAlign: 'right', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{classNames[cls] || cls}</span>
                        <div style={{ flex: 1, backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-sm)', height: '28px', position: 'relative', overflow: 'hidden' }}>
                          <div style={{
                            width: `${(data.f1 * 100)}%`,
                            height: '100%',
                            backgroundColor: classColors[cls] || '#6b7280',
                            borderRadius: 'var(--radius-sm)',
                            transition: 'width 0.5s ease'
                          }} />
                        </div>
                        <span style={{ minWidth: '50px', fontWeight: 700, fontSize: '0.85rem' }}>{data.f1.toFixed(4)}</span>
                      </div>
                    );
                  });
                })()}
              </div>
            </div>
          )}

          {/* Multi-Class Confusion Matrix */}
          {metrics?.multiclass?.confusion_matrix && (
            <div className={`${styles.chartCard} glass`} style={{ marginBottom: '1.5rem' }}>
              <div className={styles.chartHeader}>
                <h3>Multi-Class Confusion Matrix</h3>
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', marginTop: '0.5rem' }}>
                  <thead>
                    <tr>
                      <th style={{ padding: '0.5rem', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 500, minWidth: '120px' }}></th>
                      {metrics.multiclass.confusion_matrix.labels.map((lbl: string, i: number) => (
                        <th key={i} style={{ padding: '0.5rem', textAlign: 'center', color: 'var(--text-muted)', fontWeight: 500, fontSize: '0.7rem', minWidth: '70px' }}>
                          {lbl}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.multiclass.confusion_matrix.matrix.map((row: number[], ri: number) => {
                      const rowMax = Math.max(...row);
                      return (
                        <tr key={ri}>
                          <td style={{ padding: '0.5rem', fontWeight: 600, fontSize: '0.75rem' }}>
                            {metrics.multiclass.confusion_matrix.labels[ri]}
                          </td>
                          {row.map((val: number, ci: number) => {
                            const isDiag = ri === ci;
                            const intensity = rowMax > 0 ? val / rowMax : 0;
                            return (
                              <td key={ci} style={{
                                padding: '0.5rem',
                                textAlign: 'center',
                                fontWeight: val > 0 ? 700 : 400,
                                fontSize: '0.9rem',
                                backgroundColor: isDiag
                                  ? `rgba(16,185,129,${0.1 + intensity * 0.3})`
                                  : val > 0
                                    ? `rgba(239,68,68,${0.05 + intensity * 0.2})`
                                    : 'transparent',
                                borderRadius: 'var(--radius-sm)',
                                color: val === 0 ? 'var(--text-muted)' : 'var(--text-primary)'
                              }}>
                                {val}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Per-Class Metrics Table */}
          {metrics?.multiclass?.per_class && (
            <div className={`${styles.chartCard} glass`} style={{ marginBottom: '1.5rem' }}>
              <div className={styles.chartHeader}>
                <h3>Per-Class Metrics</h3>
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid var(--border-color)' }}>
                      <th style={{ padding: '0.75rem', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 600 }}>Class</th>
                      <th style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-muted)', fontWeight: 600 }}>Precision</th>
                      <th style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-muted)', fontWeight: 600 }}>Recall</th>
                      <th style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-muted)', fontWeight: 600 }}>F1</th>
                      <th style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-muted)', fontWeight: 600 }}>Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(() => {
                      const classNames: Record<string, string> = {
                        '00': 'Good Weld', '01': 'Excessive Penetration', '02': 'Burn Through',
                        '06': 'Overlap', '07': 'Lack of Fusion', '08': 'Excessive Convexity', '11': 'Crater Cracks'
                      };
                      const order = ['11', '00', '01', '02', '06', '07', '08'];
                      return order.map((cls) => {
                        const d = metrics.multiclass.per_class[cls];
                        if (!d) return null;
                        return (
                          <tr key={cls} style={{ borderBottom: '1px solid var(--border-color)' }}>
                            <td style={{ padding: '0.75rem', fontWeight: 600 }}>{classNames[cls] || cls}</td>
                            <td style={{ padding: '0.75rem', textAlign: 'right' }}>{d.precision.toFixed(4)}</td>
                            <td style={{ padding: '0.75rem', textAlign: 'right' }}>{d.recall.toFixed(4)}</td>
                            <td style={{ padding: '0.75rem', textAlign: 'right', fontWeight: 700, color: d.f1 >= 0.8 ? 'var(--success)' : d.f1 >= 0.5 ? '#f59e0b' : 'var(--danger)' }}>{d.f1.toFixed(4)}</td>
                            <td style={{ padding: '0.75rem', textAlign: 'right' }}>{d.support}</td>
                          </tr>
                        );
                      });
                    })()}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Pipeline Score Summary */}
          <div className={`${styles.chartCard} glass`} style={{ marginBottom: '1.5rem' }}>
            <div className={styles.chartHeader}>
              <h3>🏆 Pipeline Score Summary</h3>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '2rem', padding: '1rem 0', flexWrap: 'wrap', justifyContent: 'center' }}>
              <div style={{ textAlign: 'center', flex: 1, minWidth: '200px' }}>
                <p style={{ fontSize: '3rem', fontWeight: 800, background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)', WebkitBackgroundClip: 'text', color: 'transparent' }}>{metrics?.pipeline?.final_score?.toFixed(4) || 'N/A'}</p>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.25rem' }}>Final Score = 0.6 × BinF1 + 0.4 × TypeMacroF1</p>
              </div>
              <div style={{ display: 'flex', gap: '1.5rem' }}>
                <div style={{ textAlign: 'center', padding: '1rem 1.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-lg)' }}>
                  <p style={{ fontSize: '1.5rem', fontWeight: 700, color: '#3b82f6' }}>{metrics?.pipeline?.binary_f1?.toFixed(4) || 'N/A'}</p>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Binary F1 (×0.6)</p>
                </div>
                <div style={{ textAlign: 'center', padding: '1rem 1.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-lg)' }}>
                  <p style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f59e0b' }}>{metrics?.pipeline?.type_macro_f1?.toFixed(4) || 'N/A'}</p>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Type Macro F1 (×0.4)</p>
                </div>
              </div>
            </div>
          </div>

          {/* Model Configuration Info */}
          <div className={styles.chartsGrid} style={{ marginBottom: '1.5rem' }}>
            <div className={`${styles.chartCard} glass`}>
              <div className={styles.chartHeader}>
                <h3>🎯 Pipeline Configuration</h3>
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.8, padding: '0.5rem 0' }}>
                <p>• <strong>Architecture:</strong> Chained XGBoost (Binary → 7-class Multi-class)</p>
                <p>• <strong>Features:</strong> Audio (180) + Image (128) — Physics-Based Fusion</p>
                <p>• <strong>Threshold:</strong> {metrics?.pipeline?.best_threshold?.toFixed(3) || 'N/A'} (Youden-J optimized)</p>
                <p>• <strong>Calibration:</strong> Isotonic Regression on OOF probabilities</p>
                <p>• <strong>Training:</strong> {metrics?.pipeline?.training || 'Parallel binary + multiclass'}</p>
                <p>• <strong>Selected Features:</strong> {metrics?.pipeline?.n_features || 93} (from 308 extracted)</p>
              </div>
            </div>
            <div className={`${styles.chartCard} glass`}>
              <div className={styles.chartHeader}>
                <h3>📊 Aggregate Multi-Class Metrics</h3>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
                {[
                  { label: 'Macro F1', value: metrics?.multiclass?.macro_f1, color: '#10b981' },
                  { label: 'Weighted F1', value: metrics?.multiclass?.weighted_f1, color: '#3b82f6' },
                  { label: 'Macro Precision', value: metrics?.multiclass?.macro_precision, color: '#f59e0b' },
                  { label: 'Macro Recall', value: metrics?.multiclass?.macro_recall, color: '#8b5cf6' },
                ].map((m, i) => (
                  <div key={i} style={{ padding: '1rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', borderLeft: `3px solid ${m.color}` }}>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>{m.label}</p>
                    <p style={{ fontSize: '1.25rem', fontWeight: 700 }}>{m.value != null ? m.value.toFixed(4) : 'N/A'}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Overfitting / Underfitting Report */}
          {diagnostics && (
            <div className={`${styles.chartCard} glass`} style={{ marginBottom: '1.5rem' }}>
              <div className={styles.chartHeader}>
                <h3>📈 Overfitting / Underfitting Report</h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Train vs Validation</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                {/* Binary Model */}
                {diagnostics.binary && (
                  <div style={{ padding: '1.25rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-lg)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                      <h4 style={{ margin: 0 }}>Binary Model</h4>
                      <span style={{ padding: '0.25rem 0.75rem', borderRadius: '9999px', fontSize: '0.7rem', fontWeight: 600, backgroundColor: diagnostics.binary.status === 'good' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)', color: diagnostics.binary.status === 'good' ? 'var(--success)' : '#f59e0b' }}>
                        {diagnostics.binary.verdict}
                      </span>
                    </div>
                    <table style={{ width: '100%', fontSize: '0.85rem', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                          <th style={{ textAlign: 'left', padding: '0.5rem 0', color: 'var(--text-muted)', fontWeight: 500 }}>Metric</th>
                          <th style={{ textAlign: 'right', padding: '0.5rem 0', color: 'var(--text-muted)', fontWeight: 500 }}>Train</th>
                          <th style={{ textAlign: 'right', padding: '0.5rem 0', color: 'var(--text-muted)', fontWeight: 500 }}>Val</th>
                          <th style={{ textAlign: 'right', padding: '0.5rem 0', color: 'var(--text-muted)', fontWeight: 500 }}>Gap</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[
                          { name: 'Log Loss', train: diagnostics.binary.train_log_loss, val: diagnostics.binary.val_log_loss },
                          { name: 'F1 Score', train: diagnostics.binary.train_f1, val: diagnostics.binary.val_f1 },
                          { name: 'Accuracy', train: diagnostics.binary.train_accuracy, val: diagnostics.binary.val_accuracy },
                        ].map((row, i) => {
                          const gap = row.train - row.val;
                          return (
                            <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                              <td style={{ padding: '0.5rem 0', fontWeight: 500 }}>{row.name}</td>
                              <td style={{ textAlign: 'right', padding: '0.5rem 0' }}>{row.train.toFixed(4)}</td>
                              <td style={{ textAlign: 'right', padding: '0.5rem 0' }}>{row.val.toFixed(4)}</td>
                              <td style={{ textAlign: 'right', padding: '0.5rem 0', color: Math.abs(gap) > 0.05 ? '#f59e0b' : 'var(--success)', fontWeight: 600 }}>
                                {gap > 0 ? '+' : ''}{gap.toFixed(4)}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* Multiclass Model */}
                {diagnostics.multiclass && (
                  <div style={{ padding: '1.25rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-lg)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                      <h4 style={{ margin: 0 }}>Multiclass Model</h4>
                      <span style={{ padding: '0.25rem 0.75rem', borderRadius: '9999px', fontSize: '0.7rem', fontWeight: 600, backgroundColor: diagnostics.multiclass.status === 'good' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)', color: diagnostics.multiclass.status === 'good' ? 'var(--success)' : '#f59e0b' }}>
                        {diagnostics.multiclass.verdict}
                      </span>
                    </div>
                    <table style={{ width: '100%', fontSize: '0.85rem', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                          <th style={{ textAlign: 'left', padding: '0.5rem 0', color: 'var(--text-muted)', fontWeight: 500 }}>Metric</th>
                          <th style={{ textAlign: 'right', padding: '0.5rem 0', color: 'var(--text-muted)', fontWeight: 500 }}>Train</th>
                          <th style={{ textAlign: 'right', padding: '0.5rem 0', color: 'var(--text-muted)', fontWeight: 500 }}>Val</th>
                          <th style={{ textAlign: 'right', padding: '0.5rem 0', color: 'var(--text-muted)', fontWeight: 500 }}>Gap</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[
                          { name: 'Log Loss', train: diagnostics.multiclass.train_log_loss, val: diagnostics.multiclass.val_log_loss },
                          { name: 'Macro F1', train: diagnostics.multiclass.train_f1, val: diagnostics.multiclass.val_f1 },
                          { name: 'Accuracy', train: diagnostics.multiclass.train_accuracy, val: diagnostics.multiclass.val_accuracy },
                        ].map((row, i) => {
                          const gap = row.train - row.val;
                          return (
                            <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                              <td style={{ padding: '0.5rem 0', fontWeight: 500 }}>{row.name}</td>
                              <td style={{ textAlign: 'right', padding: '0.5rem 0' }}>{row.train.toFixed(4)}</td>
                              <td style={{ textAlign: 'right', padding: '0.5rem 0' }}>{row.val.toFixed(4)}</td>
                              <td style={{ textAlign: 'right', padding: '0.5rem 0', color: Math.abs(gap) > 0.05 ? '#f59e0b' : 'var(--success)', fontWeight: 600 }}>
                                {gap > 0 ? '+' : ''}{gap.toFixed(4)}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ===== PAGE: EXPORT & DATA CARD ===== */}
      {activePage === 'export' && (
        <div className="animate-fade-in">
          {/* Download Buttons */}
          <div className={`${styles.chartCard} glass`}>
            <div className={styles.chartHeader}>
              <h3>📥 Exportable Reports</h3>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
              <a href={`${API}/stats`} target="_blank" rel="noopener noreferrer" className={styles.actionBtn} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', padding: '1rem', textDecoration: 'none', fontSize: '0.9rem' }}>
                <Download size={18} /> Dataset Stats (JSON)
              </a>
              <a href={`${API}/metrics`} target="_blank" rel="noopener noreferrer" className={styles.actionBtn} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', padding: '1rem', textDecoration: 'none', fontSize: '0.9rem' }}>
                <Download size={18} /> All Metrics (JSON)
              </a>
              <a href={`${API}/runs`} target="_blank" rel="noopener noreferrer" className={styles.actionBtn} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', padding: '1rem', textDecoration: 'none', fontSize: '0.9rem' }}>
                <Download size={18} /> Runs Data (JSON)
              </a>
            </div>
          </div>

          {/* Data Card */}
          <div className={`${styles.chartCard} glass`} style={{ marginTop: '1.5rem' }}>
            <div className={styles.chartHeader}>
              <h3>📋 Model Data Card</h3>
            </div>
            <div style={{ lineHeight: 1.8, fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
              <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem', marginTop: '1rem' }}>1. Sample Definition</h4>
              <p>A &quot;sample&quot; represents a single continuous welding run, composed of three modalities:</p>
              <ul style={{ paddingLeft: '1.5rem', margin: '0.5rem 0' }}>
                <li><strong>Sensor Telemetry</strong> (CSV): Time-series structural dynamics (Current, Voltage, Feed, Pressure)</li>
                <li><strong>Acoustic Signature</strong> (FLAC): High-frequency audio during welding</li>
                <li><strong>Visual Keyframes</strong> (JPG): Representative frames from AVI video</li>
              </ul>

              <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem', marginTop: '1.5rem' }}>2. Preprocessing</h4>
              <ul style={{ paddingLeft: '1.5rem', margin: '0.5rem 0' }}>
                <li><strong>No leakage:</strong> run_id, folder names, Part No are excluded from features</li>
                <li><strong>Sensor:</strong> Global aggregates (mean, std, percentiles, IQR, range, first-diff stats, tail-end stats)</li>
                <li><strong>Audio:</strong> 13 MFCCs + deltas, spectral centroid/rolloff, ZCR, RMS energy</li>
                <li><strong>Vision:</strong> OpenCV color histograms (32 bins × 3 channels) + Canny edge density</li>
              </ul>

              <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem', marginTop: '1.5rem' }}>3. Model Architecture</h4>
              <p><strong>Chained XGBoost Pipeline:</strong></p>
              <ul style={{ paddingLeft: '1.5rem', margin: '0.5rem 0' }}>
                <li>Stage 1 — Binary Gate: XGBoost predicts P(Defect). If below threshold → &quot;00&quot;</li>
                <li>Stage 2 — Multi-class: XGBoost ranks defect types. Falls back from &quot;00&quot; if binary says defect</li>
                <li>Threshold tuned end-to-end on <code>0.6 × Binary_F1 + 0.4 × Type_MacroF1</code></li>
              </ul>

              <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem', marginTop: '1.5rem' }}>4. Strengths &amp; Failure Cases</h4>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '0.5rem' }}>
                <div style={{ padding: '1rem', backgroundColor: 'rgba(16, 185, 129, 0.05)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                  <p style={{ fontWeight: 600, color: 'var(--success)', marginBottom: '0.5rem' }}>✅ Strengths</p>
                  <ul style={{ paddingLeft: '1rem', fontSize: '0.8rem', margin: 0 }}>
                    <li>Robust to missing modalities (zero-vector fallback)</li>
                    <li>Lightweight inference (ms-level, no GPU required)</li>
                    <li>End-to-end threshold optimization</li>
                  </ul>
                </div>
                <div style={{ padding: '1rem', backgroundColor: 'rgba(239, 68, 68, 0.05)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                  <p style={{ fontWeight: 600, color: 'var(--danger)', marginBottom: '0.5rem' }}>⚠️ Known Weaknesses</p>
                  <ul style={{ paddingLeft: '1rem', fontSize: '0.8rem', margin: 0 }}>
                    <li>Global pooling loses temporal alignment</li>
                    <li>Rare defect types may be under-represented</li>
                    <li>Crater vs burn-through confusion at end-of-run</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ===== PAGE: INSPECTION LOGS ===== */}
      {activePage === 'logs' && (
        <div className="animate-fade-in">
          {/* Search & Filter Bar */}
          <div className={`${styles.chartCard} glass`} style={{ marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
              <div style={{ flex: 1, minWidth: '250px', position: 'relative' }}>
                <Search size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                <input
                  type="text"
                  placeholder="Search by Run ID..."
                  value={logSearch}
                  onChange={(e) => setLogSearch(e.target.value)}
                  style={{ width: '100%', padding: '0.75rem 0.75rem 0.75rem 2.5rem', backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)', color: 'var(--text-primary)', fontSize: '0.9rem', outline: 'none' }}
                />
              </div>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                {['all', 'pass', 'fail'].map(f => (
                  <button
                    key={f}
                    onClick={() => setLogFilter(f)}
                    style={{ padding: '0.6rem 1.25rem', borderRadius: 'var(--radius-md)', fontSize: '0.8rem', fontWeight: 600, backgroundColor: logFilter === f ? (f === 'pass' ? 'rgba(16,185,129,0.15)' : f === 'fail' ? 'rgba(239,68,68,0.15)' : 'rgba(59,130,246,0.15)') : 'var(--bg-tertiary)', color: logFilter === f ? (f === 'pass' ? 'var(--success)' : f === 'fail' ? 'var(--danger)' : 'var(--accent-primary)') : 'var(--text-secondary)', border: '1px solid ' + (logFilter === f ? 'transparent' : 'var(--border-color)'), cursor: 'pointer', transition: 'all 150ms ease' }}
                  >
                    {f === 'all' ? `All (${runs.length})` : f === 'pass' ? `Pass (${runs.filter(r => String(r.label_code).padStart(2, '0') === '00').length})` : `Defect (${runs.filter(r => String(r.label_code).padStart(2, '0') !== '00').length})`}
                  </button>
                ))}
              </div>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{filteredRuns.length} results</span>
            </div>
          </div>

          {/* Full Logs Table */}
          <div className={`${styles.chartCard} glass`}>
            <div className={styles.chartHeader}>
              <h3>📋 Test Sample Logs</h3>
            </div>
            <div className={styles.tableContainer} style={{ maxHeight: '600px', overflowY: 'auto' }}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>SAMPLE ID</th>
                    <th>STATUS</th>
                    <th>PREDICTION</th>
                    <th>P(DEFECT)</th>
                    <th>AUDIO</th>
                    <th>IMAGES</th>
                    <th>ACTION</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRuns.map((r: any) => {
                    const sid = r.sample_id || r.run_id;
                    const code = String(r.pred_label_code ?? r.label_code ?? '').padStart(2, '0');
                    const isPass = code === '00';
                    const pDefect = r.p_defect ?? 0;
                    return (
                      <tr key={sid}>
                        <td style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>{sid}</td>
                        <td>
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', padding: '0.2rem 0.75rem', borderRadius: '9999px', fontSize: '0.7rem', fontWeight: 600, backgroundColor: isPass ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)', color: isPass ? 'var(--success)' : 'var(--danger)' }}>
                            {isPass ? '✓ PASS' : '✗ DEFECT'}
                          </span>
                        </td>
                        <td>
                          <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', backgroundColor: LABEL_COLORS[code] || '#6b7280', marginRight: '0.5rem' }}></span>
                          {LABEL_MAP[code] || `Code ${code}`}
                        </td>
                        <td style={{ fontFamily: 'monospace', fontWeight: 600, color: pDefect > 0.5 ? 'var(--danger)' : 'var(--success)' }}>{(pDefect * 100).toFixed(1)}%</td>
                        <td style={{ color: r.has_audio !== false ? 'var(--success)' : 'var(--text-muted)' }}>{r.has_audio !== false ? '✓' : '✗'}</td>
                        <td>{r.n_images ?? '—'}</td>
                        <td>
                          <button
                            className={styles.actionBtn}
                            style={{ padding: '4px 14px', fontSize: '0.7rem' }}
                            onClick={() => { setActivePage('inspector'); handleSelectRun(sid); }}
                          >
                            Analyze →
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ===== PAGE: DATA MODELS ===== */}
      {activePage === 'models' && (
        <div className="animate-fade-in">
          {/* Architecture Overview */}
          <div className={styles.statsGrid} style={{ gridTemplateColumns: 'repeat(3, 1fr)', marginBottom: '1.5rem' }}>
            <div className={`${styles.statCard} glass`} style={{ background: 'linear-gradient(135deg, rgba(139,92,246,0.1) 0%, rgba(59,130,246,0.1) 100%)' }}>
              <div className={styles.statIconWrapper} style={{ backgroundColor: 'rgba(139,92,246,0.2)', color: '#8b5cf6' }}>
                <GitBranch size={24} />
              </div>
              <div className={styles.statContent}>
                <p className={styles.statLabel}>Architecture</p>
                <h3 className={styles.statValue} style={{ fontSize: '1.1rem' }}>Chained XGBoost</h3>
                <p className={styles.statTrend} style={{ color: 'var(--text-muted)' }}>Binary → Multi-class</p>
              </div>
            </div>
            <div className={`${styles.statCard} glass`}>
              <div className={styles.statIconWrapper} style={{ backgroundColor: 'rgba(59,130,246,0.1)', color: 'var(--accent-primary)' }}>
                <Layers size={24} />
              </div>
              <div className={styles.statContent}>
                <p className={styles.statLabel}>Feature Fusion</p>
                <h3 className={styles.statValue} style={{ fontSize: '1.1rem' }}>Physics-Based</h3>
                <p className={styles.statTrend} style={{ color: 'var(--text-muted)' }}>Audio (180) + Vision (128)</p>
              </div>
            </div>
            <div className={`${styles.statCard} glass`}>
              <div className={styles.statIconWrapper} style={{ backgroundColor: 'rgba(16,185,129,0.1)', color: 'var(--success)' }}>
                <Target size={24} />
              </div>
              <div className={styles.statContent}>
                <p className={styles.statLabel}>Optimal Threshold</p>
                <h3 className={styles.statValue}>{metrics?.pipeline?.best_threshold?.toFixed(3) || 'N/A'}</h3>
                <p className={styles.statTrend} style={{ color: 'var(--success)' }}>Youden-J optimized</p>
              </div>
            </div>
          </div>

          <div className={styles.chartsGrid}>
            {/* Stage 1: Binary Gate */}
            <div className={`${styles.chartCard} glass`}>
              <div className={styles.chartHeader}>
                <h3>🔒 Stage 1 — Binary Gate</h3>
                <span style={{ fontSize: '0.75rem', padding: '0.2rem 0.6rem', borderRadius: '9999px', backgroundColor: 'rgba(59,130,246,0.15)', color: 'var(--accent-primary)', fontWeight: 600 }}>XGBClassifier</span>
              </div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>Predicts P(Defect) — if below threshold, classify as "Good Weld" (00).</p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
                {[
                  { label: 'Binary F1', value: metrics?.pipeline?.binary_f1 || metrics?.binary?.f1, color: '#10b981' },
                  { label: 'Precision', value: metrics?.binary?.precision, color: '#3b82f6' },
                  { label: 'Recall', value: metrics?.binary?.recall, color: '#f59e0b' },
                  { label: 'ROC-AUC', value: metrics?.binary?.roc_auc, color: '#8b5cf6' },
                  { label: 'PR-AUC', value: metrics?.binary?.pr_auc, color: '#ec4899' },
                  { label: 'ECE (Calibration)', value: metrics?.binary?.ece, color: '#6366f1' },
                ].map((m, i) => (
                  <div key={i} style={{ padding: '0.85rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', borderLeft: `3px solid ${m.color}` }}>
                    <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>{m.label}</p>
                    <p style={{ fontSize: '1.15rem', fontWeight: 700 }}>{m.value != null ? m.value.toFixed(4) : 'N/A'}</p>
                  </div>
                ))}
              </div>

              {/* Overfitting Check */}
              {diagnostics?.binary && (
                <div style={{ marginTop: '1.25rem', padding: '1rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                    <p style={{ fontWeight: 600, fontSize: '0.85rem' }}>Fit Diagnostics</p>
                    <span style={{ padding: '0.15rem 0.6rem', borderRadius: '9999px', fontSize: '0.65rem', fontWeight: 600, backgroundColor: diagnostics.binary.status === 'good' ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)', color: diagnostics.binary.status === 'good' ? 'var(--success)' : '#f59e0b' }}>
                      {diagnostics.binary.verdict}
                    </span>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem', fontSize: '0.8rem' }}>
                    <div><span style={{ color: 'var(--text-muted)' }}>Train F1: </span><strong>{diagnostics.binary.train_f1?.toFixed(4)}</strong></div>
                    <div><span style={{ color: 'var(--text-muted)' }}>Val F1: </span><strong>{diagnostics.binary.val_f1?.toFixed(4)}</strong></div>
                    <div><span style={{ color: 'var(--text-muted)' }}>Gap: </span><strong style={{ color: Math.abs(diagnostics.binary.train_f1 - diagnostics.binary.val_f1) > 0.05 ? '#f59e0b' : 'var(--success)' }}>{(diagnostics.binary.train_f1 - diagnostics.binary.val_f1).toFixed(4)}</strong></div>
                  </div>
                </div>
              )}
            </div>

            {/* Stage 2: Multi-class Classifier */}
            <div className={`${styles.chartCard} glass`}>
              <div className={styles.chartHeader}>
                <h3>🎯 Stage 2 — Multi-class Classifier</h3>
                <span style={{ fontSize: '0.75rem', padding: '0.2rem 0.6rem', borderRadius: '9999px', backgroundColor: 'rgba(245,158,11,0.15)', color: '#f59e0b', fontWeight: 600 }}>XGBClassifier</span>
              </div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>Ranks defect types for runs classified as defective by Stage 1.</p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
                {[
                  { label: 'Pipeline Type F1', value: metrics?.pipeline?.type_macro_f1, color: '#10b981' },
                  { label: 'Weighted F1', value: metrics?.multiclass?.weighted_f1, color: '#3b82f6' },
                  { label: 'Macro Precision', value: metrics?.multiclass?.macro_precision, color: '#f59e0b' },
                  { label: 'Macro Recall', value: metrics?.multiclass?.macro_recall, color: '#8b5cf6' },
                ].map((m, i) => (
                  <div key={i} style={{ padding: '0.85rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', borderLeft: `3px solid ${m.color}` }}>
                    <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>{m.label}</p>
                    <p style={{ fontSize: '1.15rem', fontWeight: 700 }}>{m.value != null ? m.value.toFixed(4) : 'N/A'}</p>
                  </div>
                ))}
              </div>

              {/* Overfitting Check */}
              {diagnostics?.multiclass && (
                <div style={{ marginTop: '1.25rem', padding: '1rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                    <p style={{ fontWeight: 600, fontSize: '0.85rem' }}>Fit Diagnostics</p>
                    <span style={{ padding: '0.15rem 0.6rem', borderRadius: '9999px', fontSize: '0.65rem', fontWeight: 600, backgroundColor: diagnostics.multiclass.status === 'good' ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)', color: diagnostics.multiclass.status === 'good' ? 'var(--success)' : '#f59e0b' }}>
                      {diagnostics.multiclass.verdict}
                    </span>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem', fontSize: '0.8rem' }}>
                    <div><span style={{ color: 'var(--text-muted)' }}>Train F1: </span><strong>{diagnostics.multiclass.train_f1?.toFixed(4)}</strong></div>
                    <div><span style={{ color: 'var(--text-muted)' }}>Val F1: </span><strong>{diagnostics.multiclass.val_f1?.toFixed(4)}</strong></div>
                    <div><span style={{ color: 'var(--text-muted)' }}>Gap: </span><strong style={{ color: Math.abs(diagnostics.multiclass.train_f1 - diagnostics.multiclass.val_f1) > 0.05 ? '#f59e0b' : 'var(--success)' }}>{(diagnostics.multiclass.train_f1 - diagnostics.multiclass.val_f1).toFixed(4)}</strong></div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Feature Engineering Summary */}
          <div className={`${styles.chartCard} glass`} style={{ marginTop: '1.5rem' }}>
            <div className={styles.chartHeader}>
              <h3>🧮 Feature Engineering Pipeline</h3>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
              <div style={{ padding: '1.25rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-lg)', borderTop: '3px solid #8b5cf6' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                  <Volume2 size={18} color="#8b5cf6" />
                  <span style={{ fontWeight: 600 }}>Audio Features (180)</span>
                </div>
                <ul style={{ paddingLeft: '1.25rem', fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                  <li>13 MFCCs + std deviations + delta/delta-delta</li>
                  <li>Sub-band energy ratios (config-invariant)</li>
                  <li>Spectral entropy, centroid, rolloff, bandwidth</li>
                  <li>ZCR, RMS energy, spectral contrast, flatness</li>
                  <li>Temporal pooling (mean, std, percentiles)</li>
                </ul>
              </div>
              <div style={{ padding: '1.25rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-lg)', borderTop: '3px solid #ec4899' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                  <ImageIcon size={18} color="#ec4899" />
                  <span style={{ fontWeight: 600 }}>Vision Features (128)</span>
                </div>
                <ul style={{ paddingLeft: '1.25rem', fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                  <li>Bead geometry (width, centre brightness)</li>
                  <li>Surface texture (GLCM, Laplacian variance)</li>
                  <li>Temporal consistency (frame differencing)</li>
                  <li>Keyframe extraction from AVI via OpenCV</li>
                </ul>
              </div>
              <div style={{ padding: '1.25rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-lg)', borderTop: '3px solid #3b82f6' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                  <Activity size={18} color="#3b82f6" />
                  <span style={{ fontWeight: 600 }}>Feature Selection</span>
                </div>
                <ul style={{ paddingLeft: '1.25rem', fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                  <li>308 raw features extracted per sample</li>
                  <li>93 selected via 2-pass importance ranking</li>
                  <li>60-iteration RandomizedSearchCV</li>
                  <li>GroupKFold(5) on config_folder</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Pipeline Score */}
          <div className={`${styles.chartCard} glass`} style={{ marginTop: '1.5rem' }}>
            <div className={styles.chartHeader}>
              <h3>🏆 End-to-End Pipeline Score</h3>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '2rem', padding: '1rem 0' }}>
              <div style={{ textAlign: 'center', flex: 1 }}>
                <p style={{ fontSize: '3rem', fontWeight: 800, background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)', WebkitBackgroundClip: 'text', color: 'transparent' }}>{metrics?.pipeline?.final_score?.toFixed(4) || 'N/A'}</p>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.25rem' }}>Final Score = 0.6 × BinF1 + 0.4 × TypeMacroF1</p>
              </div>
              <div style={{ display: 'flex', gap: '1.5rem' }}>
                <div style={{ textAlign: 'center', padding: '1rem 1.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-lg)' }}>
                  <p style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--success)' }}>{metrics?.pipeline?.binary_f1?.toFixed(4) || 'N/A'}</p>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Binary F1 (×0.6)</p>
                </div>
                <div style={{ textAlign: 'center', padding: '1rem 1.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-lg)' }}>
                  <p style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f59e0b' }}>{metrics?.pipeline?.type_macro_f1?.toFixed(4) || 'N/A'}</p>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Type Macro F1 (×0.4)</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
