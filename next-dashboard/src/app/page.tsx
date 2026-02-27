"use client";

import React, { useState, useEffect } from 'react';
import styles from './page.module.css';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, ReferenceArea, BarChart, Bar, Cell, PieChart, Pie, Legend
} from 'recharts';
import {
  Activity, AlertTriangle, CheckCircle, Clock, Volume2, Image as ImageIcon,
  Database, TrendingUp, Download, FileText, BarChart3, Shield, Zap, Eye
} from 'lucide-react';

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
  const [activePage, setActivePage] = useState('overview');
  const [runs, setRuns] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [diagnostics, setDiagnostics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  // Inspector state
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [inferData, setInferData] = useState<any>(null);
  const [inferLoading, setInferLoading] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/runs`).then(r => r.json()).catch(() => ({ runs: [] })),
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
    setSelectedRun(run_id);
    setInferLoading(true);
    setInferData(null);
    fetch(`${API}/infer/${run_id}`)
      .then(res => res.json())
      .then(data => { setInferData(data); setInferLoading(false); })
      .catch(() => setInferLoading(false));
  };

  const totalRuns = stats?.total_runs || runs.length;
  const anomalies = runs.filter(r => r.label_code !== '00').length;
  const passRate = totalRuns > 0 ? (((totalRuns - anomalies) / totalRuns) * 100).toFixed(1) : '0';

  // ——— NAVIGATION ———
  const navItems = [
    { key: 'overview', label: 'Dataset Overview', icon: <Database size={18} /> },
    { key: 'inspector', label: 'Run Inspector', icon: <Eye size={18} /> },
    { key: 'evaluation', label: 'Evaluation Report', icon: <BarChart3 size={18} /> },
    { key: 'export', label: 'Export & Data Card', icon: <FileText size={18} /> },
  ];

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
            onClick={() => setActivePage(item.key)}
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
                        labelFormatter={(label: string) => `Code ${label}`}
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
                <h3>Inspection Database Logs</h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{runs.length} runs</span>
              </div>
              <div className={styles.tableContainer} style={{ flex: 1, maxHeight: '500px', overflowY: 'auto' }}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>RUN ID</th>
                      <th>LABEL</th>
                      <th>ROWS</th>
                      <th>ACTION</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((r) => (
                      <tr key={r.run_id} onClick={() => handleSelectRun(r.run_id)} style={{ cursor: 'pointer', backgroundColor: selectedRun === r.run_id ? 'rgba(59, 130, 246, 0.1)' : 'transparent' }}>
                        <td style={{ fontFamily: 'monospace' }}>{r.run_id}</td>
                        <td>
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', padding: '0.2rem 0.6rem', borderRadius: '9999px', fontSize: '0.75rem', fontWeight: 600, backgroundColor: r.label_code === '00' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)', color: r.label_code === '00' ? 'var(--success)' : 'var(--danger)' }}>
                            {String(r.label_code).padStart(2, '0')}
                          </span>
                        </td>
                        <td>{r.csv_rows || '—'}</td>
                        <td>
                          <button className={styles.actionBtn} style={{ padding: '4px 12px', fontSize: '0.7rem' }}>Analyze</button>
                        </td>
                      </tr>
                    ))}
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
                  <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                    {/* Verdict Banner */}
                    <div style={{ padding: '1.25rem', borderRadius: 'var(--radius-lg)', backgroundColor: inferData.inference.pred_label_code === "00" ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)', border: `1px solid ${inferData.inference.pred_label_code === "00" ? 'var(--success)' : 'var(--danger)'}` }}>
                      <h2 style={{ color: inferData.inference.pred_label_code === "00" ? 'var(--success)' : 'var(--danger)', marginBottom: '0.5rem', fontSize: '1.25rem' }}>
                        {inferData.inference.pred_label_code === "00" ? '✅ PASSED (Code 00)' : `⚠️ FAILED — ${LABEL_MAP[inferData.inference.pred_label_code] || 'Defect'} (${inferData.inference.pred_label_code})`}
                      </h2>
                      <div style={{ display: 'flex', gap: '2rem' }}>
                        {inferData.inference.pred_label_code !== "00" && (
                          <>
                            <div>
                              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>P(Defect)</span>
                              <p style={{ fontWeight: 700, fontSize: '1.1rem' }}>{(inferData.inference.p_defect * 100).toFixed(1)}%</p>
                            </div>
                            {inferData.inference.type_confidence != null && (
                              <div>
                                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Confidence</span>
                                <p style={{ fontWeight: 700, fontSize: '1.1rem' }}>{(inferData.inference.type_confidence * 100).toFixed(1)}%</p>
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </div>

                    {/* Sensor Telemetry */}
                    {inferData.sensor_telemetry && inferData.sensor_telemetry.values && (
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

                    {/* Images + Audio Row */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                      <div style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: 'var(--radius-lg)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                          <ImageIcon size={16} color="var(--accent-primary)" />
                          <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Visual Frames</span>
                        </div>
                        <div style={{ display: 'flex', gap: '0.5rem', overflowX: 'auto' }}>
                          {inferData.images?.map((img: string, i: number) => (
                            <img key={i} src={`${API}${img}`} alt="frame" style={{ height: '70px', borderRadius: '4px', objectFit: 'cover' }} />
                          ))}
                          {(!inferData.images || inferData.images.length === 0) && (
                            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>No frames available</p>
                          )}
                        </div>
                      </div>
                      <div style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: 'var(--radius-lg)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                          <Volume2 size={16} color="var(--accent-secondary)" />
                          <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Acoustic Signature</span>
                        </div>
                        {inferData.audio && (
                          <audio controls src={`${API}${inferData.audio}`} style={{ width: '100%', height: '36px' }} />
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ===== PAGE: EVALUATION REPORT ===== */}
      {activePage === 'evaluation' && (
        <div className="animate-fade-in">
          {/* Final Score Hero */}
          <div className={styles.statsGrid} style={{ gridTemplateColumns: 'repeat(3, 1fr)', marginBottom: '1.5rem' }}>
            <div className={`${styles.statCard} glass`} style={{ background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)', borderColor: 'var(--accent-primary)' }}>
              <div className={styles.statIconWrapper} style={{ backgroundColor: 'rgba(59, 130, 246, 0.2)', color: 'var(--accent-primary)' }}>
                <TrendingUp size={24} />
              </div>
              <div className={styles.statContent}>
                <p className={styles.statLabel}>🏆 Final Score</p>
                <h3 className={styles.statValue}>{metrics?.pipeline?.final_score?.toFixed(4) || 'N/A'}</h3>
                <p className={styles.statTrend} style={{ color: 'var(--accent-primary)' }}>0.6×BinF1 + 0.4×TypeMacroF1</p>
              </div>
            </div>

            <div className={`${styles.statCard} glass`}>
              <div className={styles.statIconWrapper} style={{ backgroundColor: 'rgba(16, 185, 129, 0.1)', color: 'var(--success)' }}>
                <CheckCircle size={24} />
              </div>
              <div className={styles.statContent}>
                <p className={styles.statLabel}>Binary F1</p>
                <h3 className={styles.statValue}>{metrics?.pipeline?.binary_f1?.toFixed(4) || metrics?.binary?.f1?.toFixed(4) || 'N/A'}</h3>
                <p className={styles.statTrend} style={{ color: 'var(--success)' }}>Defect detection</p>
              </div>
            </div>

            <div className={`${styles.statCard} glass`}>
              <div className={styles.statIconWrapper} style={{ backgroundColor: 'rgba(245, 158, 11, 0.1)', color: '#f59e0b' }}>
                <BarChart3 size={24} />
              </div>
              <div className={styles.statContent}>
                <p className={styles.statLabel}>Type Macro-F1</p>
                <h3 className={styles.statValue}>{metrics?.pipeline?.type_macro_f1?.toFixed(4) || 'N/A'}</h3>
                <p className={styles.statTrend} style={{ color: '#f59e0b' }}>Defect classification</p>
              </div>
            </div>
          </div>

          {/* Detailed Metrics Grid */}
          <div className={styles.chartsGrid}>
            {/* Phase 2 Binary Metrics */}
            <div className={`${styles.chartCard} glass`}>
              <div className={styles.chartHeader}>
                <h3>Phase 2: Binary Classification</h3>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
                {[
                  { label: 'Precision', value: metrics?.binary?.precision, color: '#3b82f6' },
                  { label: 'Recall', value: metrics?.binary?.recall, color: '#10b981' },
                  { label: 'ROC-AUC', value: metrics?.binary?.roc_auc, color: '#8b5cf6' },
                  { label: 'PR-AUC', value: metrics?.binary?.pr_auc, color: '#f59e0b' },
                  { label: 'ECE (Calibration)', value: metrics?.binary?.ece, color: '#ec4899' },
                  { label: 'Threshold', value: metrics?.pipeline?.best_pipeline_threshold || metrics?.binary?.best_threshold, color: '#6366f1' },
                ].map((m, i) => (
                  <div key={i} style={{ padding: '1rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', borderLeft: `3px solid ${m.color}` }}>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>{m.label}</p>
                    <p style={{ fontSize: '1.25rem', fontWeight: 700 }}>{m.value != null ? m.value.toFixed(4) : 'N/A'}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Phase 3 Multiclass Metrics */}
            <div className={`${styles.chartCard} glass`}>
              <div className={styles.chartHeader}>
                <h3>Phase 3: Multi-Class Metrics</h3>
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

              {/* Model Info */}
              <div style={{ marginTop: '1.5rem', padding: '1rem', backgroundColor: 'rgba(59, 130, 246, 0.05)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                <p style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--accent-primary)' }}>🎯 Pipeline Configuration</p>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                  <p>• <strong>Architecture:</strong> Chained XGBoost (Binary → Multi-class)</p>
                  <p>• <strong>Features:</strong> Sensor + Audio + Image (Late Fusion)</p>
                  <p>• <strong>Threshold:</strong> {metrics?.pipeline?.best_pipeline_threshold?.toFixed(3) || 'N/A'} (optimized on FinalScore)</p>
                  <p>• <strong>Calibration:</strong> CalibratedClassifierCV (Sigmoid)</p>
                </div>
              </div>
            </div>
          </div>

          {/* Confidence Quality */}
          <div className={`${styles.chartCard} glass`} style={{ marginTop: '1.5rem' }}>
            <div className={styles.chartHeader}>
              <h3>📊 Confidence Quality Assessment</h3>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
              <div style={{ padding: '1.5rem', textAlign: 'center', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>ECE (Expected Calibration Error)</p>
                <p style={{ fontSize: '2rem', fontWeight: 700, color: (metrics?.binary?.ece || 0) < 0.1 ? 'var(--success)' : '#f59e0b' }}>{metrics?.binary?.ece?.toFixed(4) || 'N/A'}</p>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>Lower is better (≤ 0.05 = excellent)</p>
              </div>
              <div style={{ padding: '1.5rem', textAlign: 'center', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>Confidence Definition</p>
                <p style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>Calibrated Probability</p>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>CalibratedClassifierCV (sigmoid)</p>
              </div>
              <div style={{ padding: '1.5rem', textAlign: 'center', backgroundColor: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>Decision Rule</p>
                <p style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>Fixed Threshold</p>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>τ = {metrics?.pipeline?.best_pipeline_threshold?.toFixed(3) || '0.500'} (frozen at test-time)</p>
              </div>

              {/* Overfitting / Underfitting Report */}
              {diagnostics && (
                <div className={`${styles.chartCard} glass`} style={{ marginTop: '1.5rem' }}>
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
          </div>
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
    </div>
  );
}
