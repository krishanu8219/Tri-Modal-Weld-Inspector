"use client";

import React, { useState, useEffect } from 'react';
import styles from './page.module.css';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter
} from 'recharts';
import { Activity, AlertTriangle, CheckCircle, Clock, Volume2, Image as ImageIcon } from 'lucide-react';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const [runs, setRuns] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [inferData, setInferData] = useState<any>(null);
  const [inferLoading, setInferLoading] = useState(false);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/runs')
      .then(res => res.json())
      .then(data => {
        setRuns(data.runs || []);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  const handleSelectRun = (run_id: string) => {
    setSelectedRun(run_id);
    setInferLoading(true);
    setInferData(null);
    fetch(`http://127.0.0.1:8000/infer/${run_id}`)
      .then(res => res.json())
      .then(data => {
        setInferData(data);
        setInferLoading(false);
      })
      .catch(err => {
        console.error(err);
        setInferLoading(false);
      });
  };

  const chartData = runs.map((r, i) => ({
    time: `Run ${i + 1}`,
    accuracy: r.label_code === "00" ? 95 + Math.random() * 4 : 60 + Math.random() * 20,
    anomalies: r.label_code === "00" ? 0 : 1
  })).slice(0, 20);

  const totalRuns = runs.length;
  const anomalies = runs.filter(r => r.label_code !== '00').length;
  const accuracy = totalRuns > 0 ? (((totalRuns - anomalies) / totalRuns) * 100).toFixed(1) : 0;

  return (
    <div className={`${styles.dashboard} animate-fade-in`}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>System Overview</h1>
          <p className={styles.subtitle}>Real-time metrics from Tri-Modal Weld Inspector</p>
        </div>

        <div className={styles.tabs}>
          <button
            className={`${styles.tab} ${activeTab === 'overview' ? styles.activeTab : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button
            className={`${styles.tab} ${activeTab === 'analytics' ? styles.activeTab : ''}`}
            onClick={() => setActiveTab('analytics')}
          >
            Analytics
          </button>
        </div>
      </div>

      <div className={styles.statsGrid}>
        <div className={`${styles.statCard} glass delay-100`}>
          <div className={styles.statIconWrapper} style={{ backgroundColor: 'rgba(59, 130, 246, 0.1)', color: 'var(--accent-primary)' }}>
            <Activity size={24} />
          </div>
          <div className={styles.statContent}>
            <p className={styles.statLabel}>Total Valid Scans</p>
            <h3 className={styles.statValue}>{loading ? '...' : totalRuns}</h3>
            <p className={styles.statTrend} style={{ color: 'var(--success)' }}>Active Pipeline</p>
          </div>
        </div>

        <div className={`${styles.statCard} glass delay-200`}>
          <div className={styles.statIconWrapper} style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', color: 'var(--danger)' }}>
            <AlertTriangle size={24} />
          </div>
          <div className={styles.statContent}>
            <p className={styles.statLabel}>Ground Truth Anomalies</p>
            <h3 className={styles.statValue}>{loading ? '...' : anomalies}</h3>
            <p className={styles.statTrend} style={{ color: 'var(--text-muted)' }}>Historical Data</p>
          </div>
        </div>

        <div className={`${styles.statCard} glass delay-300`}>
          <div className={styles.statIconWrapper} style={{ backgroundColor: 'rgba(16, 185, 129, 0.1)', color: 'var(--success)' }}>
            <CheckCircle size={24} />
          </div>
          <div className={styles.statContent}>
            <p className={styles.statLabel}>Historical Pass Rate</p>
            <h3 className={styles.statValue}>{loading ? '...' : `${accuracy}%`}</h3>
            <p className={styles.statTrend} style={{ color: 'var(--success)' }}>Nominal performance</p>
          </div>
        </div>
      </div>

      {/* Main Analysis Section */}
      <div className={styles.chartsGrid}>
        <div className={`${styles.tableCard} glass`} style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <div className={styles.chartHeader}>
            <h3>Inspection Database Logs</h3>
            <button className={styles.actionBtn}>Refresh</button>
          </div>
          <div className={styles.tableContainer} style={{ flex: 1, maxHeight: '400px', overflowY: 'auto' }}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>RUN ID</th>
                  <th>FILES</th>
                  <th>MODALITY</th>
                  <th>ACTION</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.run_id} onClick={() => handleSelectRun(r.run_id)} style={{ cursor: 'pointer', backgroundColor: selectedRun === r.run_id ? 'rgba(59, 130, 246, 0.1)' : 'transparent' }}>
                    <td>{r.run_id}</td>
                    <td>{r.csv_rows} rows</td>
                    <td>Tri-Modal</td>
                    <td>
                      <button className={styles.actionBtn} style={{ padding: '4px 8px', fontSize: '0.7rem' }}>Analyze</button>
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
            {selectedRun && <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{selectedRun}</span>}
          </div>

          <div className={styles.chartBody} style={{ height: 'auto', minHeight: '300px' }}>
            {!selectedRun && !inferLoading && (
              <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', flexDirection: 'column', gap: '1rem' }}>
                <Activity size={48} opacity={0.5} />
                <p>Select a run from the logs to analyze</p>
              </div>
            )}

            {inferLoading && (
              <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-primary)', flexDirection: 'column', gap: '1rem' }}>
                <Activity size={48} className={styles.spin} />
                <p>Running ML Pipeline...</p>
              </div>
            )}

            {inferData && (
              <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div style={{ padding: '1.5rem', borderRadius: 'var(--radius-lg)', backgroundColor: inferData.inference.pred_label_code === "00" ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)', border: `1px solid ${inferData.inference.pred_label_code === "00" ? 'var(--success)' : 'var(--danger)'}` }}>
                  <h2 style={{ color: inferData.inference.pred_label_code === "00" ? 'var(--success)' : 'var(--danger)', marginBottom: '0.5rem' }}>
                    {inferData.inference.pred_label_code === "00" ? '✅ PASSED (Code 00)' : `⚠️ FAILED (Defect ${inferData.inference.pred_label_code})`}
                  </h2>
                  <div style={{ display: 'flex', gap: '2rem' }}>
                    <div>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Defect Probability</span>
                      <p style={{ fontWeight: 600, fontSize: '1.2rem' }}>{(inferData.inference.p_defect * 100).toFixed(2)}%</p>
                    </div>
                    <div>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Classification Confidence</span>
                      <p style={{ fontWeight: 600, fontSize: '1.2rem' }}>{(inferData.inference.type_confidence * 100).toFixed(2)}%</p>
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '1rem' }}>
                  <div style={{ flex: 1, backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: 'var(--radius-lg)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                      <ImageIcon size={18} color="var(--accent-primary)" />
                      <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Visual Frames</span>
                    </div>
                    <div style={{ display: 'flex', gap: '0.5rem', overflowX: 'auto', paddingBottom: '0.5rem' }}>
                      {inferData.images.map((img: string, i: number) => (
                        <img key={i} src={`http://127.0.0.1:8000${img}`} alt="frame" style={{ height: '80px', borderRadius: '4px', objectFit: 'cover' }} />
                      ))}
                    </div>
                  </div>

                  <div style={{ flex: 1, backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: 'var(--radius-lg)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                      <Volume2 size={18} color="var(--accent-secondary)" />
                      <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Acoustic Signature</span>
                    </div>
                    <audio controls src={`http://127.0.0.1:8000${inferData.audio}`} style={{ width: '100%', height: '36px' }} />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
