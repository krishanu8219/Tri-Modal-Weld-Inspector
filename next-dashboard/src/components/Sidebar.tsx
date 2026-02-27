import React from 'react';
import styles from './Sidebar.module.css';
import { LayoutDashboard, Activity, Database, Settings, LogOut, FileImage, ShieldAlert } from 'lucide-react';

export default function Sidebar() {
    return (
        <aside className={styles.sidebar}>
            <div className={styles.logo}>
                <ShieldAlert className={styles.logoIcon} />
                <h2>WeldInspector</h2>
            </div>

            <nav className={styles.nav}>
                <div className={styles.navGroup}>
                    <p className={styles.navGroupTitle}>MAIN</p>
                    <a href="#" className={`${styles.navItem} ${styles.active}`}>
                        <LayoutDashboard size={20} />
                        <span>Dashboard</span>
                    </a>
                    <a href="#" className={styles.navItem}>
                        <Activity size={20} />
                        <span>Real-time Scans</span>
                    </a>
                    <a href="#" className={styles.navItem}>
                        <FileImage size={20} />
                        <span>Inspection Logs</span>
                    </a>
                </div>

                <div className={styles.navGroup}>
                    <p className={styles.navGroupTitle}>SYSTEM</p>
                    <a href="#" className={styles.navItem}>
                        <Database size={20} />
                        <span>Data Models</span>
                    </a>
                    <a href="#" className={styles.navItem}>
                        <Settings size={20} />
                        <span>Settings</span>
                    </a>
                </div>
            </nav>

            <div className={styles.footer}>
                <button className={styles.logoutBtn}>
                    <LogOut size={20} />
                    <span>Logout</span>
                </button>
            </div>
        </aside>
    );
}
