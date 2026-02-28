"use client";

import React from 'react';
import styles from './Sidebar.module.css';
import { LayoutDashboard, Database, LogOut, FileImage, ShieldAlert, PanelLeftClose, PanelLeft, Cpu } from 'lucide-react';
import { useNavigation, PageKey } from './NavigationContext';

interface SidebarProps {
    collapsed: boolean;
    onToggle: () => void;
}

const NAV_ITEMS: { key: PageKey; label: string; icon: React.ReactNode; group: 'main' | 'system' }[] = [
    { key: 'overview', label: 'Dashboard', icon: <LayoutDashboard size={20} />, group: 'main' },
    { key: 'logs', label: 'Inspection Logs', icon: <FileImage size={20} />, group: 'main' },
    { key: 'models', label: 'Data Models', icon: <Cpu size={20} />, group: 'system' },
];

export default function Sidebar({ collapsed, onToggle }: SidebarProps) {
    const { activePage, setActivePage } = useNavigation();

    const mainItems = NAV_ITEMS.filter(i => i.group === 'main');
    const systemItems = NAV_ITEMS.filter(i => i.group === 'system');

    return (
        <aside className={`${styles.sidebar} ${collapsed ? styles.collapsed : ''}`}>
            <div className={styles.logo}>
                <ShieldAlert className={styles.logoIcon} size={24} />
                {!collapsed && <h2>WeldInspector</h2>}
            </div>

            {/* Toggle button */}
            <button className={styles.toggleBtn} onClick={onToggle} title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
                {collapsed ? <PanelLeft size={20} /> : <PanelLeftClose size={20} />}
            </button>

            <nav className={styles.nav}>
                <div className={styles.navGroup}>
                    {!collapsed && <p className={styles.navGroupTitle}>MAIN</p>}
                    {mainItems.map(item => (
                        <a
                            key={item.key}
                            href="#"
                            className={`${styles.navItem} ${activePage === item.key ? styles.active : ''}`}
                            onClick={(e) => { e.preventDefault(); setActivePage(item.key); }}
                            title={item.label}
                        >
                            {item.icon}
                            {!collapsed && <span>{item.label}</span>}
                        </a>
                    ))}
                </div>

                <div className={styles.navGroup}>
                    {!collapsed && <p className={styles.navGroupTitle}>SYSTEM</p>}
                    {systemItems.map(item => (
                        <a
                            key={item.key}
                            href="#"
                            className={`${styles.navItem} ${activePage === item.key ? styles.active : ''}`}
                            onClick={(e) => { e.preventDefault(); setActivePage(item.key); }}
                            title={item.label}
                        >
                            {item.icon}
                            {!collapsed && <span>{item.label}</span>}
                        </a>
                    ))}
                </div>
            </nav>

            <div className={styles.footer}>
                <button className={styles.logoutBtn} title="Logout">
                    <LogOut size={20} />
                    {!collapsed && <span>Logout</span>}
                </button>
            </div>
        </aside>
    );
}
