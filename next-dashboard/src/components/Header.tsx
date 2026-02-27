import React from 'react';
import styles from './Header.module.css';
import { Bell, Search, User } from 'lucide-react';

export default function Header() {
    return (
        <header className={styles.header}>
            <div className={styles.searchContainer}>
                <Search className={styles.searchIcon} size={18} />
                <input
                    type="text"
                    placeholder="Search inspections, batches..."
                    className={styles.searchInput}
                />
            </div>

            <div className={styles.actions}>
                <button className={styles.iconBtn}>
                    <Bell size={20} />
                    <span className={styles.badge}></span>
                </button>
                <div className={styles.divider}></div>
                <div className={styles.profileBtn}>
                    <div className={styles.avatar}>
                        <User size={18} />
                    </div>
                    <div className={styles.profileInfo}>
                        <span className={styles.name}>Admin</span>
                        <span className={styles.role}>System Ops</span>
                    </div>
                </div>
            </div>
        </header>
    );
}
