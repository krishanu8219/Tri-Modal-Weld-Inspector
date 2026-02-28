"use client";

import React, { useState } from 'react';
import Sidebar from './Sidebar';
import Header from './Header';
import { NavigationProvider } from './NavigationContext';

export default function LayoutShell({ children }: { children: React.ReactNode }) {
    const [collapsed, setCollapsed] = useState(false);

    return (
        <NavigationProvider>
            <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(c => !c)} />
            <main
                style={{
                    marginLeft: collapsed ? '72px' : '280px',
                    display: 'flex',
                    flexDirection: 'column',
                    minHeight: '100vh',
                    transition: 'margin-left 300ms cubic-bezier(0.4, 0, 0.2, 1)',
                }}
            >
                <Header />
                <div style={{ flex: 1, padding: '2rem', overflowY: 'auto' }}>
                    {children}
                </div>
            </main>
        </NavigationProvider>
    );
}
