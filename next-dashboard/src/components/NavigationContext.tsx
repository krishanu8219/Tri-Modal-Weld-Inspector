"use client";

import React, { createContext, useContext, useState, ReactNode } from 'react';

export type PageKey = 'overview' | 'inspector' | 'evaluation' | 'export' | 'logs' | 'models';

interface NavigationContextType {
    activePage: PageKey;
    setActivePage: (page: PageKey) => void;
}

const NavigationContext = createContext<NavigationContextType>({
    activePage: 'overview',
    setActivePage: () => { },
});

export function NavigationProvider({ children }: { children: ReactNode }) {
    const [activePage, setActivePage] = useState<PageKey>('overview');
    return (
        <NavigationContext.Provider value={{ activePage, setActivePage }}>
            {children}
        </NavigationContext.Provider>
    );
}

export function useNavigation() {
    return useContext(NavigationContext);
}
