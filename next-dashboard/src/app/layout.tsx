import type { Metadata } from 'next';
import { Outfit } from 'next/font/google';
import './globals.css';
import Sidebar from '@/components/Sidebar';
import Header from '@/components/Header';

const outfit = Outfit({
  subsets: ['latin'],
  variable: '--font-outfit',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'WeldInspector Next',
  description: 'Tri-Modal Weld Inspector Dashboard',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={outfit.variable}>
      <body>
        <Sidebar />
        <main style={{ marginLeft: '280px', display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <Header />
          <div style={{ flex: 1, padding: '2rem', overflowY: 'auto' }}>
            {children}
          </div>
        </main>
      </body>
    </html>
  );
}
