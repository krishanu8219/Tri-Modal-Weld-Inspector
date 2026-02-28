import type { Metadata } from 'next';
import { Outfit } from 'next/font/google';
import './globals.css';
import LayoutShell from '@/components/LayoutShell';

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
    <html lang="en" className={outfit.variable} suppressHydrationWarning>
      <body suppressHydrationWarning>
        <LayoutShell>{children}</LayoutShell>
      </body>
    </html>
  );
}
