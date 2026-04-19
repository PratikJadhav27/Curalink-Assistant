import React from 'react';
import { Toaster } from 'react-hot-toast';
import { motion } from 'framer-motion';
import ChatPage from './pages/ChatPage';
import Sidebar from './components/Sidebar';
import './index.css';

export default function App() {
  return (
    <div className="min-h-screen bg-surface font-sans">
      {/* Ambient background glow */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-brand-900/30 rounded-full blur-3xl" />
        <div className="absolute top-1/2 -right-40 w-96 h-96 bg-teal-900/20 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-1/3 w-80 h-80 bg-brand-900/20 rounded-full blur-3xl" />
      </div>

      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: '#1a1a2e',
            color: '#e5e7eb',
            border: '1px solid rgba(255,255,255,0.08)',
            fontSize: '13px',
          },
        }}
      />

      <Sidebar />
      <main className="relative z-10 h-screen">
        <ChatPage />
      </main>
    </div>
  );
}
