import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, Plus, Trash2, ChevronLeft, ChevronRight, Activity } from 'lucide-react';
import { useChatStore } from '../store/chatStore';
import { fetchHistory, fetchConversation, deleteConversation } from '../api/client';

export default function Sidebar() {
  const {
    sessionId,
    conversations,
    sidebarOpen,
    setSidebarOpen,
    startNewChat,
    setConversations,
    loadConversation,
    setSessionId,
  } = useChatStore();

  const [loadingConv, setLoadingConv] = useState(null);

  useEffect(() => {
    loadHistory();
  }, [sessionId]);

  const loadHistory = async () => {
    try {
      const data = await fetchHistory();
      setConversations(data);
    } catch (e) {
      // silent
    }
  };

  const handleLoad = async (sid) => {
    if (sid === sessionId) return;
    setLoadingConv(sid);
    try {
      const data = await fetchConversation(sid);
      const msgs = data.messages.map((m) => ({
        role: m.role,
        content: m.content,
        structuredData: m.structuredData,
      }));
      loadConversation(sid, msgs);
    } catch (e) {
      // silent
    } finally {
      setLoadingConv(null);
    }
  };

  const handleDelete = async (e, sid) => {
    e.stopPropagation();
    try {
      await deleteConversation(sid);
      setConversations(conversations.filter((c) => c.sessionId !== sid));
      if (sid === sessionId) startNewChat();
    } catch (e) {
      // silent
    }
  };

  return (
    <>
      {/* Toggle button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="fixed top-4 left-4 z-50 w-8 h-8 rounded-lg glass flex items-center justify-center text-gray-400 hover:text-white transition-colors"
        title={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
      >
        {sidebarOpen ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
      </button>

      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ x: -280, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -280, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="fixed left-0 top-0 h-full w-64 glass border-r border-white/5 z-40 flex flex-col"
          >
            {/* Logo */}
            <div className="px-5 pt-6 pb-4 border-b border-white/5">
              <div className="flex items-center gap-2.5 ml-8">
                <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-brand-500 to-teal-500 flex items-center justify-center">
                  <Activity size={14} className="text-white" />
                </div>
                <div>
                  <span className="font-bold text-white text-sm tracking-tight">Curalink</span>
                  <p className="text-xs text-gray-600">AI Research Assistant</p>
                </div>
              </div>
            </div>

            {/* New Chat */}
            <div className="px-3 py-3">
              <button
                onClick={startNewChat}
                className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl text-sm font-medium text-gray-300 hover:text-white hover:bg-white/5 transition-all border border-dashed border-white/10 hover:border-brand-500/30"
              >
                <Plus size={15} />
                New research chat
              </button>
            </div>

            {/* Conversations */}
            <div className="flex-1 overflow-y-auto px-3 pb-4 space-y-1">
              {conversations.length === 0 && (
                <p className="text-xs text-gray-600 text-center mt-8 px-4">
                  No conversations yet. Start asking questions!
                </p>
              )}
              {conversations.map((conv) => (
                <motion.button
                  key={conv.sessionId}
                  onClick={() => handleLoad(conv.sessionId)}
                  className={`w-full text-left px-3 py-2.5 rounded-xl text-xs transition-all group relative ${
                    conv.sessionId === sessionId
                      ? 'bg-brand-600/20 border border-brand-500/25 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                  whileHover={{ x: 2 }}
                >
                  <div className="flex items-start gap-2">
                    <MessageSquare size={12} className="mt-0.5 shrink-0 opacity-60" />
                    <div className="flex-1 min-w-0">
                      <p className="truncate font-medium leading-snug">
                        {conv.title || conv.patientContext?.disease || 'Research Session'}
                      </p>
                      {conv.patientContext?.disease && (
                        <p className="text-gray-600 mt-0.5 truncate">{conv.patientContext.disease}</p>
                      )}
                    </div>
                  </div>
                  {loadingConv === conv.sessionId && (
                    <div className="absolute inset-0 bg-black/30 rounded-xl flex items-center justify-center">
                      <div className="w-3 h-3 border-2 border-brand-400 border-t-transparent rounded-full animate-spin" />
                    </div>
                  )}
                  <button
                    onClick={(e) => handleDelete(e, conv.sessionId)}
                    className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 p-1 rounded-md hover:bg-red-500/20 hover:text-red-400 text-gray-600 transition-all"
                  >
                    <Trash2 size={11} />
                  </button>
                </motion.button>
              ))}
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </>
  );
}
