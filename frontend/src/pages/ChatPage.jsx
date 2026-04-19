import React, { useRef, useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Sparkles, BookOpen, FlaskConical, Activity, Zap } from 'lucide-react';
import toast from 'react-hot-toast';
import { useChatStore } from '../store/chatStore';
import { sendMessage } from '../api/client';
import ChatMessage from '../components/ChatMessage';
import PatientContextPanel from '../components/PatientContextPanel';

const EXAMPLE_QUERIES = [
  { icon: <Activity size={14} />, text: 'Latest treatment for lung cancer' },
  { icon: <FlaskConical size={14} />, text: 'Clinical trials for diabetes' },
  { icon: <BookOpen size={14} />, text: 'Top researchers in Alzheimer\'s disease' },
  { icon: <Zap size={14} />, text: 'Recent studies on heart disease and statins' },
];

function EmptyState({ onExample }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center h-full py-20 text-center px-8"
    >
      {/* Hero icon */}
      <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-brand-600 to-teal-500 flex items-center justify-center mb-6 shadow-2xl shadow-brand-900/50 glow-brand">
        <Sparkles size={36} className="text-white" />
      </div>

      <h1 className="text-3xl font-bold text-white mb-2 tracking-tight">
        Ask Curalink
      </h1>
      <p className="text-gray-400 text-base mb-8 max-w-md leading-relaxed">
        Your AI-powered medical research companion. Get structured, source-backed insights from <span className="text-brand-400 font-medium">PubMed</span>, <span className="text-teal-400 font-medium">OpenAlex</span>, and <span className="text-yellow-400 font-medium">ClinicalTrials.gov</span>.
      </p>

      {/* Features */}
      <div className="grid grid-cols-3 gap-3 mb-10 w-full max-w-lg">
        {[
          { icon: <BookOpen size={16} />, label: '100–200+', sub: 'Papers retrieved' },
          { icon: <FlaskConical size={16} />, label: 'Live trials', sub: 'ClinicalTrials.gov' },
          { icon: <Sparkles size={16} />, label: 'AI reasoning', sub: 'Llama-3 powered' },
        ].map((f, i) => (
          <div key={i} className="glass rounded-xl p-3 text-center">
            <div className="text-brand-400 flex justify-center mb-1">{f.icon}</div>
            <p className="text-white font-semibold text-sm">{f.label}</p>
            <p className="text-gray-600 text-xs">{f.sub}</p>
          </div>
        ))}
      </div>

      {/* Example queries */}
      <p className="text-xs text-gray-600 mb-3 font-medium uppercase tracking-wider">Try an example</p>
      <div className="grid grid-cols-2 gap-2 w-full max-w-lg">
        {EXAMPLE_QUERIES.map((q, i) => (
          <motion.button
            key={i}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => onExample(q.text)}
            className="glass-hover rounded-xl px-4 py-3 text-left text-sm text-gray-300 flex items-center gap-2 hover:text-white transition-colors"
          >
            <span className="text-brand-400 shrink-0">{q.icon}</span>
            {q.text}
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}

export default function ChatPage() {
  const {
    messages,
    isLoading,
    sessionId,
    patientContext,
    error,
    sidebarOpen,
    addMessage,
    setLoading,
    setError,
    clearError,
    setSessionId,
  } = useChatStore();

  const [input, setInput] = useState('');
  const [expandedQuery, setExpandedQuery] = useState('');
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = useCallback(async (text) => {
    const msg = (text || input).trim();
    if (!msg || isLoading) return;

    setInput('');
    setExpandedQuery('');
    clearError();

    addMessage({ role: 'user', content: msg });
    setLoading(true);

    try {
      const res = await sendMessage({
        message: msg,
        sessionId,
        patientContext,
      });

      if (!sessionId) setSessionId(res.sessionId);
      if (res.queryExpanded) setExpandedQuery(res.queryExpanded);

      addMessage({
        role: 'assistant',
        content: res.answer,
        structuredData: {
          conditionOverview: res.conditionOverview,
          researchInsights: res.publications || [],
          clinicalTrials: res.clinicalTrials || [],
        },
        stats: {
          totalPublicationsRetrieved: res.publications?.length || 0,  // we'll use what's returned
          topPublicationsShown: res.publications?.length || 0,
          totalTrialsRetrieved: res.clinicalTrials?.length || 0,
          topTrialsShown: res.clinicalTrials?.length || 0,
        },
      });
    } catch (err) {
      const errMsg = err.response?.data?.error || err.message || 'Something went wrong.';
      setError(errMsg);
      toast.error(errMsg);
      addMessage({
        role: 'assistant',
        content: `I encountered an error: **${errMsg}**\n\nPlease check that all services are running and try again.`,
      });
    } finally {
      setLoading(false);
    }
  }, [input, isLoading, sessionId, patientContext]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const marginLeft = sidebarOpen ? 'ml-64' : 'ml-0';

  return (
    <div className={`flex flex-col h-screen transition-all duration-300 ${marginLeft}`}>
      {/* Chat area */}
      <div className="flex-1 overflow-y-auto px-4 pt-6">
        <div className="max-w-4xl mx-auto">
          {messages.length === 0 ? (
            <EmptyState onExample={(q) => handleSend(q)} />
          ) : (
            <div className="py-6 space-y-6">
              {/* Expanded query badge */}
              <AnimatePresence>
                {expandedQuery && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col gap-1 text-xs text-gray-500 mb-4 w-full max-w-3xl"
                  >
                    <details className="cursor-pointer group">
                      <summary className="flex items-center gap-2 outline-none select-none text-gray-500 hover:text-gray-300 transition-colors list-none">
                        <Sparkles size={11} className="text-brand-400 transition-transform group-open:text-brand-300" />
                        <span>View Expanded AI Search Query</span>
                      </summary>
                      <div className="mt-2 p-3 bg-white/5 rounded-lg border border-white/10 text-brand-300/80 font-mono text-[11px] leading-relaxed break-words">
                        {expandedQuery}
                      </div>
                    </details>
                  </motion.div>
                )}
              </AnimatePresence>

              {messages.map((msg, i) => (
                <ChatMessage key={i} message={msg} />
              ))}

              {isLoading && <ChatMessage isTyping />}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Input area */}
      <div className="border-t border-white/5 bg-surface/80 backdrop-blur-xl px-4 py-4">
        <div className="max-w-4xl mx-auto space-y-3">
          {messages.length > 0 && <PatientContextPanel />}

          <div className="flex gap-3 items-end">
            <div className="flex-1 glass rounded-2xl border border-white/8 focus-within:border-brand-500/40 transition-colors">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about a disease, treatment, clinical trial…"
                rows={1}
                className="w-full bg-transparent px-4 pt-3.5 pb-3 text-sm text-white placeholder-gray-600 focus:outline-none resize-none max-h-36 leading-relaxed"
                style={{ height: 'auto' }}
                onInput={(e) => {
                  e.target.style.height = 'auto';
                  e.target.style.height = e.target.scrollHeight + 'px';
                }}
                disabled={isLoading}
              />
            </div>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleSend()}
              disabled={isLoading || !input.trim()}
              className={`w-12 h-12 rounded-2xl flex items-center justify-center shadow-lg transition-all duration-200
                ${input.trim() && !isLoading
                  ? 'bg-gradient-to-br from-brand-600 to-brand-500 shadow-brand-900/50 text-white'
                  : 'bg-white/5 text-gray-600 cursor-not-allowed'
                }`}
            >
              {isLoading ? (
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <Send size={18} />
              )}
            </motion.button>
          </div>

          {messages.length === 0 && (
            <div className="px-1">
              <PatientContextPanel />
            </div>
          )}

          <p className="text-center text-xs text-gray-700">
            Powered by PubMed · OpenAlex · ClinicalTrials.gov · Llama-3 via Groq
          </p>
        </div>
      </div>
    </div>
  );
}
