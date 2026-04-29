import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { Bot, User, ChevronDown, ChevronUp, BookOpen, FlaskConical, TrendingUp, Zap } from 'lucide-react';
import PublicationCard from './PublicationCard';
import ClinicalTrialCard from './ClinicalTrialCard';

function ThinkingDots() {
  return (
    <div className="flex items-center gap-1.5 py-1">
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="w-2 h-2 rounded-full bg-brand-400"
          animate={{ opacity: [0.3, 1, 0.3], scale: [0.8, 1.1, 0.8] }}
          transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
        />
      ))}
    </div>
  );
}

function SynthesisPanel({ content }) {
  const [open, setOpen] = useState(true);
  if (!content) return null;
  return (
    <div className="mb-3 border border-brand-500/25 rounded-xl overflow-hidden bg-brand-950/20">
      <button
        onClick={() => setOpen((p) => !p)}
        className="w-full flex items-center justify-between px-4 py-3 bg-brand-900/40 hover:bg-brand-900/60 transition-colors"
      >
        <span className="flex items-center gap-2 text-sm font-semibold text-brand-300">
          <TrendingUp size={14} />
          AI Research Synthesis
        </span>
        {open ? <ChevronUp size={14} className="text-gray-500" /> : <ChevronDown size={14} className="text-gray-500" />}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-5 py-4 prose-ai text-sm leading-relaxed">
              <ReactMarkdown>{content}</ReactMarkdown>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function StructuredSection({ title, icon, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="mt-4 border border-white/5 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen((p) => !p)}
        className="w-full flex items-center justify-between px-4 py-3 bg-white/3 hover:bg-white/5 transition-colors"
      >
        <span className="flex items-center gap-2 text-sm font-semibold text-white">
          {icon}
          {title}
        </span>
        {open ? <ChevronUp size={16} className="text-gray-500" /> : <ChevronDown size={16} className="text-gray-500" />}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="p-4 grid gap-3">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function ChatMessage({ message, isTyping }) {
  const isUser = message?.role === 'user';

  if (isTyping) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex gap-3 items-start max-w-3xl"
      >
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-brand-600 to-teal-500 flex items-center justify-center shrink-0 shadow-lg shadow-brand-900/50">
          <Bot size={16} className="text-white" />
        </div>
        <div className="glass rounded-2xl rounded-tl-sm px-5 py-4">
          <ThinkingDots />
        </div>
      </motion.div>
    );
  }

  if (isUser) {
    return (
      <motion.div
        initial={{ opacity: 0, x: 16 }}
        animate={{ opacity: 1, x: 0 }}
        className="flex gap-3 items-start max-w-2xl ml-auto flex-row-reverse"
      >
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-gray-700 to-gray-600 flex items-center justify-center shrink-0 shadow">
          <User size={16} className="text-gray-200" />
        </div>
        <div className="bg-brand-600/20 border border-brand-500/20 rounded-2xl rounded-tr-sm px-5 py-3 text-sm text-gray-200 leading-relaxed">
          {message.content}
        </div>
      </motion.div>
    );
  }

  // Assistant message
  const structured = message.structuredData;
  const publications = structured?.researchInsights || [];
  const trials = structured?.clinicalTrials || [];
  const overview = structured?.conditionOverview;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex gap-3 items-start max-w-4xl w-full"
    >
      {/* Avatar */}
      <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-brand-600 to-teal-500 flex items-center justify-center shrink-0 shadow-lg shadow-brand-900/50 mt-0.5">
        <Bot size={16} className="text-white" />
      </div>

      <div className="flex-1 min-w-0">
        {/* Synthesis Panel — full 4-section analysis */}
        <SynthesisPanel content={overview} />

        {/* Brief intro card */}
        <div className="glass rounded-2xl rounded-tl-sm px-5 py-4">
          <div className="prose-ai text-sm">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        </div>

        {/* Publications Section */}
        {publications.length > 0 && (
          <StructuredSection
            title={`Research Publications (${publications.length})`}
            icon={<BookOpen size={15} className="text-brand-400" />}
            defaultOpen={publications.length <= 4}
          >
            {publications.map((pub, i) => (
              <PublicationCard key={i} pub={pub} index={i} />
            ))}
          </StructuredSection>
        )}

        {/* Clinical Trials Section */}
        {trials.length > 0 && (
          <StructuredSection
            title={`Clinical Trials (${trials.length})`}
            icon={<FlaskConical size={15} className="text-teal-400" />}
            defaultOpen={trials.length <= 3}
          >
            {trials.map((trial, i) => (
              <ClinicalTrialCard key={i} trial={trial} index={i} />
            ))}
          </StructuredSection>
        )}

        {/* Stats bar */}
        {message.stats && (
          <div className="mt-2 flex items-center gap-3 flex-wrap">
            <span className="flex items-center gap-1 text-xs text-gray-600">
              <TrendingUp size={11} />
              {message.stats.totalPublicationsRetrieved} papers retrieved →&nbsp;
              <span className="text-brand-400 font-medium">{message.stats.topPublicationsShown} shown</span>
            </span>
            <span className="flex items-center gap-1 text-xs text-gray-600">
              <Zap size={11} />
              {message.stats.totalTrialsRetrieved} trials retrieved →&nbsp;
              <span className="text-teal-400 font-medium">{message.stats.topTrialsShown} shown</span>
            </span>
          </div>
        )}
      </div>
    </motion.div>
  );
}
