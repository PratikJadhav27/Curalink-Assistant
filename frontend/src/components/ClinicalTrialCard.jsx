import React from 'react';
import { motion } from 'framer-motion';
import { ExternalLink, MapPin, Phone, Activity, CheckCircle, Clock } from 'lucide-react';

function getStatusBadge(status = '') {
  const s = status.toUpperCase();
  if (s === 'RECRUITING') return { cls: 'badge-green', icon: <Activity size={10} /> };
  if (s === 'COMPLETED') return { cls: 'badge-blue', icon: <CheckCircle size={10} /> };
  if (s === 'ACTIVE, NOT RECRUITING') return { cls: 'badge-yellow', icon: <Clock size={10} /> };
  return { cls: 'badge-purple', icon: <Clock size={10} /> };
}

export default function ClinicalTrialCard({ trial, index }) {
  const { cls, icon } = getStatusBadge(trial.status);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.07 }}
      className="glass rounded-xl p-4 border border-white/5 hover:border-teal-500/30 transition-all duration-200"
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2 mb-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`badge ${cls}`}>
            {icon}
            {trial.status || 'Unknown'}
          </span>
          {trial.phase && (
            <span className="badge badge-purple">{trial.phase}</span>
          )}
        </div>
        {trial.url && (
          <a
            href={trial.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-600 hover:text-teal-400 transition-colors"
          >
            <ExternalLink size={14} />
          </a>
        )}
      </div>

      {/* Title */}
      <p className="text-sm font-semibold text-white leading-snug mb-3 line-clamp-2">
        {trial.title}
      </p>

      {/* NCT ID */}
      {trial.nctId && (
        <p className="text-xs font-mono text-teal-500 mb-2">{trial.nctId}</p>
      )}

      {/* Location */}
      {trial.location && (
        <div className="flex items-center gap-1.5 text-xs text-gray-500 mb-1.5">
          <MapPin size={11} className="text-teal-500 shrink-0" />
          <span>{trial.location}</span>
        </div>
      )}

      {/* Contact */}
      {trial.contact && (
        <div className="flex items-center gap-1.5 text-xs text-gray-500 mb-2">
          <Phone size={11} className="text-brand-400 shrink-0" />
          <span className="truncate">{trial.contact}</span>
        </div>
      )}

      {/* Eligibility snippet */}
      {trial.eligibility && (
        <div className="mt-2 pt-2 border-t border-white/5">
          <p className="text-xs text-gray-600 font-medium mb-1">Eligibility:</p>
          <p className="text-xs text-gray-500 line-clamp-2 leading-relaxed">
            {trial.eligibility}
          </p>
        </div>
      )}
    </motion.div>
  );
}
