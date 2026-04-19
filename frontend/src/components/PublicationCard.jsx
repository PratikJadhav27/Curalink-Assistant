import React from 'react';
import { motion } from 'framer-motion';
import { ExternalLink, Calendar, Users, BookOpen, Award } from 'lucide-react';

const SOURCE_BADGE_MAP = {
  pubmed: 'badge-red',
  openalex: 'badge-blue',
  default: 'badge-purple',
};

function getSourceBadge(source = '') {
  const s = source.toLowerCase();
  if (s.includes('pubmed')) return SOURCE_BADGE_MAP.pubmed;
  if (s.includes('openalex')) return SOURCE_BADGE_MAP.openalex;
  return SOURCE_BADGE_MAP.default;
}

function getSourceShort(source = '') {
  if (source.toLowerCase().includes('pubmed')) return 'PubMed';
  if (source.toLowerCase().includes('openalex')) return 'OpenAlex';
  return source.split('/')[0];
}

export default function PublicationCard({ pub, index }) {
  const relevance = pub.relevanceScore ? Math.round(pub.relevanceScore * 100) : null;

  return (
    <motion.a
      href={pub.url || '#'}
      target="_blank"
      rel="noopener noreferrer"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      className="source-card block group no-underline"
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={getSourceBadge(pub.source)}>
            <BookOpen size={10} />
            {getSourceShort(pub.source)}
          </span>
          {pub.year && (
            <span className="badge badge-yellow">
              <Calendar size={10} /> {pub.year}
            </span>
          )}
          {relevance !== null && (
            <span className="badge badge-teal text-xs">
              <Award size={10} /> {relevance}% match
            </span>
          )}
        </div>
        <ExternalLink
          size={14}
          className="text-gray-600 group-hover:text-brand-400 transition-colors shrink-0 mt-0.5"
        />
      </div>

      {/* Title */}
      <p className="text-sm font-semibold text-white leading-snug mb-2 group-hover:text-brand-300 transition-colors line-clamp-2">
        {pub.title || 'Untitled'}
      </p>

      {/* Authors */}
      {pub.authors?.length > 0 && (
        <div className="flex items-center gap-1.5 text-xs text-gray-500 mb-2">
          <Users size={11} />
          <span className="truncate">
            {pub.authors.slice(0, 3).join(', ')}
            {pub.authors.length > 3 ? ` +${pub.authors.length - 3} more` : ''}
          </span>
        </div>
      )}

      {/* Abstract snippet */}
      {pub.abstract && (
        <p className="text-xs text-gray-500 leading-relaxed line-clamp-2 mt-1">
          {pub.abstract}
        </p>
      )}
    </motion.a>
  );
}
