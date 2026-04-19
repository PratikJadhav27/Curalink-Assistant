import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { User, MapPin, Brain, ChevronDown, ChevronUp, Stethoscope } from 'lucide-react';
import { useChatStore } from '../store/chatStore';

export default function PatientContextPanel() {
  const { patientContext, setPatientContext } = useChatStore();
  const [open, setOpen] = useState(false);

  const hasContext = patientContext.name || patientContext.disease || patientContext.location;

  return (
    <div className="glass rounded-xl border border-white/5 overflow-hidden">
      <button
        onClick={() => setOpen((p) => !p)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/3 transition-colors"
      >
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-lg bg-teal-500/20 flex items-center justify-center">
            <Stethoscope size={13} className="text-teal-400" />
          </div>
          <span className="text-sm font-semibold text-white">Patient Context</span>
          {hasContext && (
            <span className="badge badge-teal text-xs">Active</span>
          )}
        </div>
        {open ? <ChevronUp size={15} className="text-gray-500" /> : <ChevronDown size={15} className="text-gray-500" />}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 grid grid-cols-2 gap-3 border-t border-white/5 pt-3">
              <div>
                <label className="flex items-center gap-1.5 text-xs text-gray-500 mb-1.5 font-medium">
                  <User size={11} /> Patient Name
                </label>
                <input
                  type="text"
                  placeholder="e.g. John Smith"
                  value={patientContext.name}
                  onChange={(e) => setPatientContext({ name: e.target.value })}
                  className="input-field text-xs py-2"
                />
              </div>

              <div>
                <label className="flex items-center gap-1.5 text-xs text-gray-500 mb-1.5 font-medium">
                  <Brain size={11} /> Disease / Condition
                </label>
                <input
                  type="text"
                  placeholder="e.g. Parkinson's disease"
                  value={patientContext.disease}
                  onChange={(e) => setPatientContext({ disease: e.target.value })}
                  className="input-field text-xs py-2"
                />
              </div>

              <div>
                <label className="flex items-center gap-1.5 text-xs text-gray-500 mb-1.5 font-medium">
                  <MapPin size={11} /> Location
                </label>
                <input
                  type="text"
                  placeholder="e.g. Toronto, Canada"
                  value={patientContext.location}
                  onChange={(e) => setPatientContext({ location: e.target.value })}
                  className="input-field text-xs py-2"
                />
              </div>

              <div>
                <label className="flex items-center gap-1.5 text-xs text-gray-500 mb-1.5 font-medium">
                  Additional Info
                </label>
                <input
                  type="text"
                  placeholder="Age, symptoms, etc."
                  value={patientContext.additionalInfo}
                  onChange={(e) => setPatientContext({ additionalInfo: e.target.value })}
                  className="input-field text-xs py-2"
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
