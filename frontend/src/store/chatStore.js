import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const useChatStore = create(
  persist(
    (set, get) => ({
      sessionId: null,
      messages: [],
      conversations: [],
      patientContext: { name: '', disease: '', location: '', additionalInfo: '' },
      isLoading: false,
      error: null,
      sidebarOpen: true,

      setPatientContext: (ctx) =>
        set((s) => ({ patientContext: { ...s.patientContext, ...ctx } })),

      setSessionId: (id) => set({ sessionId: id }),

      setSidebarOpen: (open) => set({ sidebarOpen: open }),

      addMessage: (msg) =>
        set((s) => ({ messages: [...s.messages, msg] })),

      setLoading: (loading) => set({ isLoading: loading }),

      setError: (error) => set({ error }),

      clearError: () => set({ error: null }),

      startNewChat: () =>
        set({ sessionId: null, messages: [], error: null }),

      setConversations: (convs) => set({ conversations: convs }),

      loadConversation: (sessionId, messages) =>
        set({ sessionId, messages, error: null }),
    }),
    {
      name: 'curalink-store',
      partialize: (state) => ({
        patientContext: state.patientContext,
        sidebarOpen: state.sidebarOpen,
      }),
    }
  )
);
