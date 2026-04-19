import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:5000/api',
  timeout: 120000,
});

export const sendMessage = async ({ message, sessionId, patientContext }) => {
  const res = await api.post('/chat', { message, sessionId, patientContext });
  return res.data;
};

export const fetchHistory = async () => {
  const res = await api.get('/history');
  return res.data;
};

export const fetchConversation = async (sessionId) => {
  const res = await api.get(`/history/${sessionId}`);
  return res.data;
};

export const deleteConversation = async (sessionId) => {
  const res = await api.delete(`/history/${sessionId}`);
  return res.data;
};
