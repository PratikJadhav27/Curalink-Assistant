const express = require('express');
const Conversation = require('../models/Conversation');

const router = express.Router();

// GET /api/history - list all conversations (sessions)
router.get('/', async (req, res) => {
  try {
    const conversations = await Conversation.find(
      {},
      { sessionId: 1, title: 1, patientContext: 1, createdAt: 1, updatedAt: 1 }
    ).sort({ updatedAt: -1 }).limit(50);
    res.json(conversations);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/history/:sessionId - full conversation messages
router.get('/:sessionId', async (req, res) => {
  try {
    const conversation = await Conversation.findOne({ sessionId: req.params.sessionId });
    if (!conversation) return res.status(404).json({ error: 'Conversation not found' });
    res.json(conversation);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// DELETE /api/history/:sessionId - delete a conversation
router.delete('/:sessionId', async (req, res) => {
  try {
    await Conversation.deleteOne({ sessionId: req.params.sessionId });
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
