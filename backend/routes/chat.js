const express = require('express');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const Conversation = require('../models/Conversation');

const router = express.Router();
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

// POST /api/chat - Send a message
router.post('/', async (req, res) => {
  try {
    const { message, sessionId, patientContext } = req.body;

    if (!message) return res.status(400).json({ error: 'Message is required' });

    // Find or create conversation
    const sid = sessionId || uuidv4();
    let conversation = await Conversation.findOne({ sessionId: sid });

    if (!conversation) {
      conversation = new Conversation({
        sessionId: sid,
        patientContext: patientContext || {},
        messages: [],
      });
    } else if (patientContext) {
      // Update patient context if provided
      conversation.patientContext = { ...conversation.patientContext, ...patientContext };
    }

    // Append user message
    conversation.messages.push({ role: 'user', content: message });

    // Build conversation history for AI (last 6 turns = 3 exchanges)
    const history = conversation.messages.slice(-12).map((m) => ({
      role: m.role,
      content: m.content,
    }));

    // Call AI service
    const aiResponse = await axios.post(
      `${AI_SERVICE_URL}/research`,
      {
        query: message,
        patient_context: conversation.patientContext,
        conversation_history: history.slice(0, -1), // history excluding latest user msg
      },
      { timeout: 120000 }
    );

    const aiData = aiResponse.data;

    // Append assistant answer
    conversation.messages.push({
      role: 'assistant',
      content: aiData.answer,
      structuredData: {
        conditionOverview: aiData.conditionOverview,
        researchInsights: aiData.publications || [],
        clinicalTrials: aiData.clinicalTrials || [],
      },
    });

    await conversation.save();

    res.json({
      sessionId: sid,
      answer: aiData.answer,
      conditionOverview: aiData.conditionOverview,
      publications: aiData.publications || [],
      clinicalTrials: aiData.clinicalTrials || [],
      queryExpanded: aiData.queryExpanded,
    });
  } catch (err) {
    console.error('Chat error:', err.message);
    if (err.code === 'ECONNREFUSED') {
      return res.status(503).json({ error: 'AI service unavailable. Please try again shortly.' });
    }
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
