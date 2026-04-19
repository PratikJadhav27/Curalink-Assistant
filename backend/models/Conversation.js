const mongoose = require('mongoose');

const MessageSchema = new mongoose.Schema({
  role: { type: String, enum: ['user', 'assistant'], required: true },
  content: { type: String, required: true },
  structuredData: {
    conditionOverview: String,
    researchInsights: [
      {
        title: String,
        authors: [String],
        year: Number,
        source: String,
        url: String,
        abstract: String,
        relevanceScore: Number,
      },
    ],
    clinicalTrials: [
      {
        title: String,
        status: String,
        eligibility: String,
        location: String,
        contact: String,
        nctId: String,
        url: String,
      },
    ],
  },
  timestamp: { type: Date, default: Date.now },
});

const ConversationSchema = new mongoose.Schema(
  {
    sessionId: { type: String, required: true, unique: true, index: true },
    patientContext: {
      name: String,
      disease: String,
      location: String,
      additionalInfo: String,
    },
    messages: [MessageSchema],
    title: String,
  },
  { timestamps: true }
);

// Auto-generate title from first user message
ConversationSchema.pre('save', function (next) {
  if (this.messages.length > 0 && !this.title) {
    const firstUserMsg = this.messages.find((m) => m.role === 'user');
    if (firstUserMsg) {
      this.title = firstUserMsg.content.substring(0, 60) + (firstUserMsg.content.length > 60 ? '...' : '');
    }
  }
  next();
});

module.exports = mongoose.model('Conversation', ConversationSchema);
