import React, { useState } from 'react';
import '../styles/ChatInput.css';

const ChatInput = ({ onSendMessage, isLoading }) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSendMessage(message);
      setMessage('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const suggestedQuestions = [
    "What are the key provisions of the Right to Education Act?",
    "What are UGC guidelines for university admissions?",
    "What are student rights in educational institutions?",
    "How can I file a complaint against an educational institution?"
  ];

  const handleSuggestedQuestion = (question) => {
    if (!isLoading) {
      setMessage(question);
    }
  };

  return (
    <div className="chat-input-container">
      {message === '' && (
        <div className="suggested-questions">
          <p className="suggestions-title">Try asking:</p>
          <div className="suggestions-grid">
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                className="suggestion-btn"
                onClick={() => handleSuggestedQuestion(question)}
                disabled={isLoading}
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="input-form">
        <div className="input-wrapper">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me about Indian Education Law... (Press Enter to send, Shift+Enter for new line)"
            className="message-input"
            rows="3"
            disabled={isLoading}
          />
          <button 
            type="submit" 
            className="send-button"
            disabled={isLoading || !message.trim()}
          >
            {isLoading ? (
              <span className="loading-spinner">⏳</span>
            ) : (
              <span className="send-icon">➤</span>
            )}
          </button>
        </div>
        <div className="input-footer">
          <small className="input-disclaimer">
            This chatbot provides information from Indian education law sources. 
            Always verify important legal information with official sources.
          </small>
        </div>
      </form>
    </div>
  );
};

export default ChatInput;