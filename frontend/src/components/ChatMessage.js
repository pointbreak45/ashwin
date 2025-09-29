import React from 'react';
import '../styles/ChatMessage.css';

const ChatMessage = ({ message }) => {
  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const formatContent = (content) => {
    // Convert markdown-like formatting to HTML
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br>');
  };

  return (
    <div className={`chat-message ${message.type}`}>
      <div className="message-content">
        <div className="message-header">
          <span className="message-sender">
            {message.type === 'bot' ? '⚖️ Legal Assistant' : '👤 You'}
          </span>
          <span className="message-time">
            {formatTimestamp(message.timestamp)}
          </span>
        </div>
        <div 
          className="message-text"
          dangerouslySetInnerHTML={{ __html: formatContent(message.content) }}
        />
        {message.type === 'bot' && (
          <div className="bot-disclaimer">
            <small>
              Information provided is based on available legal documents. 
              Consult a legal professional for official advice.
            </small>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;