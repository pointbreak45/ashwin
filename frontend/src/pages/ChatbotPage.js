import React, { useState, useRef, useEffect } from 'react';
import ChatMessage from '../components/ChatMessage';
import ChatInput from '../components/ChatInput';
import LoadingIndicator from '../components/LoadingIndicator';
import { sendChatMessage } from '../utils/api';
import '../styles/ChatbotPage.css';

const ChatbotPage = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: `Welcome to the Indian Education Law Assistant! 

I specialize in Indian Education Law and can help you with queries related to:
• Right to Education Act (RTE)
• University Grants Commission (UGC) regulations
• Central Board of Secondary Education (CBSE) rules
• State education policies and regulations
• Educational institution compliance
• Student rights and obligations

Please ask your question, and I'll provide information based on official Indian education law sources.

**Important:** This system provides information from Indian education law sources. It is not a substitute for professional legal advice.`,
      timestamp: new Date()
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (userMessage) => {
    if (!userMessage.trim()) return;

    // Add user message
    const userMsg = {
      id: messages.length + 1,
      type: 'user',
      content: userMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    try {
      // Call API to get bot response
      const response = await sendChatMessage(userMessage);
      
      const botMsg = {
        id: messages.length + 2,
        type: 'bot',
        content: response.data.answer || "I apologize, but I couldn't process your request. Please try again or rephrase your question.",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMsg]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMsg = {
        id: messages.length + 2,
        type: 'bot',
        content: "No clear answer found in the uploaded documents. Please consult official sources or a lawyer.\n\nThis system provides information from Indian education law sources. It is not a substitute for professional legal advice.",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([
      {
        id: 1,
        type: 'bot',
        content: `Welcome to the Indian Education Law Assistant! 

I specialize in Indian Education Law and can help you with queries related to:
• Right to Education Act (RTE)
• University Grants Commission (UGC) regulations
• Central Board of Secondary Education (CBSE) rules
• State education policies and regulations
• Educational institution compliance
• Student rights and obligations

Please ask your question, and I'll provide information based on official Indian education law sources.

**Important:** This system provides information from Indian education law sources. It is not a substitute for professional legal advice.`,
        timestamp: new Date()
      }
    ]);
  };

  return (
    <div className="chatbot-page">
      <div className="chat-container">
        <div className="chat-header">
          <div className="header-content">
            <h1>Indian Education Law Assistant</h1>
            <p>Specialized legal assistant for Indian education law queries</p>
          </div>
          <button className="clear-chat-btn" onClick={clearChat}>
            Clear Chat
          </button>
        </div>
        
        <div className="messages-container">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          {isLoading && <LoadingIndicator />}
          <div ref={messagesEndRef} />
        </div>
        
        <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
      </div>
    </div>
  );
};

export default ChatbotPage;