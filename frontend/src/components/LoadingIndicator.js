import React from 'react';
import '../styles/LoadingIndicator.css';

const LoadingIndicator = () => {
  return (
    <div className="loading-indicator">
      <div className="loading-message">
        <div className="loading-avatar">⚖️</div>
        <div className="loading-content">
          <div className="loading-dots">
            <span>Legal Assistant is analyzing your query</span>
            <div className="dots">
              <span>.</span>
              <span>.</span>
              <span>.</span>
            </div>
          </div>
          <div className="loading-subtitle">
            Searching through Indian education law database...
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingIndicator;