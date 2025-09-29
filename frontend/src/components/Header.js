import React from 'react';
import '../styles/Header.css';

const Header = () => {
  return (
    <header className="app-header">
      <div className="header-container">
        <div className="logo-section">
          <div className="logo">⚖️</div>
          <div className="title-section">
            <h1>Indian Education Law Assistant</h1>
            <p>Specialized Legal Guidance for Education Matters</p>
          </div>
        </div>
        <div className="header-info">
          <div className="info-badge">
            <span className="badge-label">Specialized in</span>
            <span className="badge-value">Indian Education Law</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;