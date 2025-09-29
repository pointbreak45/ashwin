import React from 'react';
import '../styles/Footer.css';

const Footer = () => {
  return (
    <footer className="app-footer">
      <div className="footer-container">
        <div className="footer-content">
          <div className="disclaimer-section">
            <h4>⚠️ Important Legal Disclaimer</h4>
            <p>
              This system provides information from Indian education law sources. 
              It is not a substitute for professional legal advice. Always consult 
              qualified legal professionals for specific legal matters.
            </p>
          </div>
          
          <div className="footer-info">
            <div className="footer-links">
              <div className="link-section">
                <h5>Resources</h5>
                <ul>
                  <li>Right to Education Act</li>
                  <li>UGC Regulations</li>
                  <li>CBSE Guidelines</li>
                  <li>State Education Policies</li>
                </ul>
              </div>
              <div className="link-section">
                <h5>Legal Sources</h5>
                <ul>
                  <li>Ministry of Education</li>
                  <li>Supreme Court Judgments</li>
                  <li>High Court Orders</li>
                  <li>Educational Tribunals</li>
                </ul>
              </div>
            </div>
            
            <div className="copyright">
              <p>&copy; 2024 Indian Education Law Assistant. All rights reserved.</p>
              <p className="version">Version 1.0.0 | Legal Database Updated</p>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;