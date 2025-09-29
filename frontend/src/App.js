import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ChatbotPage from './pages/ChatbotPage';
import Header from './components/Header';
import Footer from './components/Footer';
import './styles/App.css';

function App() {
  return (
    <div className="App">
      <Router>
        <Header />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<ChatbotPage />} />
            <Route path="/chat" element={<ChatbotPage />} />
          </Routes>
        </main>
        <Footer />
      </Router>
    </div>
  );
}

export default App;