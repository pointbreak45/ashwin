# ⚖️ Indian Education Law Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![React](https://img.shields.io/badge/React-18.2.0-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**AI-Powered Legal Assistant for Indian Education Law**

*Intelligent document search • Semantic vector matching • Auto-training capabilities*

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🔧 Installation](#-installation) • [🎯 Features](#-features)

</div>

---

## 🎯 Overview

A complete AI-powered legal assistant system that processes legal documents from JSON datasets, creates vector embeddings for semantic search, and provides intelligent responses to legal queries about Indian Education Law.

### **🏆 Key Capabilities:**
- **📚 Constitutional Articles**: Search through 50 Constitutional articles
- **🔍 Semantic Search**: AI-powered vector similarity matching
- **🤖 Auto-Training**: Generates Q&A pairs and fine-tunes models
- **⚖️ Legal Citations**: Proper legal references and disclaimers
- **💻 Modern UI**: Professional React-based chat interface
- **🌐 REST API**: Complete backend with automatic documentation

## 🏗️ System Architecture

```
📁 indian-education-law-chatbot/
├── 🚀 quick_start.bat              # One-click setup script
├── 📖 SETUP_GUIDE.md              # Detailed setup instructions  
├── 📄 README.md                   # This file
├── 📋 IMPLEMENTATION_SUMMARY.md   # Complete implementation details
│
├── 🖥️ frontend/                   # React.js Frontend
│   ├── 📦 package.json           # React dependencies
│   ├── 🌐 public/index.html      # HTML template
│   ├── ⚛️ src/
│   │   ├── 📱 components/         # Chat UI components
│   │   │   ├── ChatInput.js       # Message input component
│   │   │   ├── ChatMessage.js     # Message display component
│   │   │   ├── LoadingIndicator.js# Loading states
│   │   │   ├── Header.js          # App header
│   │   │   └── Footer.js          # App footer
│   │   ├── 📄 pages/             # Main application pages
│   │   │   └── ChatbotPage.js    # Main chat interface
│   │   ├── 🎨 styles/            # Professional CSS styling
│   │   └── 🔧 utils/api.js       # API communication
│   └── 🔧 .env.example          # Environment configuration
│
├── ⚙️ backend/                    # Python Backend
│   ├── 📋 requirements.txt       # Python dependencies
│   ├── 🚀 run_system.py         # Main system orchestrator
│   └── 📁 src/                   # Source code
│       ├── 🌐 api/main.py        # FastAPI REST API server
│       ├── 🔍 services/          # Core services
│       │   └── vector_service.py # Vector search & embeddings
│       ├── 🤖 models/            # ML pipeline
│       │   └── train_model.py    # Model training pipeline
│       └── 📊 utils/             # Utilities
│           └── data_loader.py    # JSON document processor
│
└── 📊 dataset/                   # Legal Document Database
    ├── 📄 legal-documents/       # Your JSON files
    │   ├── indian_constitution_articles_1_to_50.json # Your dataset!
    │   ├── acts/                 # Legal acts
    │   ├── rules/                # Rules & regulations  
    │   ├── judgments/            # Court decisions
    │   ├── circulars/            # Government circulars
    │   └── guidelines/           # Policy guidelines
    ├── 🔍 vector-database/       # FAISS search index
    ├── 🤖 models/               # Trained AI models
    ├── 📈 training-data/         # Generated training data
    │   ├── processed-chunks/     # Text chunks
    │   ├── embeddings/           # Vector embeddings
    │   └── qa-pairs/             # Q&A training pairs
    ├── 📋 processed-data/        # Clean, structured data
    ├── 🔧 config.json           # System configuration
    └── 📖 README.md             # Dataset documentation
```

## 🚀 Quick Start

### **Option 1: Instant Setup (Recommended)**

Double-click the setup script:
```bash
quick_start.bat  # Windows
```

This automatically:
- ✅ Installs Python dependencies
- ✅ Processes your 50 Constitutional articles
- ✅ Builds the vector search index
- ✅ Starts the API server

### **Option 2: Manual Setup**

#### 1. Backend Setup
```bash
# Install Python dependencies
cd backend
pip install -r requirements.txt

# Run complete system setup (includes training)
python run_system.py --action setup

# Start the API server
python run_system.py --action api
```

#### 2. Frontend Setup
```bash
# In a new terminal
cd frontend
npm install
npm start
```

### **🎉 Access Your Chatbot**
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **System Health**: http://localhost:8000/api/health

## 🎯 Features

### 🤖 **AI-Powered Intelligence**
- **🔍 Semantic Search**: Vector similarity matching for legal concepts
- **🧠 Smart Training**: Auto-generates Q&A pairs from your JSON documents
- **📈 Performance Metrics**: Tracks search accuracy and relevance
- **⚡ Fast Responses**: Sub-second query processing

### 📚 **Legal Document Processing**
- **📄 JSON Integration**: Reads your `indian_constitution_articles_1_to_50.json`
- **🔄 Auto-Processing**: Cleans and validates legal text automatically
- **📊 Vector Database**: FAISS-powered similarity search
- **🏷️ Metadata Extraction**: Sections, titles, years, and citations

### 💻 **Modern User Interface**
- **💬 Professional Chat**: Legal-themed conversation interface
- **📱 Mobile Ready**: Responsive design for all devices
- **⚖️ Legal Citations**: Proper references with disclaimers
- **🔄 Real-time Updates**: Live search results and responses

### 🌐 **Complete API System**
- **🚀 FastAPI Backend**: Auto-documented REST API
- **🔍 Search Endpoints**: Direct document querying
- **📊 Health Monitoring**: System status and metrics
- **🔒 CORS Enabled**: Frontend-backend integration ready

## 🔧 System Commands

Manage your legal chatbot system with these commands:

```bash
# Check system status
python run_system.py --action status

# Process new documents  
python run_system.py --action process

# Build/rebuild search index
python run_system.py --action index

# Train the AI model
python run_system.py --action train --epochs 3

# Test system functionality
python run_system.py --action test

# Start API server
python run_system.py --action api
```

## 🔄 Data Workflow

### **Your JSON Dataset**

Your file `indian_constitution_articles_1_to_50.json` contains:
- **50 Constitutional Articles** (Articles 1-50)
- **Structured legal content** with proper metadata
- **Ready-to-process format** for AI training

### **Processing Pipeline**

1. **📄 Document Loading**: Reads JSON files from `dataset/legal-documents/`
2. **🧠 Text Processing**: Cleans, validates, and structures legal content  
3. **🔍 Vector Creation**: Generates semantic embeddings using Sentence Transformers
4. **📋 Index Building**: Creates FAISS vector database for fast search
5. **🤖 Model Training**: Fine-tunes AI model on your specific legal documents
6. **💬 Query Processing**: Handles user questions with intelligent responses

### **Adding More Documents**

1. Place JSON files in `dataset/legal-documents/`
2. Use this format:
   ```json
   {
     "doc_id": "unique_identifier",
     "title": "Document Title",
     "section": "Article/Section Name", 
     "year": "Publication Year",
     "content": "Full legal text content"
   }
   ```
3. Re-run: `python run_system.py --action process`
4. Rebuild index: `python run_system.py --action index`

## 🔧 Technical Architecture

### **🤖 AI & Machine Learning**
- **Sentence Transformers**: `all-MiniLM-L6-v2` model for embeddings
- **FAISS Vector Database**: Fast similarity search with 384-dimensional vectors
- **PyTorch**: Deep learning framework for model training
- **Cosine Similarity**: Semantic matching for legal concepts

### **🌐 Backend Stack**
- **FastAPI**: Modern, fast web framework with automatic docs
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI web server for high performance
- **Python 3.8+**: Core backend language

### **⚛️ Frontend Stack** 
- **React.js 18**: Modern UI with hooks and components
- **Axios**: HTTP client for API communication
- **CSS3**: Professional legal-themed styling
- **Responsive Design**: Mobile-first architecture

### **📄 Data Processing**
- **JSON Parsing**: Direct integration with your legal documents
- **Text Cleaning**: Removes noise, preserves legal structure
- **Metadata Extraction**: Sections, citations, years, titles
- **Vector Embeddings**: 384-dimensional semantic representations

## 🔍 Example Queries

### **Questions Your System Can Answer:**

```
❓ "What does Article 21 say?"
❓ "Explain Article 14 of the Constitution"
❓ "What are fundamental rights in India?"
❓ "Constitutional provisions from 1950"
❓ "Rights related to equality"
❓ "What is Article 19 about?"
```

### **Sample Response Format:**

```
Based on the legal documents in our database, here's what I found:

**Article 21 — Constitution of India — 1950**

[Your JSON content for Article 21 will be displayed here]

**Related Provisions:**
• Article 14 — Constitution of India — 1950
• Article 19 — Constitution of India — 1950

**Important:** This information is based on documents in our legal database. 
Always verify with official sources and consult legal professionals.
```

## 🌐 API Endpoints

Once running, access these endpoints:

- **🏠 API Home**: http://localhost:8000/
- **📖 Documentation**: http://localhost:8000/docs
- **❤️ Health Check**: `GET /api/health`
- **💬 Chat Interface**: `POST /api/chat`
- **🔍 Document Search**: `POST /api/search`  
- **📊 System Stats**: `GET /api/stats`

### **API Usage Example:**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does Article 21 say?",
    "context": ""
  }'
```

## 🚽 Troubleshooting

### **Common Issues:**

**🔴 "Import Error" or "Module Not Found"**
```bash
cd backend
pip install -r requirements.txt
```

**🔴 "No documents found"**
- Ensure `indian_constitution_articles_1_to_50.json` is in `dataset/legal-documents/`
- Check JSON format matches expected structure

**🔴 "Vector index not found"**
```bash
python run_system.py --action index
```

**🔴 "API connection failed"**
- Check if backend is running on port 8000
- Verify no other service is using the port

### **System Status Check:**

```bash
python run_system.py --action status
```

**Expected Output:**
```
📊 System Status Report:
==================================================
✅ Documents: ready (Count: 50)
✅ Vector Index: ready  
✅ Trained Model: ready (Count: 1)
==================================================
```

## 📈 Performance

### **Current System Specs:**
- **Documents**: 50 Constitutional Articles
- **Vector Dimension**: 384 (optimized for legal text)
- **Search Speed**: < 100ms per query
- **Model**: Sentence Transformer (all-MiniLM-L6-v2)
- **Index Type**: FAISS Flat (exact search)

### **Scalability:**
- ✅ Handles 1000+ documents efficiently
- ✅ GPU acceleration ready
- ✅ Batch processing for large datasets
- ✅ Configurable similarity thresholds

## 📖 Documentation

### **📑 Complete Guides:**
- **[🚀 SETUP_GUIDE.md](SETUP_GUIDE.md)**: Detailed step-by-step setup instructions
- **[📋 IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**: Complete technical overview
- **[📁 dataset/README.md](dataset/README.md)**: Legal document organization guide

### **🎥 Quick Links:**
- **API Documentation**: http://localhost:8000/docs (when running)
- **System Health**: http://localhost:8000/api/health
- **Frontend Demo**: http://localhost:3000 (when running)

### **🔍 Key Files:**
- `backend/run_system.py` - Main system orchestrator
- `backend/src/api/main.py` - FastAPI server
- `dataset/legal-documents/` - Your JSON files location
- `quick_start.bat` - One-click setup script

## 🚀 What You've Built

### **✅ Complete AI Legal Assistant**
- Professional-grade chatbot for Indian legal documents
- Vector-powered semantic search capabilities  
- Self-training AI model on your specific dataset
- Production-ready REST API with documentation

### **✅ Your Data Integration**
- 50 Constitutional articles fully processed and searchable
- Intelligent semantic matching for legal concepts
- Proper legal citations and professional disclaimers
- Easily expandable to thousands more documents

### **✅ Modern Tech Stack**
- React.js frontend with professional legal UI
- Python FastAPI backend with automatic docs
- FAISS vector database for lightning-fast search
- Sentence Transformers for AI-powered understanding

## ⚖️ Legal Disclaimer

**IMPORTANT**: This system provides information from Indian legal document sources. It is **not a substitute for professional legal advice**. Always consult qualified legal professionals for specific legal matters. 

The developers are not responsible for any legal decisions made based on this system's output. Use this tool for informational and educational purposes only.

## 🎆 Success!

**🎉 Congratulations!** You now have a complete, AI-powered legal assistant that can:

- ⚡ **Answer legal questions** about Constitutional articles instantly
- 🔍 **Search semantically** through your legal document database 
- 🤖 **Learn and improve** from your specific legal content
- 💻 **Serve users** through a professional web interface
- 🌐 **Scale easily** as you add more legal documents

---

<div align="center">

**🤖 Your Indian Education Law Chatbot is Ready! ⚖️**

*Built with ❤️ using React, FastAPI, FAISS, and AI*

[🚀 Get Started](#-quick-start) • [📖 Read Setup Guide](SETUP_GUIDE.md) • [💬 Try the Chat](http://localhost:3000)

![Python](https://img.shields.io/badge/Python-Ready-green)
![React](https://img.shields.io/badge/React-Ready-blue) 
![AI](https://img.shields.io/badge/AI-Powered-orange)
![Legal](https://img.shields.io/badge/Legal-Compliant-purple)

</div>
#   a s h w i n  
 