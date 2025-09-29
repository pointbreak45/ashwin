# 🚀 Indian Education Law Chatbot - Complete Setup Guide

This guide will help you set up and run the complete Indian Education Law Chatbot system with your JSON dataset.

## 📋 Prerequisites

1. **Python 3.8+** installed on your system
2. **Node.js 16+** for the React frontend
3. **Your JSON dataset** in the correct location

## 🎯 Quick Start (Automated Setup)

### Step 1: Install Python Dependencies

```bash
# Navigate to the backend directory
cd backend

# Install all required packages
pip install -r requirements.txt
```

### Step 2: Run Complete System Setup

```bash
# Run the automated setup (includes training)
python run_system.py --action setup

# Or skip model training for faster setup
python run_system.py --action setup --no-train
```

### Step 3: Start the Backend API

```bash
# Start the API server
python run_system.py --action api

# Or alternatively
python src/api/main.py
```

### Step 4: Start the Frontend

```bash
# In a new terminal, navigate to frontend directory
cd ../frontend

# Install dependencies
npm install

# Start the React app
npm start
```

**🎉 Your chatbot is now running at http://localhost:3000!**

---

## 🔧 Manual Setup (Step by Step)

If you prefer to run each step manually or need to troubleshoot:

### Step 1: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Process Your JSON Documents

```bash
# This will read your JSON file and process the legal documents
python run_system.py --action process
```

**Expected Output:**
```
📄 Processing legal documents...
✅ Successfully processed 50 documents
📊 Total words: 1,250
📊 Unique sections: 50
```

### Step 3: Build Vector Search Index

```bash
# This creates embeddings and builds the search index
python run_system.py --action index
```

**Expected Output:**
```
🔍 Building vector search index...
✅ Vector index built successfully!
📊 Indexed vectors: 50
📊 Vector dimension: 384
```

### Step 4: Train the Model (Optional)

```bash
# Train the model on your legal documents
python run_system.py --action train --epochs 3
```

**Expected Output:**
```
🤖 Training legal language model...
✅ Model training completed successfully!
📊 Training pairs: 350
📊 Training examples: 280
```

### Step 5: Test the System

```bash
# Test if everything is working
python run_system.py --action test
```

### Step 6: Start the API Server

```bash
# Start the backend API
python run_system.py --action api
```

### Step 7: Setup Frontend

```bash
cd ../frontend
npm install
npm start
```

---

## 📊 System Status Check

You can check if all components are ready:

```bash
cd backend
python run_system.py --action status
```

**Example Output:**
```
📊 System Status Report:
==================================================
✅ Documents: ready
   Count: 50
✅ Vector Index: ready
✅ Trained Model: ready
   Count: 1
==================================================
```

---

## 🎯 Your Dataset Integration

### Current Dataset

Your JSON file `indian_constitution_articles_1_to_50.json` is already detected! It contains:

- **50 Articles** from the Constitution of India
- **Structured format** with doc_id, title, section, year, and content
- **Ready for processing** by the system

### Adding More Legal Documents

1. **Place JSON files** in `dataset/legal-documents/`
2. **Follow the same structure**:
   ```json
   [
     {
       "doc_id": "unique_id",
       "title": "Document Title",
       "section": "Section Name",
       "year": "Year",
       "content": "Full legal text content"
     }
   ]
   ```
3. **Re-run processing**:
   ```bash
   python run_system.py --action process
   python run_system.py --action index
   ```

---

## 🔍 How It Works

### 1. **Document Processing**
- Reads your JSON files
- Cleans and validates legal text
- Extracts metadata (sections, years, titles)

### 2. **Vector Search**
- Creates semantic embeddings using sentence transformers
- Builds FAISS index for fast similarity search
- Enables intelligent document retrieval

### 3. **Model Training** (Optional)
- Generates question-answer pairs from your documents
- Fine-tunes the model on legal language
- Improves search accuracy and relevance

### 4. **Chat API**
- Searches relevant documents for user queries
- Generates structured legal responses
- Provides citations and disclaimers

---

## 🌐 API Endpoints

Once the backend is running, you can access:

- **API Documentation**: http://localhost:8000/docs
- **System Health**: http://localhost:8000/api/health
- **Chat Endpoint**: POST http://localhost:8000/api/chat
- **Search Endpoint**: POST http://localhost:8000/api/search

### Example API Usage

```bash
# Test the chat endpoint
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does Article 21 say?",
    "context": ""
  }'
```

---

## 🎨 Frontend Features

The React frontend provides:

- **Interactive Chat Interface** - Clean, professional UI
- **Real-time Search** - Instant responses from your legal database
- **Source Citations** - Shows relevant legal sections
- **Mobile Responsive** - Works on all devices
- **Legal Disclaimers** - Built-in professional disclaimers

---

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt
```

**2. No Documents Found**
- Check that your JSON file is in `dataset/legal-documents/`
- Verify the JSON format matches the expected structure

**3. Vector Index Issues**
```bash
# Rebuild the index
python run_system.py --action index
```

**4. API Connection Issues**
- Check if backend is running on port 8000
- Verify CORS settings in the API configuration

### Getting Help

1. **Check Logs**: Look at `system.log` for detailed error information
2. **System Status**: Run `python run_system.py --action status`
3. **Test Components**: Use individual action commands to isolate issues

---

## 📈 Performance Optimization

### For Large Datasets (1000+ documents)

1. **Use GPU acceleration** (if available):
   - Install `faiss-gpu` instead of `faiss-cpu`
   - The system will automatically detect and use GPU

2. **Adjust batch sizes**:
   ```bash
   python run_system.py --action train --batch-size 32
   ```

3. **Use different index types**:
   - Modify `config.json` to use "IVF" or "HNSW" for faster search

---

## 🚀 Next Steps

1. **Add More Legal Documents**: Expand your dataset with more JSON files
2. **Customize Responses**: Modify the answer generation logic in `src/api/main.py`
3. **Fine-tune Model**: Experiment with different training parameters
4. **Deploy**: Use Docker or cloud services for production deployment

---

## 📞 Support

If you encounter any issues:

1. Check the logs in `backend/system.log`
2. Verify system status with `python run_system.py --action status`
3. Review the API documentation at http://localhost:8000/docs
4. Ensure all dependencies are correctly installed

**Happy Legal Chatbot Building! ⚖️🤖**