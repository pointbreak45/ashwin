@echo off
echo ==============================================
echo  Indian Education Law Chatbot - Quick Start
echo ==============================================
echo.

echo 📦 Installing Python dependencies...
cd backend
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install Python dependencies
    pause
    exit /b 1
)

echo.
echo 🚀 Setting up the system (this may take a few minutes)...
python run_system.py --action setup --no-train
if %errorlevel% neq 0 (
    echo ❌ System setup failed
    pause
    exit /b 1
)

echo.
echo ✅ Backend setup complete! Starting API server...
echo 🌐 API will be available at http://localhost:8000
echo 📖 API docs available at http://localhost:8000/docs
echo.
echo Starting server... (Press Ctrl+C to stop)
python run_system.py --action api