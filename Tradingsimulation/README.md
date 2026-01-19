
# TradingSimulation Frontend

## ⚠️ Important Notice

**This frontend is a learning/reference version, for reference only.**

- This frontend implementation is primarily for learning and demonstration purposes, **not production-ready code**
- Some features may be incomplete, contain bugs, or use mock data
- Some components may not be fully integrated with the backend API
- Some UI interactions may have issues

**If you need frontend functionality for production use, please improve or rebuild it according to your actual requirements.**

The existence of this frontend does not affect the use of backend core functionality. You can directly use the backend API for development and integration.

---

# Terminal 1: Start backend
./utilities/shell/start_backend.sh

# Terminal 2: Start frontend
cd Tradingsimulation
npm install
npm run dev

**Note:** 
- Trading agents are automatically started by the backend when you click "Start Trading" in the web interface. You don't need to run `main.py` manually.
- The backend script is located at `utilities/shell/start_backend.sh`, not `./start_backend.sh` in the root directory.

###  Access the Dashboard

Open your browser and navigate to:
- **Frontend Dashboard**: http://localhost:5173
- **API Documentation**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

###  Start Trading

1. Navigate to the dashboard at http://localhost:5173
2. Click **"Start Trading"** button
3. The backend will automatically launch `main.py` as a background job
4. Monitor agent decisions in real-time
5. View positions, PnL, and news feed
