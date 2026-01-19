# AStock Arena Architecture

This document provides a comprehensive overview of the AStock Arena system architecture, including design decisions, component interactions, and data flows.

## Table of Contents

- [System Overview](#system-overview)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Agent Architecture](#agent-architecture)
- [Shared Prefetch System](#shared-prefetch-system)
- [API Layer](#api-layer)
- [Frontend Architecture](#frontend-architecture)
- [Database & Storage](#database--storage)
- [Deployment Architecture](#deployment-architecture)

## System Overview

AStock Arena is designed as a multi-layered system with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                 Presentation Layer                       │
│              (Vite + React Dashboard)                    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼ REST API
┌─────────────────────────────────────────────────────────┐
│                  Application Layer                       │
│                  (FastAPI Backend)                       │
└─────────────────────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Data Manager │  │Agent Spawner │  │Job Manager   │
└──────────────┘  └──────────────┘  └──────────────┘
          │              │              
          ▼              ▼              
┌─────────────────────────────────────────────────────────┐
│                    Agent Layer                           │
│        (Process-Isolated Trading Agents)                 │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Data Layer                             │
│    (Market Data, News, Snapshots, Positions)            │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              External Services                           │
│      (TinySoft API, LLM APIs, News Sources)             │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. FastAPI Backend (`api_server.py`)

**Responsibilities:**
- HTTP API endpoints for frontend
- Job management (start/stop/status)
- Live data aggregation
- Backup coordination

**Key Features:**
- Async request handling
- Process spawning and monitoring
- CORS enabled for frontend
- Comprehensive error handling

**Main Endpoints:**
```python
POST /api/run-trading        # Start trading job
GET  /api/job/{job_id}       # Get job status
GET  /api/jobs               # List all jobs
POST /api/stop/{job_id}      # Stop a running job
GET  /api/live/pnl-series    # Get PnL time series
GET  /api/live/current-positions  # Get current positions
GET  /api/live/model-stats   # Get model statistics
GET  /api/live/recent-decisions  # Get recent decisions
POST /api/backup             # Create backup
GET  /                       # Health check
```

### 2. Data Manager (`data_manager.py`)

**Responsibilities:**
- Centralized market data access
- Indicator calculation
- News aggregation
- Data caching

**Key Methods:**
```python
get_hourly_stock_data()      # Fetch hourly OHLCV data
get_latest_news()            # Aggregate news from sources
calculate_indicators()       # Compute technical indicators
fetch_realtime_quotes()      # Get live market data
```

**Related Files:**
- `data_flow/data_pipeline.py`: Data pipeline orchestration script

**Data Sources:**
- **TinySoft**: Primary market data provider
- **AKShare**: Fallback and historical data
- **EastMoney**: News and sentiment
- **Custom indicators**: RSI, MACD, Bollinger Bands

### 3. Agent Spawner

**Responsibilities:**
- Process creation and isolation
- Runtime environment setup
- Log file routing
- Process monitoring

**Process Isolation:**
```python
# Each agentic workflow runs in isolated subprocess
subprocess.Popen([
    sys.executable, 
    'main.py',
    'settings/default_config.json'  # or configs/default_config.json
], env={
    'ONLY_SIGNATURE': model_signature,
    'RUNTIME_ENV_PATH': f'settings/runtime/runtime_env_{signature}.json',  # or configs/runtime/
    **os.environ
})
```

**Note:** Configuration files can be located in either `settings/` (primary) or `configs/` (alternative location).

### 4. Trading Agents (`agent_engine/agent/agent.py`)

**Agent Lifecycle:**

```
┌─────────────┐
│ Initialize  │
│ - Load env  │
│ - Connect   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Prefetch  │◄──────┐
│ - Snapshot  │       │
│ - Validate  │       │
└──────┬──────┘       │
       │              │
       ▼              │
┌─────────────┐       │
│   Decision  │       │
│ - Analyze   │       │
│ - Trade     │       │
└──────┬──────┘       │
       │              │
       ▼              │
┌─────────────┐       │
│   Execute   │       │
│ - Validate  │       │
│ - Log       │       │
└──────┬──────┘       │
       │              │
       └──────────────┘
```

**Agent Components:**

1. **Portfolio Manager**
   - Position tracking
   - Risk calculation
   - Trade validation

2. **Decision Engine**
   - LLM prompt construction
   - Response parsing
   - Action extraction

3. **Tool Layer**
   - Market data queries
   - News retrieval
   - Mathematical calculations
   - Trade execution

## Data Flow

### Decision Window Data Flow

```
1. Decision Window Triggers (10:30 / 11:30 / 14:00)
   │
   ▼
2. Shared Prefetch Coordinator
   ├─ Acquire file lock
   ├─ Check if snapshot exists
   └─ If not, fetch data:
      ├─ Market quotes (TinySoft)
      ├─ News feed (multiple sources)
      ├─ Technical indicators
      └─ Deduplicate & aggregate
   │
   ▼
3. Write Snapshot to disk
   └─ data_flow/agent_data/shared/snapshots/<date>/<sanitized_time>_<symbols_signature>.json
   │
   ▼
4. Agents read snapshot
   ├─ Validate data completeness
   ├─ Construct context
   └─ If snapshot missing, fetch locally
   │
   ▼
5. LLM Decision Making
   ├─ Build prompt with:
   │  ├─ Market data
   │  ├─ News analysis
   │  ├─ Current portfolio
   │  ├─ Risk constraints
   │  └─ Technical indicators
   ├─ Call LLM API
   └─ Parse response
   │
   ▼
6. Trade Execution
   ├─ Validate action
   ├─ Check risk limits
   ├─ Update positions
   └─ Log decision
   │
   ▼
7. Persist State
   ├─ Position file (JSONL)
   └─ Decision log (JSONL)
```

### Snapshot Data Structure

```json
{
  "timestamp": "2025-11-17 10:30:00",
  "symbols_signature": "SH688008|SH688009|SH688111",
  "market_data": {
    "SH688008": {
      "current_price": 125.50,
      "open": 123.00,
      "high": 126.00,
      "low": 122.50,
      "volume": 1234567,
      "indicators": {
        "rsi": 65.5,
        "macd": {"macd": 1.2, "signal": 0.8, "histogram": 0.4},
        "bb_upper": 128.0,
        "bb_lower": 122.0
      }
    },
    ...
  },
  "news": [
    {
      "title": "Company X announces earnings",
      "content": "...",
      "timestamp": "2025-11-17 09:45:00",
      "source": "eastmoney",
      "sentiment": "positive"
    },
    ...
  ],
  "metadata": {
    "prefetch_duration_ms": 1234,
    "data_sources": ["tinysoft", "akshare"],
    "news_count": 15
  }
}
```

## Agent Architecture

### Base Agent Structure

```python
class AgenticWorkflow:
    def __init__(self, signature: str, config: Dict):
        self.signature = signature
        self.config = config
        self.portfolio = Portfolio()
        self.tools = ToolRegistry()
        self.llm_client = LLMClient(signature)
        
    def run_decision_window(self, window_time: str):
        """Execute single decision window."""
        # 1. Load snapshot
        snapshot = self.load_shared_snapshot(window_time)
        
        # 2. Build context
        context = self.build_decision_context(snapshot)
        
        # 3. Call LLM
        response = self.llm_client.generate(context)
        
        # 4. Parse & validate
        action = self.parse_action(response)
        
        # 5. Execute trade
        self.execute_action(action)
        
        # 6. Log decision
        self.log_decision(action, response)
```

### Tool System

Agents have access to a set of tools:

```python
AGENT_TOOLS = [
    {
        "name": "get_stock_price",
        "description": "Get current stock price and indicators",
        "parameters": {
            "symbol": "str",
            "indicators": "List[str]"
        }
    },
    {
        "name": "get_news",
        "description": "Retrieve recent news for stock",
        "parameters": {
            "symbol": "str",
            "hours": "int"
        }
    },
    {
        "name": "execute_trade",
        "description": "Execute buy/sell order",
        "parameters": {
            "symbol": "str",
            "action": "Literal['BUY', 'SELL']",
            "quantity": "int"
        }
    },
    {
        "name": "calculate",
        "description": "Perform calculations",
        "parameters": {
            "expression": "str"
        }
    }
]
```

## Shared Prefetch System

### Design Goals

1. **Fair Comparison**: All agents see identical data
2. **Cost Efficiency**: Single API call per window
3. **Consistency**: Atomic snapshot generation
4. **Fallback**: Local fetch if shared fails

### Implementation

```python
class SharedPrefetchCoordinator:
    def __init__(self, base_dir: Optional[str] = None, ttl_seconds: int = 600):
        self.base_dir = Path(base_dir) if base_dir else default_dir
        self.snapshots_dir = self.base_dir / "snapshots"
        self.logs_dir = self.base_dir / "logs"
        self.lock_dir = self.base_dir / "locks"
        
    def ensure_snapshot(
        self,
        today_date: str,
        current_time: str,
        symbols_signature: str,
        builder: Callable[[], Dict[str, Any]],
    ) -> SnapshotResult:
        """Load a fresh snapshot if it exists; otherwise build one using the builder callable."""
        existing = self._load_snapshot(today_date, current_time, symbols_signature)
        if existing:
            return existing
        
        with _FileLock(self._lock_path()):
            # Build and save snapshot using builder
            snapshot = builder()
            self._save_snapshot(snapshot)
            return SnapshotResult(snapshot, path, created=True)
```

### File Locking

Uses `fcntl` (Unix) or `msvcrt` (Windows) for cross-process synchronization:

```python
class _FileLock:
    def __enter__(self):
        self.fd = open(self.lock_file, 'w')
        if os.name == 'posix':
            import fcntl
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX)
        elif os.name == 'nt':
            import msvcrt
            msvcrt.locking(self.fd.fileno(), msvcrt.LK_LOCK, 1)
        return self
```

## API Layer

### FastAPI Application Structure

```python
app = FastAPI(
    title="AStock Arena API",
    version="1.0.0",
    description="Multi-agentic workflow LLM trading system"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(job_router, prefix="/api/job")
app.include_router(live_router, prefix="/api/live")
app.include_router(backup_router, prefix="/api")
```

### Job Management

```python
class JobManager:
    def __init__(self):
        self.jobs = {}  # job_id -> Job
        
    def create_job(self, config_path: str) -> str:
        """Create and start new trading job."""
        job_id = str(uuid.uuid4())
        process = self._spawn_process(config_path)
        self.jobs[job_id] = Job(job_id, process, "running")
        return job_id
        
    def get_status(self, job_id: str) -> Dict:
        """Get job status and logs."""
        job = self.jobs.get(job_id)
        if not job:
            return {"status": "not_found"}
        return {
            "status": "running" if job.process.poll() is None else "completed",
            "logs": self._read_logs(job)
        }
```

## Frontend Architecture

### React Component Hierarchy

```
App
├── MainNavigation
├── DashboardView
│   ├── TopBanners
│   ├── LeftStockSidebar
│   ├── CenterContent
│   │   ├── StockChart
│   │   ├── AIModelCard (multiple)
│   │   └── DecisionTimeline
│   ├── RightInfoPanel
│   │   ├── LiveNewsPanel
│   │   └── PositionsPanel
│   └── PortfolioAnalytics
└── GlobalContext
    ├── ModelDataProvider
    └── WebSocketProvider
```

### State Management

Using React Context for global state:

```typescript
interface ModelData {
  signature: string;
  positions: Position[];
  pnl: number;
  decisions: Decision[];
  lastUpdate: Date;
}

const ModelDataContext = createContext<{
  models: ModelData[];
  refreshData: () => Promise<void>;
  isLoading: boolean;
}>(null);
```

### Data Fetching Strategy

```typescript
// Polling with caching
const CACHE_DURATION = 45000; // 45 seconds

function useModelData() {
  const [data, setData] = useState<ModelData[]>([]);
  const [lastFetch, setLastFetch] = useState<Date>(null);
  
  const fetchData = async () => {
    const now = new Date();
    if (lastFetch && now.getTime() - lastFetch.getTime() < CACHE_DURATION) {
      return; // Use cached data
    }
    
    const responses = await Promise.all(
      MODELS.map(sig => 
        fetch(`/api/live/pnl-series?signature=${sig}`)
      )
    );
    
    const newData = await Promise.all(responses.map(r => r.json()));
    setData(newData);
    setLastFetch(now);
  };
  
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);
  
  return { data, refresh: fetchData };
}
```

## Database & Storage

### File-Based Storage

The system uses file-based storage for simplicity and transparency:

```
data_flow/
├── ai_stock_data.json           # Hourly OHLCV + indicators
├── news.csv                     # News feed
├── data_pipeline.py             # Data pipeline orchestration
├── pnl_snapshots/               # PnL snapshots (JSON files)
├── debug/                       # Debug logs and error tracking
├── agent_data/                  # Legacy per-agent state (for compatibility)
│   ├── <signature>/
│   │   ├── position/
│   │   │   └── position.jsonl   # Position history
│   │   └── log/
│   │       └── <date>/
│   │           └── <sanitized_time>.jsonl  # Decision logs (one file per decision)
│   └── shared/
│       ├── snapshots/
│       │   └── <date>/
│       │       └── <sanitized_time>_<symbols_signature>.json
│       ├── logs/
│       │   └── <date>/
│       │       └── <sanitized_time>.jsonl
│       └── locks/
│           └── <lock_files>
└── trading_summary_each_agent/  # Current per-agent data
    ├── <signature>/
    │   ├── position/
    │   │   └── position.jsonl   # Position history
    │   └── log/
    │       └── <date>/
    │           └── <sanitized_time>.jsonl
    └── shared/
        ├── snapshots/
        ├── logs/
        └── locks/
```

### Position Storage Format (JSONL)

```jsonl
{"id": 1, "date": "2025-11-17", "decision_time": "2025-11-17 10:30:00", "decision_count": 1, "positions": {"CASH": 95000.0, "SH688008": {"shares": 100, "avg_price": 125.50, "purchase_date": "2025-11-17"}}, "this_action": {"action": "BUY", "symbol": "SH688008", "amount": 100}}
{"id": 2, "date": "2025-11-17", "decision_time": "2025-11-17 11:30:00", "decision_count": 2, "positions": {"CASH": 101350.0, "SH688008": {"shares": 50, "avg_price": 125.50, "purchase_date": "2025-11-17"}}, "this_action": {"action": "SELL", "symbol": "SH688008", "amount": 50}}
```

### Backup System

Automated backups on every run:

```python
def create_backup(retain: int = 10):
    """Create timestamped backup with integrity checks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backups/{timestamp}"  # Note: backups/ (plural) is the primary location
    
    # Copy files
    shutil.copytree("data_flow", f"{backup_dir}/data_flow")
    shutil.copy("settings/default_config.json", backup_dir)
    # Also backup configs/ if it exists
    if os.path.exists("configs"):
        shutil.copytree("configs", f"{backup_dir}/configs")
    
    # Create tarball
    with tarfile.open(f"{backup_dir}/backup.tar.gz", "w:gz") as tar:
        tar.add(f"{backup_dir}/data_flow")
        tar.add(f"{backup_dir}/settings")
        if os.path.exists(f"{backup_dir}/configs"):
            tar.add(f"{backup_dir}/configs")
    
    # Generate manifest with checksums
    manifest = generate_sha256_manifest(backup_dir)
    
    # Cleanup old backups
    cleanup_old_backups(retain)
```

**Note:** Backups are stored in `backups/` (plural) directory with timestamped folders. The `backup/` (singular) directory may exist for legacy compatibility.

## Deployment Architecture

### Development Deployment

```
┌─────────────────┐     ┌─────────────────┐
│   Frontend      │     │   Backend       │
│   localhost:5173│────▶│   localhost:8000│
│   (npm run dev) │     │   (uvicorn)     │
└─────────────────┘     └────────┬────────┘
                                 │
                        ┌────────┴────────┐
                        │  Trading Agents │
                        │  (subprocesses) │
                        └─────────────────┘
```

### Production Deployment (Recommended)

```
┌──────────────┐
│   Nginx      │  (Reverse proxy)
│   Port 80    │
└──────┬───────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│   Frontend   │  │   Backend    │
│   (Static)   │  │   (Gunicorn) │
└──────────────┘  └──────┬───────┘
                         │
                  ┌──────┴──────┐
                  │   Agents    │
                  │ (systemd)   │
                  └─────────────┘
```

### Docker Deployment (Future)

```dockerfile
# Example Dockerfile structure
FROM python:3.10

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0"]
```

## Performance Considerations

### Bottlenecks

1. **LLM API Latency**: 2-8 seconds per decision
2. **Market Data Fetching**: TinySoft API rate limits
3. **Indicator Calculation**: ~40 bars per stock
4. **Frontend Polling**: 30-45 second intervals

### Optimizations

1. **Shared Prefetch**: Reduces redundant API calls by 80%
2. **Data Caching**: Frontend caches for 45 seconds
3. **Async Processing**: Backend uses async/await
4. **Process Isolation**: Parallel agent execution
5. **Indicator Windowing**: Limits context to prevent token overflow

### Scalability

- **Horizontal**: Add more agents (up to API rate limits)
- **Vertical**: Increase API concurrency limits
- **Caching**: Redis for shared snapshot distribution (future)
- **Database**: PostgreSQL for position history (future)

## Security Considerations

### API Keys

- Stored in environment variables
- Never committed to version control
- Loaded at runtime only

### Data Access

- File permissions restricted to user
- No external access to data files
- Backup encryption (optional, future)

### Rate Limiting

- Respect external API rate limits
- Implement backoff strategies
- Monitor API usage

---

This architecture is designed for research flexibility and operational reliability. For questions or suggestions, please open an issue on GitHub.
