import { useState } from 'react';
import { Play, Loader2, CheckCircle, XCircle, Clock } from 'lucide-react';

interface TradingJob {
  job_id: string;
  pid: number;
  started_at: string;
  status?: 'running' | 'finished' | 'failed';
  log_tail?: string;
}

export function TradingControl() {
  const [job, setJob] = useState<TradingJob | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startTrading = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Start trading job
      const res = await fetch('http://localhost:8000/api/run-trading', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_path: 'configs/default_config.json' })
      });
      
      if (!res.ok) {
        throw new Error(`Failed to start: ${res.statusText}`);
      }
      
      const data = await res.json();
      setJob(data);
      
      // Poll for status updates
      pollJobStatus(data.job_id);
    } catch (err: any) {
      setError(err.message || 'Failed to start trading');
      setLoading(false);
    }
  };

  const pollJobStatus = async (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/job/${jobId}`);
        if (!res.ok) {
          if (res.status === 404) {
            clearInterval(interval);
            setError('Job not found (may have expired)');
            setLoading(false);
            return;
          }
          // For other errors (like 500), just log and continue polling
          console.warn(`Job status check failed: ${res.status}`);
          return;
        }
        
        const data = await res.json();
        setJob(prev => ({ ...prev!, ...data }));
        
        if (data.status === 'finished' || data.status === 'failed') {
          clearInterval(interval);
          setLoading(false);
        }
      } catch (err) {
        console.warn('Failed to poll job status:', err);
        // Don't stop polling on transient errors
      }
    }, 2000);
    
    // Cleanup: stop polling after 5 minutes regardless
    setTimeout(() => clearInterval(interval), 5 * 60 * 1000);
  };

  const getStatusIcon = () => {
    if (!job) return null;
    switch (job.status) {
      case 'running':
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
      case 'finished':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Clock className="w-5 h-5 text-gray-500" />;
    }
  };

  return (
    <div className="glass-card rounded-lg p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-white text-xl mb-1">LLM 决策控制</h2>
          <p className="text-sm text-gray-400">启动多模型并行决策（配置文件同后端）</p>
        </div>
        
        <button
          onClick={startTrading}
          disabled={loading}
          className={`
            flex items-center gap-2 px-6 py-3 rounded-lg font-medium
            transition-all duration-200
            ${loading 
              ? 'bg-gray-700 text-gray-400 cursor-not-allowed' 
              : 'bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700 hover:shadow-lg'
            }
          `}
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              运行中...
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              开始决策
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {job && (
        <div className="space-y-3">
          <div className="flex items-center gap-3 p-4 bg-gray-800/50 rounded-lg">
            {getStatusIcon()}
            <div className="flex-1">
              <div className="text-white text-sm font-medium">
                Job ID: {job.job_id?.slice(0, 8)}...
              </div>
              <div className="text-gray-400 text-xs">
                PID: {job.pid} | Started: {new Date(job.started_at).toLocaleTimeString('zh-CN')}
              </div>
            </div>
            <div className={`
              px-3 py-1 rounded-full text-xs font-medium
              ${job.status === 'running' ? 'bg-blue-500/20 text-blue-400' : ''}
              ${job.status === 'finished' ? 'bg-green-500/20 text-green-400' : ''}
              ${job.status === 'failed' ? 'bg-red-500/20 text-red-400' : ''}
            `}>
              {job.status === 'running' && '运行中'}
              {job.status === 'finished' && '完成'}
              {job.status === 'failed' && '失败'}
            </div>
          </div>

          {job.log_tail && (
            <div className="bg-gray-900/50 rounded-lg p-4 font-mono text-xs text-gray-300 overflow-auto max-h-48">
              <pre className="whitespace-pre-wrap">{job.log_tail}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
