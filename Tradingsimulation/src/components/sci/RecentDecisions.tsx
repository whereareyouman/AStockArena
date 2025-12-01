import { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Minus, Clock } from 'lucide-react';

interface RecentDecisionsProps {
  signature: string;
}

interface Decision {
  date: string;
  time: string;
  count: number;
  action: string;
  symbol: string;
  amount: number;
  cash: number;
  holdings: number;
  id: number;
}

export function RecentDecisions({ signature }: RecentDecisionsProps) {
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!signature) {
      return;
    }

    const fetchDecisions = async () => {
      try {
        const res = await fetch(
          `http://localhost:8000/api/live/recent-decisions?signature=${signature}&limit=10`
        );
        if (!res.ok) throw new Error('failed');
        const data = await res.json();
        setDecisions(data.decisions || []);
      } catch (err) {
        console.warn('Failed to fetch recent decisions:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchDecisions();
    const interval = setInterval(fetchDecisions, 30_000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [signature]);

  const getActionIcon = (action: string) => {
    switch (action) {
      case 'buy':
        return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'sell':
        return <TrendingDown className="w-4 h-4 text-red-500" />;
      default:
        return <Minus className="w-4 h-4 text-gray-500" />;
    }
  };

  const getActionColor = (action: string) => {
    switch (action) {
      case 'buy':
        return 'text-green-500';
      case 'sell':
        return 'text-red-500';
      default:
        return 'text-gray-400';
    }
  };

  const getActionText = (action: string) => {
    switch (action) {
      case 'buy':
        return '买入';
      case 'sell':
        return '卖出';
      default:
        return '观望';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-32">
        <div className="text-gray-400 text-sm">加载中...</div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {decisions.length === 0 ? (
        <div className="text-center text-gray-400 text-sm py-8">
          暂无决策记录
        </div>
      ) : (
        decisions.map((decision, idx) => (
          <div
            key={`${decision.id}-${idx}`}
            className="glass-card rounded-lg p-3 hover:bg-gray-800/50 transition-colors"
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-2">
                {getActionIcon(decision.action)}
                <span className={`font-medium ${getActionColor(decision.action)}`}>
                  {getActionText(decision.action)}
                </span>
                {decision.symbol && (
                  <span className="text-white text-sm">
                    {decision.symbol}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1 text-xs text-gray-400">
                <Clock className="w-3 h-3" />
                {decision.time?.split(' ')[1] || ''}
              </div>
            </div>

            {decision.action !== 'no_trade' && decision.amount && (
              <div className="text-sm text-gray-300 mb-2">
                数量: {decision.amount} 股
              </div>
            )}

            <div className="flex items-center justify-between text-xs text-gray-400">
              <span>{decision.date}</span>
              <span>持仓: {decision.holdings} | 现金: ¥{decision.cash.toFixed(0)}</span>
            </div>
          </div>
        ))
      )}
    </div>
  );
}
