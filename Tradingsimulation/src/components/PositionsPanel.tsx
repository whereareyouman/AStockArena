import { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface Position {
  symbol: string;
  shares: number;
  purchase_date: string;
  entry_price: number;
  current_price: number;
  market_value: number;
  pnl: number;
  pnl_percent: number;
}

export function PositionsPanel() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [cash, setCash] = useState<number>(0);
  const [totalEquity, setTotalEquity] = useState<number>(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPositions = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/live/current-positions?signature=gemini-2.5-flash');
        if (!res.ok) throw new Error('failed');
        const data = await res.json();
        setPositions(data.positions || []);
        setCash(data.cash || 0);
        setTotalEquity(data.total_equity || 0);
      } catch (err) {
        console.warn('Failed to fetch positions:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchPositions();
    // Refresh every 30 seconds
    const interval = setInterval(fetchPositions, 30_000);
    return () => clearInterval(interval);
  }, []);

  const totalPnL = positions.reduce((sum, pos) => sum + pos.pnl, 0);

  if (loading) {
    return (
      <div className="glass-card rounded-lg p-4">
        <div className="text-center text-gray-400">加载持仓中...</div>
      </div>
    );
  }

  return (
    <div className="glass-card rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-white">当前持仓</h3>
        <div className={`flex items-center gap-2 ${totalPnL >= 0 ? 'price-up' : 'price-down'}`}>
          {totalPnL >= 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
          <span className="text-xl">¥{Math.abs(totalPnL).toFixed(2)}</span>
        </div>
      </div>

      {positions.length === 0 ? (
        <div className="text-center text-gray-400 py-8">
          暂无持仓
        </div>
      ) : (
        <div className="space-y-3">
          {positions.map((position) => (
            <div 
              key={position.symbol} 
              className="bg-[#0F1420] rounded-lg p-3 hover:bg-[#1A1F2E] transition-colors cursor-pointer"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#00D4FF] to-[#4361EE] flex items-center justify-center">
                    <span className="text-xs">{position.symbol.slice(0, 2)}</span>
                  </div>
                  <div>
                    <div className="text-white">{position.symbol}</div>
                    <div className="text-xs text-gray-400">{position.shares} 股</div>
                  </div>
                </div>
                
                <div className={`text-right ${position.pnl >= 0 ? 'price-up' : 'price-down'}`}>
                  <div className="flex items-center gap-1 justify-end">
                    {position.pnl >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                    <span>{position.pnl >= 0 ? '+' : ''}¥{position.pnl.toFixed(2)}</span>
                  </div>
                  <div className="text-xs">
                    {position.pnl_percent >= 0 ? '+' : ''}{position.pnl_percent.toFixed(2)}%
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between text-xs text-gray-400 mt-2">
                <div>
                  <span>成本: ¥{position.entry_price.toFixed(2)}</span>
                </div>
                <div>
                  <span>现价: ¥{position.current_price.toFixed(2)}</span>
                </div>
              </div>

              {/* Position size visualization */}
              <div className="mt-2 h-1 bg-[#1A1F2E] rounded overflow-hidden">
                <div 
                  className={`h-full transition-all duration-300 ${
                    position.pnl >= 0 ? 'bg-[#00C805]' : 'bg-[#FF2E2E]'
                  }`}
                  style={{ width: `${Math.min(Math.abs(position.pnl_percent) * 10, 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Cash Balance */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-400">现金余额</span>
          <span className="text-white">¥{cash.toFixed(2)}</span>
        </div>
        <div className="flex items-center justify-between text-sm mt-2">
          <span className="text-gray-400">总权益</span>
          <span className="text-white font-medium">¥{totalEquity.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
}
