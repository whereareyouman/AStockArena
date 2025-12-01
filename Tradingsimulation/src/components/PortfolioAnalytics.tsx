import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';
import { PieChart as PieChartIcon, History, AlertTriangle } from 'lucide-react';

export function PortfolioAnalytics() {
  const holdings = [
    { name: 'AAPL', value: 9120, color: '#00D4FF' },
    { name: 'TSLA', value: 5954, color: '#4361EE' },
    { name: 'NVDA', value: 15082, color: '#00C805' },
    { name: 'MSFT', value: 15368, color: '#FFB800' },
    { name: 'Cash', value: 4476, color: '#6B7280' },
  ];

  const sectors = [
    { name: 'Technology', value: 65, color: '#00D4FF' },
    { name: 'Automotive', value: 12, color: '#4361EE' },
    { name: 'Semiconductors', value: 18, color: '#00C805' },
    { name: 'Cash', value: 5, color: '#6B7280' },
  ];

  const tradeHistory = [
    {
      time: '14:32',
      symbol: 'AAPL',
      action: 'BUY',
      quantity: 50,
      price: 182.40,
      reasoning: 'Strong momentum + AI chip announcement',
      outcome: 'In Progress',
    },
    {
      time: '13:15',
      symbol: 'TSLA',
      action: 'SELL',
      quantity: 25,
      price: 238.15,
      reasoning: 'Stop-loss triggered at -2% threshold',
      outcome: '+$156.25',
    },
    {
      time: '11:47',
      symbol: 'NVDA',
      action: 'BUY',
      quantity: 30,
      price: 502.75,
      reasoning: 'Breakout confirmation + high volume',
      outcome: 'In Progress',
    },
    {
      time: '10:22',
      symbol: 'MSFT',
      action: 'BUY',
      quantity: 40,
      price: 384.20,
      reasoning: 'Earnings beat + bullish sentiment',
      outcome: 'In Progress',
    },
  ];

  const diversificationScore = 72;
  const concentrationWarning = holdings[0].value / holdings.reduce((sum, h) => sum + h.value, 0) > 0.3;

  return (
    <div className="space-y-4">
      {/* Holdings Breakdown */}
      <div className="glass-card rounded-lg p-4">
        <div className="flex items-center gap-2 mb-4">
          <PieChartIcon className="w-5 h-5 text-[#00D4FF]" />
          <h3 className="text-white">Portfolio Breakdown</h3>
        </div>

        <div className="grid grid-cols-2 gap-4">
          {/* Asset Allocation */}
          <div>
            <h4 className="text-sm text-gray-400 mb-3">Asset Allocation</h4>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={holdings}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {holdings.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1A1F2E', 
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px'
                  }}
                  formatter={(value: number) => `$${value.toFixed(2)}`}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="space-y-1 mt-3">
              {holdings.map((item) => (
                <div key={item.name} className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="text-white">{item.name}</span>
                  </div>
                  <span className="text-gray-400">${item.value.toFixed(0)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Sector Exposure */}
          <div>
            <h4 className="text-sm text-gray-400 mb-3">Sector Exposure</h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={sectors} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis type="number" stroke="rgba(255,255,255,0.5)" tick={{ fontSize: 10 }} />
                <YAxis 
                  type="category" 
                  dataKey="name" 
                  stroke="rgba(255,255,255,0.5)"
                  tick={{ fontSize: 10 }}
                  width={100}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1A1F2E', 
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px'
                  }}
                  formatter={(value: number) => `${value}%`}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {sectors.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Warnings & Score */}
        <div className="mt-4 pt-4 border-t border-gray-800 grid grid-cols-2 gap-4">
          <div className="bg-[#0F1420] rounded p-3">
            <div className="text-xs text-gray-400 mb-1">Diversification Score</div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-2 bg-[#1A1F2E] rounded overflow-hidden">
                <div 
                  className={`h-full transition-all ${
                    diversificationScore >= 70 ? 'bg-[#00C805]' : 'bg-[#FFB800]'
                  }`}
                  style={{ width: `${diversificationScore}%` }}
                />
              </div>
              <span className="text-white">{diversificationScore}</span>
            </div>
          </div>
          
          {concentrationWarning && (
            <div className="bg-[#FFB800] bg-opacity-10 border border-[#FFB800] rounded p-3">
              <div className="flex items-center gap-2 text-[#FFB800] text-xs">
                <AlertTriangle className="w-4 h-4" />
                <span>High concentration detected</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Trade History Timeline */}
      <div className="glass-card rounded-lg p-4">
        <div className="flex items-center gap-2 mb-4">
          <History className="w-5 h-5 text-[#00D4FF]" />
          <h3 className="text-white">Trade History</h3>
        </div>

        <div className="space-y-3">
          {tradeHistory.map((trade, idx) => (
            <div 
              key={idx}
              className="bg-[#0F1420] rounded-lg p-3 hover:bg-[#1A1F2E] transition-colors"
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-3">
                  <div className="flex flex-col items-center">
                    <div className="text-xs text-gray-400">{trade.time}</div>
                    <div className="w-px h-8 bg-gray-700 my-1" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        trade.action === 'BUY' 
                          ? 'bg-[#00C805] text-white' 
                          : 'bg-[#FF2E2E] text-white'
                      }`}>
                        {trade.action}
                      </span>
                      <span className="text-white">{trade.symbol}</span>
                      <span className="text-xs text-gray-400">
                        {trade.quantity} @ ${trade.price}
                      </span>
                    </div>
                    <p className="text-xs text-gray-400 leading-relaxed max-w-md">
                      {trade.reasoning}
                    </p>
                  </div>
                </div>
                
                <div className={`text-xs px-2 py-1 rounded ${
                  trade.outcome === 'In Progress' 
                    ? 'bg-[#FFB800] bg-opacity-10 text-[#FFB800]' 
                    : trade.outcome.startsWith('+')
                    ? 'bg-[#00C805] bg-opacity-10 text-[#00C805]'
                    : 'bg-[#FF2E2E] bg-opacity-10 text-[#FF2E2E]'
                }`}>
                  {trade.outcome}
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 text-center">
          <button className="text-xs text-[#00D4FF] hover:underline">
            View Full History →
          </button>
        </div>
      </div>
    </div>
  );
}
