import { useState, useEffect } from 'react';
import { LineChart, Line, CandlestickChart, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, ComposedChart } from 'recharts';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface StockChartProps {
  symbol: string;
  currentPrice: number;
  priceChange: number;
  percentChange: number;
}

export function StockChart({ symbol, currentPrice, priceChange, percentChange }: StockChartProps) {
  const [timeframe, setTimeframe] = useState<'1min' | '5min' | '1hr' | '1day'>('5min');
  const [chartType, setChartType] = useState<'line' | 'candle'>('line');
  const [showVolume, setShowVolume] = useState(true);
  const [chartData, setChartData] = useState<any[]>([]);

  useEffect(() => {
    // Generate mock historical data
    const generateData = () => {
      const data = [];
      const basePrice = currentPrice;
      const points = timeframe === '1min' ? 60 : timeframe === '5min' ? 78 : timeframe === '1hr' ? 48 : 30;
      
      for (let i = points; i >= 0; i--) {
        const variance = (Math.random() - 0.5) * (basePrice * 0.02);
        const price = basePrice + variance;
        const open = price + (Math.random() - 0.5) * 2;
        const close = price + (Math.random() - 0.5) * 2;
        const high = Math.max(open, close) + Math.random() * 1;
        const low = Math.min(open, close) - Math.random() * 1;
        
        data.push({
          time: `${i}m`,
          price: parseFloat(price.toFixed(2)),
          open: parseFloat(open.toFixed(2)),
          high: parseFloat(high.toFixed(2)),
          low: parseFloat(low.toFixed(2)),
          close: parseFloat(close.toFixed(2)),
          volume: Math.floor(Math.random() * 1000000) + 500000,
        });
      }
      return data;
    };

    setChartData(generateData());
    
    // Simulate real-time updates
    const interval = setInterval(() => {
      setChartData(prev => {
        const newData = [...prev.slice(1)];
        const lastPrice = newData[newData.length - 1].price;
        const newPrice = lastPrice + (Math.random() - 0.5) * 0.5;
        newData.push({
          time: 'now',
          price: parseFloat(newPrice.toFixed(2)),
          open: parseFloat((newPrice - 0.2).toFixed(2)),
          high: parseFloat((newPrice + 0.3).toFixed(2)),
          low: parseFloat((newPrice - 0.4).toFixed(2)),
          close: parseFloat(newPrice.toFixed(2)),
          volume: Math.floor(Math.random() * 1000000) + 500000,
        });
        return newData;
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [timeframe, currentPrice]);

  const timeframes = [
    { value: '1min', label: '1m' },
    { value: '5min', label: '5m' },
    { value: '1hr', label: '1h' },
    { value: '1day', label: '1d' },
  ] as const;

  return (
    <div className="glass-card rounded-lg p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-white mb-1">{symbol}</h3>
          <div className="flex items-center gap-3">
            <span className="text-2xl">${currentPrice.toFixed(2)}</span>
            <div className={`flex items-center gap-1 ${priceChange >= 0 ? 'price-up' : 'price-down'}`}>
              {priceChange >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
              <span>{priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}</span>
              <span>({percentChange >= 0 ? '+' : ''}{percentChange.toFixed(2)}%)</span>
            </div>
          </div>
        </div>
        
        {/* Controls */}
        <div className="flex items-center gap-2">
          <div className="flex gap-1 bg-[#0F1420] rounded p-1">
            {timeframes.map((tf) => (
              <button
                key={tf.value}
                onClick={() => setTimeframe(tf.value)}
                className={`px-3 py-1 rounded text-xs transition-colors ${
                  timeframe === tf.value 
                    ? 'bg-[#1CE479] text-[#0F1420]' 
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                {tf.label}
              </button>
            ))}
          </div>
          
          <div className="flex gap-1 bg-[#0F1420] rounded p-1">
            <button
              onClick={() => setChartType('line')}
              className={`px-3 py-1 rounded text-xs transition-colors ${
                chartType === 'line' 
                  ? 'bg-[#1CE479] text-[#0F1420]' 
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Line
            </button>
            <button
              onClick={() => setChartType('candle')}
              className={`px-3 py-1 rounded text-xs transition-colors ${
                chartType === 'candle' 
                  ? 'bg-[#1CE479] text-[#0F1420]' 
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Candle
            </button>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData}>
            <defs>
              <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#1CE479" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#1CE479" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis 
              dataKey="time" 
              stroke="rgba(255,255,255,0.5)"
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              stroke="rgba(255,255,255,0.5)"
              domain={['auto', 'auto']}
              tick={{ fontSize: 12 }}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1A1F2E', 
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px'
              }}
            />
            {chartType === 'line' ? (
              <>
                <Area 
                  type="monotone" 
                  dataKey="price" 
                  stroke="#1CE479" 
                  strokeWidth={2}
                  fill="url(#colorPrice)"
                />
              </>
            ) : (
              <Line 
                type="monotone" 
                dataKey="close" 
                stroke="#1CE479" 
                strokeWidth={2}
                dot={false}
              />
            )}
            {showVolume && (
              <Bar 
                dataKey="volume" 
                fill="rgba(0, 212, 255, 0.2)" 
                yAxisId="volume"
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Volume Toggle */}
      <div className="mt-2 flex items-center justify-between text-xs text-gray-400">
        <button
          onClick={() => setShowVolume(!showVolume)}
          className="hover:text-white transition-colors"
        >
          {showVolume ? 'Hide' : 'Show'} Volume
        </button>
        <div className="flex gap-4">
          <span>Bid: ${(currentPrice - 0.02).toFixed(2)}</span>
          <span>Ask: ${(currentPrice + 0.02).toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
}
