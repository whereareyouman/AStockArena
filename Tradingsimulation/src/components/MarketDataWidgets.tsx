import { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Activity } from 'lucide-react';

interface MarketIndex {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
}

interface SectorData {
  name: string;
  change: number;
}

export function MarketDataWidgets() {
  const [indices, setIndices] = useState<MarketIndex[]>([
    { symbol: 'SPY', name: 'S&P 500', price: 452.80, change: 2.45, changePercent: 0.54 },
    { symbol: 'QQQ', name: 'Nasdaq', price: 387.20, change: -1.30, changePercent: -0.33 },
    { symbol: 'DIA', name: 'Dow Jones', price: 356.45, change: 0.85, changePercent: 0.24 },
  ]);

  const [sectors, setSectors] = useState<SectorData[]>([
    { name: 'Technology', change: 1.2 },
    { name: 'Healthcare', change: -0.5 },
    { name: 'Finance', change: 0.8 },
    { name: 'Energy', change: -1.3 },
    { name: 'Consumer', change: 0.4 },
    { name: 'Industrials', change: 0.6 },
  ]);

  const [vix, setVix] = useState(15.42);
  const [mostActive, setMostActive] = useState([
    { symbol: 'AAPL', volume: '125.4M', change: 2.3 },
    { symbol: 'TSLA', volume: '98.2M', change: -1.9 },
    { symbol: 'NVDA', volume: '87.6M', change: 3.1 },
    { symbol: 'MSFT', volume: '76.3M', change: 1.4 },
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      // Update indices
      setIndices(prev => prev.map(index => {
        const change = (Math.random() - 0.5) * 0.5;
        const newPrice = index.price + change;
        const newChange = index.change + change;
        const newChangePercent = (newChange / (newPrice - newChange)) * 100;
        
        return {
          ...index,
          price: parseFloat(newPrice.toFixed(2)),
          change: parseFloat(newChange.toFixed(2)),
          changePercent: parseFloat(newChangePercent.toFixed(2)),
        };
      }));

      // Update VIX
      setVix(prev => parseFloat((prev + (Math.random() - 0.5) * 0.2).toFixed(2)));
    }, 4000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-4">
      {/* Major Indices */}
      <div className="glass-card rounded-lg p-4">
        <h3 className="text-white mb-3">Major Indices</h3>
        <div className="grid grid-cols-3 gap-3">
          {indices.map((index) => (
            <div key={index.symbol} className="bg-[#0F1420] rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">{index.symbol}</div>
              <div className="text-white mb-1">${index.price.toFixed(2)}</div>
              <div className={`text-xs flex items-center gap-1 ${
                index.change >= 0 ? 'price-up' : 'price-down'
              }`}>
                {index.change >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                <span>{index.change >= 0 ? '+' : ''}{index.change.toFixed(2)}</span>
                <span>({index.changePercent >= 0 ? '+' : ''}{index.changePercent.toFixed(2)}%)</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Sector Performance Heatmap */}
      <div className="glass-card rounded-lg p-4">
        <h3 className="text-white mb-3">Sector Performance</h3>
        <div className="grid grid-cols-3 gap-2">
          {sectors.map((sector) => (
            <div 
              key={sector.name}
              className={`rounded-lg p-3 text-center transition-all ${
                sector.change >= 0 
                  ? 'bg-[#00C805] bg-opacity-20 hover:bg-opacity-30' 
                  : 'bg-[#FF2E2E] bg-opacity-20 hover:bg-opacity-30'
              }`}
            >
              <div className="text-xs text-white mb-1">{sector.name}</div>
              <div className={`${sector.change >= 0 ? 'price-up' : 'price-down'}`}>
                {sector.change >= 0 ? '+' : ''}{sector.change.toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Most Active & VIX */}
      <div className="grid grid-cols-2 gap-4">
        {/* Most Active */}
        <div className="glass-card rounded-lg p-4">
          <h3 className="text-white mb-3">Most Active</h3>
          <div className="space-y-2">
            {mostActive.map((stock) => (
              <div key={stock.symbol} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <span className="text-white">{stock.symbol}</span>
                  <span className="text-gray-400">{stock.volume}</span>
                </div>
                <span className={stock.change >= 0 ? 'price-up' : 'price-down'}>
                  {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* VIX */}
        <div className="glass-card rounded-lg p-4">
          <h3 className="text-white mb-3 flex items-center gap-2">
            <Activity className="w-4 h-4" />
            VIX Index
          </h3>
          <div className="text-center">
            <div className="text-3xl text-[#FFB800] mb-2">{vix.toFixed(2)}</div>
            <div className="text-xs text-gray-400">Volatility</div>
            <div className={`mt-2 text-xs ${
              vix < 15 ? 'text-[#00C805]' : vix < 20 ? 'text-[#FFB800]' : 'text-[#FF2E2E]'
            }`}>
              {vix < 15 ? 'Low' : vix < 20 ? 'Moderate' : 'High'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
