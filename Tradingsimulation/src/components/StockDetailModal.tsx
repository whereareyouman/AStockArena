import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { StockChart } from './StockChart';
import { Badge } from './ui/badge';
import { TrendingUp, TrendingDown, Activity, DollarSign, Users, Calendar } from 'lucide-react';

interface StockDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  symbol: string;
  currentPrice: number;
  priceChange: number;
  percentChange: number;
}

export function StockDetailModal({ 
  isOpen, 
  onClose, 
  symbol, 
  currentPrice, 
  priceChange, 
  percentChange 
}: StockDetailModalProps) {
  const fundamentals = {
    marketCap: '2.89T',
    peRatio: '28.5',
    dividendYield: '0.52%',
    eps: '6.42',
    revenue: '383.3B',
    volume: '52.3M',
  };

  const technicals = {
    rsi: 64.5,
    macd: 'Bullish',
    ma50: currentPrice - 5.2,
    ma200: currentPrice - 12.8,
    support: currentPrice - 8.5,
    resistance: currentPrice + 6.3,
  };

  const aiInsights = [
    {
      type: 'Pattern Recognition',
      insight: 'Ascending triangle formation detected with 78% historical breakout probability',
      confidence: 85,
    },
    {
      type: 'Volume Analysis',
      insight: 'Above-average volume confirms strong institutional interest',
      confidence: 72,
    },
    {
      type: 'Sentiment Analysis',
      insight: 'Social media sentiment positive (+0.68), news sentiment bullish',
      confidence: 91,
    },
  ];

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-y-auto bg-[#0F1420] border-gray-800">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-2xl text-white">{symbol}</span>
              <Badge className="bg-[#1CE479] text-[#0F1420]">Live Analysis</Badge>
            </div>
            <div className="text-right">
              <div className="text-2xl text-white">${currentPrice.toFixed(2)}</div>
              <div className={`text-sm flex items-center gap-1 ${priceChange >= 0 ? 'price-up' : 'price-down'}`}>
                {priceChange >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                <span>{priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}</span>
                <span>({percentChange >= 0 ? '+' : ''}{percentChange.toFixed(2)}%)</span>
              </div>
            </div>
          </DialogTitle>
        </DialogHeader>

        <Tabs defaultValue="chart" className="mt-4">
          <TabsList className="glass-card">
            <TabsTrigger value="chart">Chart</TabsTrigger>
            <TabsTrigger value="fundamentals">Fundamentals</TabsTrigger>
            <TabsTrigger value="technicals">Technicals</TabsTrigger>
            <TabsTrigger value="ai-insights">AI Insights</TabsTrigger>
          </TabsList>

          <TabsContent value="chart" className="mt-4">
            <StockChart
              symbol={symbol}
              currentPrice={currentPrice}
              priceChange={priceChange}
              percentChange={percentChange}
            />
          </TabsContent>

          <TabsContent value="fundamentals" className="mt-4">
            <div className="glass-card rounded-lg p-6">
              <h3 className="text-white mb-4">Fundamental Data</h3>
              <div className="grid grid-cols-3 gap-6">
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <DollarSign className="w-4 h-4 text-[#1CE479]" />
                      <span className="text-sm text-gray-400">Market Cap</span>
                    </div>
                    <div className="text-xl text-white">${fundamentals.marketCap}</div>
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="w-4 h-4 text-[#1CE479]" />
                      <span className="text-sm text-gray-400">P/E Ratio</span>
                    </div>
                    <div className="text-xl text-white">{fundamentals.peRatio}</div>
                  </div>
                </div>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <TrendingUp className="w-4 h-4 text-[#1CE479]" />
                      <span className="text-sm text-gray-400">Dividend Yield</span>
                    </div>
                    <div className="text-xl text-white">{fundamentals.dividendYield}</div>
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <DollarSign className="w-4 h-4 text-[#1CE479]" />
                      <span className="text-sm text-gray-400">EPS</span>
                    </div>
                    <div className="text-xl text-white">${fundamentals.eps}</div>
                  </div>
                </div>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <DollarSign className="w-4 h-4 text-[#1CE479]" />
                      <span className="text-sm text-gray-400">Revenue (TTM)</span>
                    </div>
                    <div className="text-xl text-white">${fundamentals.revenue}</div>
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <Users className="w-4 h-4 text-[#1CE479]" />
                      <span className="text-sm text-gray-400">Volume</span>
                    </div>
                    <div className="text-xl text-white">{fundamentals.volume}</div>
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <Calendar className="w-4 h-4 text-[#1CE479]" />
                  <span className="text-sm text-gray-400">Upcoming Events</span>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between bg-[#1A1F2E] rounded p-2">
                    <span className="text-white">Earnings Report</span>
                    <span className="text-gray-400">Jan 25, 2025</span>
                  </div>
                  <div className="flex items-center justify-between bg-[#1A1F2E] rounded p-2">
                    <span className="text-white">Dividend Date</span>
                    <span className="text-gray-400">Feb 8, 2025</span>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="technicals" className="mt-4">
            <div className="glass-card rounded-lg p-6">
              <h3 className="text-white mb-4">Technical Indicators</h3>
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="bg-[#1A1F2E] rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-2">RSI (14)</div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-2xl text-white">{technicals.rsi}</span>
                      <Badge className="bg-[#1CE479] text-[#0F1420]">Neutral</Badge>
                    </div>
                    <div className="h-2 bg-[#0F1420] rounded overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-[#00C805] via-[#FFB800] to-[#FF2E2E]"
                        style={{ width: '100%' }}
                      />
                      <div 
                        className="h-full w-1 bg-white -mt-2 transition-all"
                        style={{ marginLeft: `${technicals.rsi}%` }}
                      />
                    </div>
                  </div>

                  <div className="bg-[#1A1F2E] rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-2">MACD</div>
                    <div className="flex items-center gap-2">
                      <TrendingUp className="w-5 h-5 text-[#00C805]" />
                      <span className="text-xl text-white">{technicals.macd}</span>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="bg-[#1A1F2E] rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-3">Moving Averages</div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-white">50-day MA</span>
                        <span className="text-[#1CE479]">${technicals.ma50.toFixed(2)}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-white">200-day MA</span>
                        <span className="text-[#1CE479]">${technicals.ma200.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-[#1A1F2E] rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-3">Support & Resistance</div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-white">Resistance</span>
                        <span className="text-[#FF2E2E]">${technicals.resistance.toFixed(2)}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-white">Support</span>
                        <span className="text-[#00C805]">${technicals.support.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="ai-insights" className="mt-4">
            <div className="glass-card rounded-lg p-6">
              <h3 className="text-white mb-4">AI-Powered Analysis</h3>
              <div className="space-y-4">
                {aiInsights.map((insight, idx) => (
                  <div key={idx} className="bg-[#1A1F2E] rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <Badge className="bg-[#1CE479] text-[#0F1420]">{insight.type}</Badge>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-400">Confidence</span>
                        <span className="text-sm text-[#1CE479]">{insight.confidence}%</span>
                      </div>
                    </div>
                    <p className="text-white mb-2">{insight.insight}</p>
                    <div className="h-1.5 bg-[#0F1420] rounded overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-[#1CE479] to-[#00C805]"
                        style={{ width: `${insight.confidence}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6 pt-6 border-t border-gray-800">
                <div className="bg-[#1CE479] bg-opacity-10 border border-[#1CE479] rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Activity className="w-5 h-5 text-[#1CE479]" />
                    <span className="text-white">AI Recommendation</span>
                  </div>
                  <p className="text-sm text-gray-300 mb-2">
                    Based on current market conditions and historical patterns, AI suggests a <span className="text-[#1CE479]">BUY</span> position with moderate risk.
                  </p>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-400">Target Price</span>
                    <span className="text-[#1CE479]">${(currentPrice + 8.5).toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
