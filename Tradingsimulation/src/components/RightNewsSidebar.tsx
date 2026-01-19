import { useState, useEffect } from 'react';
import { Newspaper, TrendingUp, TrendingDown, Clock, AlertCircle } from 'lucide-react';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';

interface NewsItem {
  id: string;
  title: string;
  summary: string;
  timestamp: Date;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  impact: 'high' | 'medium' | 'low';
  relatedStocks: string[];
  source: string;
  priceImpact?: number;
}

export function RightNewsSidebar() {
  const [news, setNews] = useState<NewsItem[]>([
    {
      id: '1',
      title: 'Apple announces breakthrough in AI chip technology',
      summary: 'New M4 chip shows 40% performance improvement in AI workloads, stock surges in after-hours trading.',
      timestamp: new Date(Date.now() - 300000),
      sentiment: 'bullish',
      impact: 'high',
      relatedStocks: ['AAPL'],
      source: 'Bloomberg',
      priceImpact: 2.3,
    },
    {
      id: '2',
      title: 'Tesla production challenges persist at Berlin facility',
      summary: 'Manufacturing delays could impact Q4 delivery targets according to internal sources.',
      timestamp: new Date(Date.now() - 600000),
      sentiment: 'bearish',
      impact: 'medium',
      relatedStocks: ['TSLA'],
      source: 'Reuters',
      priceImpact: -1.5,
    },
    {
      id: '3',
      title: 'NVIDIA secures major cloud infrastructure deals',
      summary: 'Partnerships with AWS, Azure, and Google Cloud expected to drive $5B in revenue.',
      timestamp: new Date(Date.now() - 900000),
      sentiment: 'bullish',
      impact: 'high',
      relatedStocks: ['NVDA', 'MSFT', 'GOOGL'],
      source: 'WSJ',
      priceImpact: 3.1,
    },
    {
      id: '4',
      title: 'Fed signals steady interest rate policy',
      summary: 'Powell indicates no immediate rate changes, market reacts positively to stability.',
      timestamp: new Date(Date.now() - 1200000),
      sentiment: 'neutral',
      impact: 'high',
      relatedStocks: ['SPY', 'QQQ'],
      source: 'CNBC',
    },
    {
      id: '5',
      title: 'Microsoft AI revenue exceeds expectations',
      summary: 'Azure AI services grow 100% YoY, beating analyst estimates significantly.',
      timestamp: new Date(Date.now() - 1800000),
      sentiment: 'bullish',
      impact: 'high',
      relatedStocks: ['MSFT'],
      source: 'Bloomberg',
      priceImpact: 2.8,
    },
  ]);

  useEffect(() => {
    // Simulate new news arriving
    const interval = setInterval(() => {
      const headlines = [
        { 
          title: 'Breaking: Major tech earnings beat expectations',
          summary: 'Tech sector shows resilience with strong Q4 results across the board.',
          stocks: ['AAPL', 'MSFT', 'GOOGL'],
        },
        { 
          title: 'Economic data shows stronger than expected growth',
          summary: 'GDP growth revised upward to 3.2%, exceeding forecasts.',
          stocks: ['SPY', 'QQQ'],
        },
        { 
          title: 'Regulatory changes proposed for AI sector',
          summary: 'New framework could impact major tech companies\' AI development.',
          stocks: ['NVDA', 'MSFT'],
        },
        { 
          title: 'Market volatility spikes on geopolitical tensions',
          summary: 'VIX jumps 15% as uncertainty increases in global markets.',
          stocks: ['VIX', 'SPY'],
        },
      ];
      
      if (Math.random() > 0.7) {
        const headline = headlines[Math.floor(Math.random() * headlines.length)];
        const sentiments: ('bullish' | 'bearish' | 'neutral')[] = ['bullish', 'bearish', 'neutral'];
        const impacts: ('high' | 'medium' | 'low')[] = ['high', 'medium', 'low'];
        const sources = ['Bloomberg', 'Reuters', 'WSJ', 'CNBC', 'FT'];

        const newNews: NewsItem = {
          id: Date.now().toString(),
          title: headline.title,
          summary: headline.summary,
          timestamp: new Date(),
          sentiment: sentiments[Math.floor(Math.random() * sentiments.length)],
          impact: impacts[Math.floor(Math.random() * impacts.length)],
          relatedStocks: headline.stocks,
          source: sources[Math.floor(Math.random() * sources.length)],
          priceImpact: (Math.random() - 0.5) * 5,
        };

        setNews(prev => [newNews, ...prev].slice(0, 12));
      }
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  const getSentimentColor = (sentiment: NewsItem['sentiment']) => {
    switch (sentiment) {
      case 'bullish':
        return 'text-[#00C805]';
      case 'bearish':
        return 'text-[#FF2E2E]';
      case 'neutral':
        return 'text-gray-400';
    }
  };

  const getSentimentIcon = (sentiment: NewsItem['sentiment']) => {
    switch (sentiment) {
      case 'bullish':
        return <TrendingUp className="w-3 h-3" />;
      case 'bearish':
        return <TrendingDown className="w-3 h-3" />;
      case 'neutral':
        return <AlertCircle className="w-3 h-3" />;
    }
  };

  const getImpactColor = (impact: NewsItem['impact']) => {
    switch (impact) {
      case 'high':
        return 'bg-[#FF2E2E] text-white';
      case 'medium':
        return 'bg-[#FFB800] text-[#0F1420]';
      case 'low':
        return 'bg-gray-600 text-white';
    }
  };

  return (
    <div className="w-96 bg-[#0F1420] border-l border-gray-800 h-screen flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Newspaper className="w-5 h-5 text-[#1CE479]" />
            <h3 className="text-white">Market News</h3>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-[#1CE479] animate-pulse" />
            <span className="text-xs text-gray-400">Live</span>
          </div>
        </div>
        <p className="text-xs text-gray-400">Real-time news impacting your positions</p>
      </div>

      {/* News Feed */}
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-3">
          {news.map((item) => (
            <div
              key={item.id}
              className="glass-card rounded-lg p-3 hover:bg-[#1A1F2E] transition-all cursor-pointer border border-transparent hover:border-[#1CE479] group"
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-1.5">
                  <div className={getSentimentColor(item.sentiment)}>
                    {getSentimentIcon(item.sentiment)}
                  </div>
                  <Badge className={`${getImpactColor(item.impact)} text-xs`}>
                    {item.impact}
                  </Badge>
                </div>
                <div className="flex items-center gap-1 text-xs text-gray-500">
                  <Clock className="w-3 h-3" />
                  <span>{Math.floor((Date.now() - item.timestamp.getTime()) / 60000)}m</span>
                </div>
              </div>

              {/* Title */}
              <h4 className="text-sm text-white mb-2 leading-tight group-hover:text-[#1CE479] transition-colors">
                {item.title}
              </h4>

              {/* Summary */}
              <p className="text-xs text-gray-400 leading-relaxed mb-3">
                {item.summary}
              </p>

              {/* Related Stocks */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 flex-wrap">
                  {item.relatedStocks.map((stock) => (
                    <span
                      key={stock}
                      className="text-xs bg-[#1CE479] bg-opacity-10 text-[#1CE479] px-2 py-0.5 rounded hover:bg-opacity-20 transition-colors cursor-pointer"
                    >
                      {stock}
                    </span>
                  ))}
                </div>
                {item.priceImpact !== undefined && (
                  <span className={`text-xs ${item.priceImpact >= 0 ? 'price-up' : 'price-down'}`}>
                    {item.priceImpact >= 0 ? '+' : ''}{item.priceImpact.toFixed(1)}%
                  </span>
                )}
              </div>

              {/* Source */}
              <div className="mt-2 pt-2 border-t border-gray-800 flex items-center justify-between">
                <span className="text-xs text-gray-500">{item.source}</span>
                <div className={`text-xs ${getSentimentColor(item.sentiment)}`}>
                  {item.sentiment.toUpperCase()}
                </div>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>

      {/* Footer Stats */}
      <div className="p-4 border-t border-gray-800">
        <div className="glass-card rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-2">News Impact Score</div>
          <div className="flex items-center gap-2 mb-2">
            <div className="flex-1 h-2 bg-[#1A1F2E] rounded overflow-hidden">
              <div className="h-full bg-gradient-to-r from-[#1CE479] to-[#00C805] w-[73%] animate-pulse" />
            </div>
            <span className="text-white text-sm">73</span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="text-center">
              <div className="text-[#00C805]">{news.filter(n => n.sentiment === 'bullish').length}</div>
              <div className="text-gray-500">Bullish</div>
            </div>
            <div className="text-center">
              <div className="text-gray-400">{news.filter(n => n.sentiment === 'neutral').length}</div>
              <div className="text-gray-500">Neutral</div>
            </div>
            <div className="text-center">
              <div className="text-[#FF2E2E]">{news.filter(n => n.sentiment === 'bearish').length}</div>
              <div className="text-gray-500">Bearish</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
