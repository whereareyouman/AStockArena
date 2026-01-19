import { useState, useEffect } from 'react';
import { Newspaper, TrendingUp, TrendingDown, Minus, AlertCircle } from 'lucide-react';
import { Badge } from './ui/badge';

interface NewsItem {
  id: string;
  title: string;
  timestamp: Date;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  impact: 'high' | 'medium' | 'low';
  relatedStocks: string[];
  source: string;
}

export function NewsAnalysisPanel() {
  const [news, setNews] = useState<NewsItem[]>([
    {
      id: '1',
      title: 'Apple announces breakthrough in AI chip technology, stock surges',
      timestamp: new Date(Date.now() - 300000),
      sentiment: 'bullish',
      impact: 'high',
      relatedStocks: ['AAPL'],
      source: 'Bloomberg',
    },
    {
      id: '2',
      title: 'Tesla faces production challenges in new Gigafactory expansion',
      timestamp: new Date(Date.now() - 600000),
      sentiment: 'bearish',
      impact: 'medium',
      relatedStocks: ['TSLA'],
      source: 'Reuters',
    },
    {
      id: '3',
      title: 'NVIDIA partners with major cloud providers for AI infrastructure',
      timestamp: new Date(Date.now() - 900000),
      sentiment: 'bullish',
      impact: 'high',
      relatedStocks: ['NVDA', 'MSFT', 'GOOGL'],
      source: 'WSJ',
    },
    {
      id: '4',
      title: 'Fed signals potential interest rate adjustment in next quarter',
      timestamp: new Date(Date.now() - 1200000),
      sentiment: 'neutral',
      impact: 'high',
      relatedStocks: ['SPY', 'QQQ'],
      source: 'CNBC',
    },
  ]);

  const [aiInsights, setAiInsights] = useState([
    {
      stock: 'AAPL',
      insight: 'Historical pattern shows 78% probability of continued upward movement following similar chip announcements.',
      confidence: 85,
    },
    {
      stock: 'TSLA',
      insight: 'Production news typically creates 2-3 day volatility window. Consider short-term position adjustments.',
      confidence: 72,
    },
  ]);

  useEffect(() => {
    // Simulate new news arriving
    const interval = setInterval(() => {
      const headlines = [
        'Breaking: Major tech earnings beat expectations',
        'Economic data shows stronger than expected growth',
        'Regulatory changes proposed for fintech sector',
        'Market volatility increases amid geopolitical tensions',
        'Industry leaders announce strategic partnership',
      ];
      
      const sources = ['Bloomberg', 'Reuters', 'WSJ', 'CNBC', 'FT'];
      const sentiments: ('bullish' | 'bearish' | 'neutral')[] = ['bullish', 'bearish', 'neutral'];
      const impacts: ('high' | 'medium' | 'low')[] = ['high', 'medium', 'low'];
      const stocks = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN'];

      if (Math.random() > 0.8) {
        const newNews: NewsItem = {
          id: Date.now().toString(),
          title: headlines[Math.floor(Math.random() * headlines.length)],
          timestamp: new Date(),
          sentiment: sentiments[Math.floor(Math.random() * sentiments.length)],
          impact: impacts[Math.floor(Math.random() * impacts.length)],
          relatedStocks: [stocks[Math.floor(Math.random() * stocks.length)]],
          source: sources[Math.floor(Math.random() * sources.length)],
        };

        setNews(prev => [newNews, ...prev].slice(0, 8));
      }
    }, 8000);

    return () => clearInterval(interval);
  }, []);

  const getSentimentIcon = (sentiment: NewsItem['sentiment']) => {
    switch (sentiment) {
      case 'bullish':
        return <TrendingUp className="w-4 h-4 text-[#00C805]" />;
      case 'bearish':
        return <TrendingDown className="w-4 h-4 text-[#FF2E2E]" />;
      case 'neutral':
        return <Minus className="w-4 h-4 text-gray-400" />;
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

  return (
    <div className="glass-card rounded-lg p-4 h-full flex flex-col">
      <div className="flex items-center gap-2 mb-4">
        <Newspaper className="w-5 h-5 text-[#00D4FF]" />
        <h3 className="text-white">Live News & Sentiment</h3>
      </div>

      <div className="flex-1 overflow-auto space-y-3 pr-2">
        {news.map((item) => (
          <div 
            key={item.id}
            className="bg-[#0F1420] rounded-lg p-3 hover:bg-[#1A1F2E] transition-all cursor-pointer border border-transparent hover:border-[#00D4FF]"
          >
            <div className="flex items-start gap-2 mb-2">
              {getSentimentIcon(item.sentiment)}
              <div className="flex-1">
                <h4 className="text-sm text-white mb-2 leading-tight">{item.title}</h4>
                
                <div className="flex items-center gap-2 flex-wrap mb-2">
                  <Badge className={getImpactColor(item.impact)}>
                    {item.impact.toUpperCase()}
                  </Badge>
                  {item.relatedStocks.map(stock => (
                    <span key={stock} className="text-xs text-[#00D4FF] bg-[#00D4FF] bg-opacity-10 px-2 py-1 rounded">
                      {stock}
                    </span>
                  ))}
                </div>

                <div className="flex items-center justify-between text-xs text-gray-400">
                  <span>{item.source}</span>
                  <span>{item.timestamp.toLocaleTimeString()}</span>
                </div>
              </div>
            </div>

            <div className={`text-xs mt-2 pt-2 border-t border-gray-800 ${getSentimentColor(item.sentiment)}`}>
              Sentiment: {item.sentiment.charAt(0).toUpperCase() + item.sentiment.slice(1)}
            </div>
          </div>
        ))}
      </div>

      {/* AI Insights Section */}
      <div className="mt-4 pt-4 border-t border-gray-800">
        <div className="flex items-center gap-2 mb-3">
          <AlertCircle className="w-4 h-4 text-[#FFB800]" />
          <span className="text-sm text-gray-400">AI Memory Insights</span>
        </div>
        <div className="space-y-2">
          {aiInsights.map((insight, idx) => (
            <div key={idx} className="bg-[#0F1420] rounded p-2">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-[#00D4FF]">{insight.stock}</span>
                <span className="text-xs text-gray-400">{insight.confidence}% confidence</span>
              </div>
              <p className="text-xs text-gray-300 leading-relaxed">{insight.insight}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
