import { useState, useEffect } from 'react';
import { Calendar, DollarSign, TrendingUp, Bell } from 'lucide-react';
import { Badge } from './ui/badge';

interface MarketEvent {
  id: string;
  type: 'earnings' | 'economic' | 'fed' | 'corporate' | 'breaking';
  title: string;
  time: Date;
  impact: 'high' | 'medium' | 'low';
  details: string;
  relatedSymbols?: string[];
}

export function MarketEventsStream() {
  const [events, setEvents] = useState<MarketEvent[]>([
    {
      id: '1',
      type: 'earnings',
      title: 'AAPL Earnings Beat',
      time: new Date(),
      impact: 'high',
      details: 'EPS: $1.52 vs $1.45 expected (+4.8% surprise)',
      relatedSymbols: ['AAPL'],
    },
    {
      id: '2',
      type: 'economic',
      title: 'CPI Data Release',
      time: new Date(Date.now() - 1800000),
      impact: 'high',
      details: 'Inflation at 3.2% YoY, slightly below 3.3% forecast',
      relatedSymbols: ['SPY', 'QQQ'],
    },
    {
      id: '3',
      type: 'corporate',
      title: 'NVDA Stock Split Announced',
      time: new Date(Date.now() - 3600000),
      impact: 'medium',
      details: '10-for-1 stock split effective next month',
      relatedSymbols: ['NVDA'],
    },
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      const eventTypes: MarketEvent['type'][] = ['earnings', 'economic', 'fed', 'corporate', 'breaking'];
      const titles = [
        'Fed Rate Decision Pending',
        'Employment Report Strong',
        'Tech Sector Rotation Detected',
        'Major M&A Announcement',
        'Volatility Spike Alert',
      ];
      
      if (Math.random() > 0.85) {
        const newEvent: MarketEvent = {
          id: Date.now().toString(),
          type: eventTypes[Math.floor(Math.random() * eventTypes.length)],
          title: titles[Math.floor(Math.random() * titles.length)],
          time: new Date(),
          impact: ['high', 'medium', 'low'][Math.floor(Math.random() * 3)] as 'high' | 'medium' | 'low',
          details: 'Market reacting to new development. AI analyzing impact on positions.',
          relatedSymbols: ['AAPL', 'TSLA', 'NVDA'].slice(0, Math.floor(Math.random() * 2) + 1),
        };
        
        setEvents(prev => [newEvent, ...prev].slice(0, 6));
      }
    }, 8000);

    return () => clearInterval(interval);
  }, []);

  const getEventIcon = (type: MarketEvent['type']) => {
    switch (type) {
      case 'earnings':
        return <DollarSign className="w-4 h-4 text-[#00C805]" />;
      case 'economic':
        return <TrendingUp className="w-4 h-4 text-[#00D4FF]" />;
      case 'fed':
        return <Calendar className="w-4 h-4 text-[#FFB800]" />;
      case 'corporate':
        return <Bell className="w-4 h-4 text-[#4361EE]" />;
      case 'breaking':
        return <Bell className="w-4 h-4 text-[#FF2E2E]" />;
    }
  };

  const getImpactColor = (impact: MarketEvent['impact']) => {
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
    <div className="glass-card rounded-lg p-4">
      <div className="flex items-center gap-2 mb-4">
        <Bell className="w-5 h-5 text-[#FFB800]" />
        <h3 className="text-white">Live Market Events</h3>
        <div className="ml-auto w-2 h-2 rounded-full bg-[#FF2E2E] animate-pulse" />
      </div>

      <div className="space-y-3">
        {events.map((event) => (
          <div 
            key={event.id}
            className="bg-[#0F1420] rounded-lg p-3 hover:bg-[#1A1F2E] transition-all cursor-pointer border border-transparent hover:border-[#00D4FF]"
          >
            <div className="flex items-start gap-3">
              <div className="mt-1">{getEventIcon(event.type)}</div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <h4 className="text-sm text-white mb-1">{event.title}</h4>
                    <p className="text-xs text-gray-400 leading-relaxed">{event.details}</p>
                  </div>
                  <Badge className={`${getImpactColor(event.impact)} ml-2 flex-shrink-0`}>
                    {event.impact}
                  </Badge>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 flex-wrap">
                    {event.relatedSymbols?.map(symbol => (
                      <span 
                        key={symbol}
                        className="text-xs text-[#00D4FF] bg-[#00D4FF] bg-opacity-10 px-2 py-0.5 rounded"
                      >
                        {symbol}
                      </span>
                    ))}
                  </div>
                  <span className="text-xs text-gray-500">
                    {event.time.toLocaleTimeString()}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
