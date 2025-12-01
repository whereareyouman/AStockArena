import { useState, useEffect } from 'react';
import { Brain, TrendingUp, TrendingDown, Clock, CheckCircle, XCircle, Loader } from 'lucide-react';
import { Badge } from './ui/badge';

interface AIDecision {
  id: string;
  timestamp: Date;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reasoning: string;
  status: 'pending' | 'filled' | 'failed';
  price?: number;
  quantity?: number;
}

export function AIDecisionStream() {
  const [decisions, setDecisions] = useState<AIDecision[]>([
    {
      id: '1',
      timestamp: new Date(Date.now() - 30000),
      symbol: 'AAPL',
      action: 'BUY',
      confidence: 87,
      reasoning: 'Strong upward momentum detected. RSI indicates oversold conditions. Historical pattern match: 89% success rate.',
      status: 'filled',
      price: 182.40,
      quantity: 50,
    },
    {
      id: '2',
      timestamp: new Date(Date.now() - 120000),
      symbol: 'TSLA',
      action: 'SELL',
      confidence: 92,
      reasoning: 'Bearish divergence on MACD. News sentiment turned negative (-0.68). Risk-reward ratio unfavorable.',
      status: 'filled',
      price: 238.15,
      quantity: 25,
    },
    {
      id: '3',
      timestamp: new Date(Date.now() - 180000),
      symbol: 'NVDA',
      action: 'HOLD',
      confidence: 76,
      reasoning: 'Consolidation phase detected. Waiting for breakout confirmation. Volume below average.',
      status: 'pending',
    },
  ]);

  useEffect(() => {
    // Simulate new AI decisions
    const interval = setInterval(() => {
      const symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN'];
      const actions: ('BUY' | 'SELL' | 'HOLD')[] = ['BUY', 'SELL', 'HOLD'];
      const reasonings = [
        'Volume surge detected with bullish pattern formation. Memory retrieved similar scenario from 2024-08-15.',
        'Breaking support level. Stop-loss triggered. Protecting capital per risk management protocol.',
        'Market volatility elevated. VIX above threshold. Maintaining neutral position.',
        'Earnings beat expectations. Institutional buying pressure increased. Positive catalyst confirmed.',
        'Technical resistance at $XXX. Taking profits at predetermined target. Risk management active.',
        'Correlation analysis suggests sector rotation. Entering position ahead of momentum.',
      ];

      if (Math.random() > 0.7) {
        const newDecision: AIDecision = {
          id: Date.now().toString(),
          timestamp: new Date(),
          symbol: symbols[Math.floor(Math.random() * symbols.length)],
          action: actions[Math.floor(Math.random() * actions.length)],
          confidence: Math.floor(Math.random() * 30) + 70,
          reasoning: reasonings[Math.floor(Math.random() * reasonings.length)],
          status: 'pending',
          price: Math.random() * 500 + 100,
          quantity: Math.floor(Math.random() * 50) + 10,
        };

        setDecisions(prev => [newDecision, ...prev].slice(0, 10));

        // Update status after a delay
        setTimeout(() => {
          setDecisions(prev => prev.map(d => 
            d.id === newDecision.id 
              ? { ...d, status: Math.random() > 0.1 ? 'filled' : 'failed' as 'filled' | 'failed' }
              : d
          ));
        }, 3000);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status: AIDecision['status']) => {
    switch (status) {
      case 'filled':
        return <CheckCircle className="w-4 h-4 text-[#00C805]" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-[#FF2E2E]" />;
      case 'pending':
        return <Loader className="w-4 h-4 text-[#FFB800] animate-spin" />;
    }
  };

  const getActionColor = (action: AIDecision['action']) => {
    switch (action) {
      case 'BUY':
        return 'bg-[#00C805] text-white';
      case 'SELL':
        return 'bg-[#FF2E2E] text-white';
      case 'HOLD':
        return 'bg-[#FFB800] text-[#0F1420]';
    }
  };

  return (
    <div className="glass-card rounded-lg p-4 h-full flex flex-col">
      <div className="flex items-center gap-2 mb-4">
        <Brain className="w-5 h-5 text-[#00D4FF]" />
        <h3 className="text-white">AI Decision Stream</h3>
        <div className="ml-auto">
          <div className="w-2 h-2 rounded-full bg-[#00C805] animate-pulse" />
        </div>
      </div>

      <div className="flex-1 overflow-auto space-y-3 pr-2">
        {decisions.map((decision) => (
          <div 
            key={decision.id}
            className="bg-[#0F1420] rounded-lg p-3 hover:bg-[#1A1F2E] transition-all cursor-pointer border border-transparent hover:border-[#00D4FF]"
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-2">
                <Badge className={getActionColor(decision.action)}>
                  {decision.action}
                </Badge>
                <span className="text-white">{decision.symbol}</span>
                {decision.action !== 'HOLD' && decision.quantity && (
                  <span className="text-xs text-gray-400">x{decision.quantity}</span>
                )}
              </div>
              
              <div className="flex items-center gap-2">
                <div className="text-xs text-gray-400 flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {decision.timestamp.toLocaleTimeString()}
                </div>
                {getStatusIcon(decision.status)}
              </div>
            </div>

            {/* Confidence Score */}
            <div className="mb-2">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-gray-400">Confidence</span>
                <span className="text-[#00D4FF]">{decision.confidence}%</span>
              </div>
              <div className="h-1.5 bg-[#1A1F2E] rounded overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-[#4361EE] to-[#00D4FF] transition-all duration-300"
                  style={{ width: `${decision.confidence}%` }}
                />
              </div>
            </div>

            {/* Reasoning */}
            <p className="text-xs text-gray-400 leading-relaxed">
              {decision.reasoning}
            </p>

            {decision.price && decision.status === 'filled' && (
              <div className="mt-2 pt-2 border-t border-gray-800 text-xs text-gray-400">
                Executed @ ${decision.price.toFixed(2)}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="mt-4 pt-4 border-t border-gray-800">
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="text-center">
            <div className="text-gray-400">Success Rate</div>
            <div className="text-[#00C805]">94.2%</div>
          </div>
          <div className="text-center">
            <div className="text-gray-400">Avg Confidence</div>
            <div className="text-[#00D4FF]">83.5%</div>
          </div>
          <div className="text-center">
            <div className="text-gray-400">Today</div>
            <div className="text-white">{decisions.length}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
