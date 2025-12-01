import { useState, useEffect } from 'react';
import { Trophy, TrendingUp, TrendingDown, Award, Target } from 'lucide-react';
import { Badge } from './ui/badge';

interface Participant {
  rank: number;
  name: string;
  pnl: number;
  pnlPercent: number;
  strategy: string;
  winRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
  trades: number;
  avatar: string;
}

export function Leaderboard() {
  const [timeframe, setTimeframe] = useState<'daily' | 'weekly' | 'monthly'>('daily');
  const [participants, setParticipants] = useState<Participant[]>([
    {
      rank: 1,
      name: 'AlphaTrader_AI',
      pnl: 12450.50,
      pnlPercent: 24.9,
      strategy: 'Momentum',
      winRate: 68.5,
      sharpeRatio: 2.4,
      maxDrawdown: -5.2,
      trades: 147,
      avatar: '🤖',
    },
    {
      rank: 2,
      name: 'QuantMaster',
      pnl: 10823.75,
      pnlPercent: 21.6,
      strategy: 'Mean Reversion',
      winRate: 71.2,
      sharpeRatio: 2.1,
      maxDrawdown: -4.8,
      trades: 203,
      avatar: '📊',
    },
    {
      rank: 3,
      name: 'You',
      pnl: 9567.25,
      pnlPercent: 19.1,
      strategy: 'Hybrid',
      winRate: 64.8,
      sharpeRatio: 1.9,
      maxDrawdown: -6.5,
      trades: 132,
      avatar: '👤',
    },
    {
      rank: 4,
      name: 'MarketWizard',
      pnl: 8234.80,
      pnlPercent: 16.5,
      strategy: 'Breakout',
      winRate: 59.3,
      sharpeRatio: 1.7,
      maxDrawdown: -7.2,
      trades: 98,
      avatar: '🧙',
    },
    {
      rank: 5,
      name: 'TrendFollower',
      pnl: 7456.30,
      pnlPercent: 14.9,
      strategy: 'Trend Following',
      winRate: 62.1,
      sharpeRatio: 1.6,
      maxDrawdown: -8.1,
      trades: 156,
      avatar: '📈',
    },
  ]);

  useEffect(() => {
    // Simulate live ranking updates
    const interval = setInterval(() => {
      setParticipants(prev => prev.map(p => {
        const change = (Math.random() - 0.5) * 500;
        const newPnl = p.pnl + change;
        const newPnlPercent = (newPnl / 50000) * 100;
        
        return {
          ...p,
          pnl: parseFloat(newPnl.toFixed(2)),
          pnlPercent: parseFloat(newPnlPercent.toFixed(1)),
        };
      }).sort((a, b) => b.pnl - a.pnl).map((p, idx) => ({ ...p, rank: idx + 1 })));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getRankColor = (rank: number) => {
    switch (rank) {
      case 1:
        return 'from-yellow-500 to-yellow-600';
      case 2:
        return 'from-gray-300 to-gray-400';
      case 3:
        return 'from-orange-400 to-orange-500';
      default:
        return 'from-[#00D4FF] to-[#4361EE]';
    }
  };

  const getRankIcon = (rank: number) => {
    switch (rank) {
      case 1:
        return <Trophy className="w-5 h-5 text-yellow-500" />;
      case 2:
        return <Award className="w-5 h-5 text-gray-400" />;
      case 3:
        return <Award className="w-5 h-5 text-orange-500" />;
      default:
        return null;
    }
  };

  return (
    <div className="glass-card rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Trophy className="w-5 h-5 text-[#FFB800]" />
          <h3 className="text-white">Competition Leaderboard</h3>
        </div>
        
        <div className="flex gap-1 bg-[#0F1420] rounded p-1">
          {(['daily', 'weekly', 'monthly'] as const).map((tf) => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              className={`px-3 py-1 rounded text-xs transition-colors capitalize ${
                timeframe === tf 
                  ? 'bg-[#00D4FF] text-[#0F1420]' 
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-2">
        {participants.map((participant) => (
          <div 
            key={participant.name}
            className={`bg-[#0F1420] rounded-lg p-4 transition-all ${
              participant.name === 'You' 
                ? 'border-2 border-[#00D4FF] shadow-lg shadow-[#00D4FF]/20' 
                : 'hover:bg-[#1A1F2E]'
            }`}
          >
            <div className="flex items-center gap-3 mb-3">
              {/* Rank */}
              <div className={`w-10 h-10 rounded-full bg-gradient-to-br ${getRankColor(participant.rank)} flex items-center justify-center flex-shrink-0`}>
                <span className="text-white">#{participant.rank}</span>
              </div>

              {/* Avatar & Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xl">{participant.avatar}</span>
                  <span className="text-white truncate">{participant.name}</span>
                  {participant.rank <= 3 && getRankIcon(participant.rank)}
                  {participant.name === 'You' && (
                    <Badge className="bg-[#00D4FF] text-[#0F1420]">You</Badge>
                  )}
                </div>
                <div className="flex items-center gap-2 text-xs text-gray-400">
                  <Target className="w-3 h-3" />
                  <span>{participant.strategy}</span>
                  <span>•</span>
                  <span>{participant.trades} trades</span>
                </div>
              </div>

              {/* PnL */}
              <div className="text-right">
                <div className={`flex items-center gap-1 justify-end ${
                  participant.pnl >= 0 ? 'price-up' : 'price-down'
                }`}>
                  {participant.pnl >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                  <span className="text-lg">
                    {participant.pnl >= 0 ? '+' : ''}${participant.pnl.toFixed(2)}
                  </span>
                </div>
                <div className={`text-xs ${participant.pnl >= 0 ? 'price-up' : 'price-down'}`}>
                  {participant.pnlPercent >= 0 ? '+' : ''}{participant.pnlPercent}%
                </div>
              </div>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-3 gap-4 pt-3 border-t border-gray-800">
              <div className="text-center">
                <div className="text-xs text-gray-400 mb-1">Win Rate</div>
                <div className="text-white">{participant.winRate}%</div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-400 mb-1">Sharpe</div>
                <div className="text-[#00D4FF]">{participant.sharpeRatio.toFixed(1)}</div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-400 mb-1">Max DD</div>
                <div className="text-[#FF2E2E]">{participant.maxDrawdown}%</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 pt-4 border-t border-gray-800">
        <div className="text-center text-xs text-gray-400">
          Live rankings update every 5 seconds
        </div>
      </div>
    </div>
  );
}
