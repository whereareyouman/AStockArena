import { BarChart3, TrendingUp, Newspaper, Trophy, Wallet, Brain, Settings, Activity, Target, LineChart } from 'lucide-react';

interface LeftNavSidebarProps {
  activeView: string;
  onViewChange: (view: string) => void;
}

export function LeftNavSidebar({ activeView, onViewChange }: LeftNavSidebarProps) {
  const menuItems = [
    { id: 'dashboard', icon: BarChart3, label: 'Live Dashboard', description: 'Real-time trading' },
    { id: 'stocks', icon: TrendingUp, label: 'Stock Analysis', description: 'Deep dive analytics' },
    { id: 'portfolio', icon: Wallet, label: 'Portfolio', description: 'Your holdings' },
    { id: 'ai-decisions', icon: Brain, label: 'AI Decisions', description: 'Decision stream' },
    { id: 'leaderboard', icon: Trophy, label: 'Leaderboard', description: 'Competition rank' },
    { id: 'market-events', icon: Activity, label: 'Market Events', description: 'Live updates' },
    { id: 'strategies', icon: Target, label: 'Strategies', description: 'Trading strategies' },
    { id: 'analytics', icon: LineChart, label: 'Analytics', description: 'Performance data' },
    { id: 'settings', icon: Settings, label: 'Settings', description: 'Preferences' },
  ];

  return (
    <div className="w-72 bg-[#0F1420] border-r border-gray-800 h-screen flex flex-col overflow-y-auto">
      {/* Logo */}
      <div className="p-6 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[#1CE479] to-[#00C805] flex items-center justify-center">
            <Brain className="w-6 h-6 text-[#0F1420]" />
          </div>
          <div>
            <h2 className="text-white">AI Trading Arena</h2>
            <p className="text-xs text-gray-400">Powered by AI</p>
          </div>
        </div>
      </div>

      {/* Navigation Menu */}
      <nav className="flex-1 p-4">
        <div className="space-y-1">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeView === item.id;
            
            return (
              <button
                key={item.id}
                onClick={() => onViewChange(item.id)}
                className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all group ${
                  isActive
                    ? 'bg-[#1CE479] text-[#0F1420] shadow-lg shadow-[#1CE479]/20'
                    : 'text-gray-400 hover:text-white hover:bg-[#1A1F2E]'
                }`}
              >
                <Icon className={`w-5 h-5 ${isActive ? 'text-[#0F1420]' : 'text-gray-400 group-hover:text-[#1CE479]'}`} />
                <div className="flex-1 text-left">
                  <div className={`text-sm ${isActive ? 'font-semibold' : ''}`}>
                    {item.label}
                  </div>
                  <div className={`text-xs ${isActive ? 'text-[#0F1420] opacity-70' : 'text-gray-500'}`}>
                    {item.description}
                  </div>
                </div>
                {isActive && (
                  <div className="w-1.5 h-8 bg-[#0F1420] rounded-full" />
                )}
              </button>
            );
          })}
        </div>
      </nav>

      {/* User Profile */}
      <div className="p-4 border-t border-gray-800">
        <div className="glass-card rounded-lg p-3">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#1CE479] to-[#00C805] flex items-center justify-center">
              <span className="text-[#0F1420]">AI</span>
            </div>
            <div className="flex-1">
              <p className="text-white text-sm">AI Trader</p>
              <p className="text-xs text-gray-400">Premium Account</p>
            </div>
          </div>
          <div className="pt-3 border-t border-gray-800">
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-gray-400">Portfolio Value</span>
              <span className="text-white">$50,526.75</span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-gray-400">Today's P&L</span>
              <span className="price-up">+$526.75 (1.05%)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
