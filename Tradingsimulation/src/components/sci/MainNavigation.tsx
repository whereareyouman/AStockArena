import { Home, BarChart3, Brain, TrendingUp, Clock, Settings, FileText } from 'lucide-react';

interface MainNavigationProps {
  activeView: string;
  onViewChange: (view: string) => void;
}

export function MainNavigation({ activeView, onViewChange }: MainNavigationProps) {
  const navItems = [
    { id: 'dashboard', label: '首页总览', icon: Home, description: '实时监控面板' },
    { id: 'stock-list', label: '股票池', icon: BarChart3, description: '10只科创板股票' },
    { id: 'ai-arena', label: 'AI竞技场', icon: Brain, description: '6个AI模型对比' },
    { id: 'sector', label: '板块轮动', icon: TrendingUp, description: '三大板块分析' },
    { id: 'timeline', label: '决策时序', icon: Clock, description: '24小时决策矩阵' },
    { id: 'settings', label: '系统配置', icon: Settings, description: '参数设置' },
  ];

  return (
    <div className="h-14 bg-[#111827] border-b border-gray-700 px-6 flex items-center gap-1">
      {navItems.map((item) => {
        const Icon = item.icon;
        const isActive = activeView === item.id;
        
        return (
          <button
            key={item.id}
            onClick={() => onViewChange(item.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              isActive
                ? 'bg-[#1E40AF] text-white shadow-lg'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            <Icon className="w-4 h-4" />
            <span className="text-sm">{item.label}</span>
          </button>
        );
      })}
    </div>
  );
}
