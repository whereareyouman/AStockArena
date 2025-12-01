import { useState, useEffect } from 'react';
import { Activity, AlertCircle, Newspaper, Bell } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { RecentDecisions } from './RecentDecisions';
import { LiveNewsPanel } from './LiveNewsPanel';
import { AI_MODELS } from '../../utils/stockData';

interface RightInfoPanelProps {
  activeSignature: string;
  onSignatureChange?: (signature: string) => void;
}

export function RightInfoPanel({ activeSignature, onSignatureChange }: RightInfoPanelProps) {
  const [systemLogs, setSystemLogs] = useState<Array<{ time: Date; message: string; type: 'info' | 'warning' | 'success' }>>([]);

  useEffect(() => {
    // 初始化系统日志
    setSystemLogs([
      { time: new Date(), message: 'TinySoft 数据源连接正常', type: 'success' },
      { time: new Date(Date.now() - 60000), message: 'Gemini AI模型已激活', type: 'success' },
      { time: new Date(Date.now() - 120000), message: '后端服务运行正常', type: 'success' },
    ]);

    // Polling for new system events (simplified - you can expand this)
    const interval = setInterval(() => {
      // In real implementation, fetch from /api/system/logs or similar
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const getTimeAgo = (date: Date) => {
    const diff = Date.now() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    if (minutes < 60) return `${minutes}分钟前`;
    const hours = Math.floor(minutes / 60);
    return `${hours}小时前`;
  };

  return (
    <div className="w-96 bg-[#111827] border-l border-gray-700 h-screen flex flex-col overflow-hidden">
      <Tabs defaultValue="decisions" className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-gray-700 flex-shrink-0">
          <TabsList className="grid w-full grid-cols-3 bg-gray-700">
            <TabsTrigger value="decisions" className="text-xs">
              <Activity className="w-3 h-3 mr-1" />
              决策流
            </TabsTrigger>
            <TabsTrigger value="news" className="text-xs">
              <Newspaper className="w-3 h-3 mr-1" />
              新闻
            </TabsTrigger>
            <TabsTrigger value="system" className="text-xs">
              <Bell className="w-3 h-3 mr-1" />
              系统
            </TabsTrigger>
          </TabsList>
        </div>

        {/* AI Decisions Tab */}
        <TabsContent value="decisions" className="flex-1 overflow-hidden m-0 flex flex-col">
          <div className="p-4 border-b border-gray-700 flex-shrink-0 space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-white text-sm">最近决策</h3>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-xs text-gray-400">实时更新</span>
              </div>
            </div>
            <div className="flex items-center justify-between text-xs text-gray-400">
              <span>模型</span>
              <select
                value={activeSignature}
                onChange={(e) => onSignatureChange?.(e.target.value)}
                className="bg-gray-800 text-white text-xs rounded px-2 py-1 border border-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                {AI_MODELS.map((model) => (
                  <option key={model.config.signature} value={model.config.signature}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto scrollbar-thin p-4">
            <RecentDecisions signature={activeSignature} />
          </div>
        </TabsContent>

        {/* News Tab */}
        <TabsContent value="news" className="flex-1 overflow-hidden m-0 flex flex-col">
          <div className="p-4 border-b border-gray-700 flex-shrink-0">
            <h3 className="text-white text-sm">市场新闻</h3>
          </div>

          <div className="flex-1 overflow-y-auto scrollbar-thin p-4">
            <LiveNewsPanel />
          </div>
        </TabsContent>

        {/* System Tab */}
        <TabsContent value="system" className="flex-1 overflow-hidden m-0 flex flex-col">
          <div className="p-4 border-b border-gray-700 flex-shrink-0">
            <h3 className="text-white text-sm">系统状态</h3>
          </div>

          <div className="flex-1 overflow-y-auto scrollbar-thin p-4">
              <div className="space-y-3">
                {systemLogs.map((log, idx) => {
                  const typeConfig = {
                    info: { icon: AlertCircle, color: 'text-blue-400' },
                    warning: { icon: AlertCircle, color: 'text-yellow-400' },
                    success: { icon: Activity, color: 'text-green-400' },
                  };
                  const config = typeConfig[log.type];
                  const LogIcon = config.icon;

                  return (
                    <div
                      key={idx}
                      className="flex items-start gap-3 p-3 glass-card rounded-lg"
                    >
                      <LogIcon className={`w-4 h-4 mt-0.5 ${config.color}`} />
                      <div className="flex-1">
                        <p className="text-white text-sm mb-1">{log.message}</p>
                        <span className="text-xs text-gray-400">{getTimeAgo(log.time)}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
