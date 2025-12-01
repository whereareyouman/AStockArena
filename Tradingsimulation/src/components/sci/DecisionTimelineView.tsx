import { useState, useEffect } from 'react';
import { Clock, TrendingUp, TrendingDown, Activity, AlertCircle } from 'lucide-react';
import { Badge } from '../ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip';
import { AI_MODELS, DEFAULT_STOCK_SYMBOLS, generateMockDecisions, type Decision } from '../../utils/stockData';

export function DecisionTimelineView() {
  const [selectedHour, setSelectedHour] = useState<number | null>(null);
  const [decisionMatrix, setDecisionMatrix] = useState<Map<string, Decision>>(new Map());
  const [hourlyDecisions, setHourlyDecisions] = useState<Decision[]>([]);

  useEffect(() => {
    // 生成24小时 x 6个AI x 10只股票的决策矩阵
    const generateDecisionMatrix = () => {
      const matrix = new Map<string, Decision>();
      const decisions = generateMockDecisions(24 * 6 * 3); // 生成足够的决策数据
      
      decisions.forEach(decision => {
        const hour = decision.timestamp.getHours();
        const key = `${hour}-${decision.aiModelId}-${decision.stockCode}`;
        matrix.set(key, decision);
      });
      
      return matrix;
    };

    setDecisionMatrix(generateDecisionMatrix());
  }, []);

  const handleHourClick = (hour: number) => {
    setSelectedHour(hour);
    
    // 获取该小时的所有决策
    const decisions: Decision[] = [];
    AI_MODELS.forEach(ai => {
      DEFAULT_STOCK_SYMBOLS.forEach(stock => {
        const key = `${hour}-${ai.id}-${stock.code}`;
        const decision = decisionMatrix.get(key);
        if (decision) {
          decisions.push(decision);
        }
      });
    });
    
    setHourlyDecisions(decisions);
  };

  const getActionColor = (action: Decision['action'] | undefined) => {
    if (!action) return '#374151';
    switch (action) {
      case 'buy':
        return '#EF4444'; // 红色买入
      case 'sell':
        return '#10B981'; // 绿色卖出
      case 'hold':
        return '#6B7280'; // 灰色持有
    }
  };

  const getActionLabel = (action: Decision['action'] | undefined) => {
    if (!action) return '-';
    switch (action) {
      case 'buy':
        return '买';
      case 'sell':
        return '卖';
      case 'hold':
        return '持';
    }
  };

  // 计算每小时的决策统计
  const getHourStats = (hour: number) => {
    let buy = 0, sell = 0, hold = 0;
    
    AI_MODELS.forEach(ai => {
      DEFAULT_STOCK_SYMBOLS.forEach(stock => {
        const key = `${hour}-${ai.id}-${stock.code}`;
        const decision = decisionMatrix.get(key);
        if (decision) {
          if (decision.action === 'buy') buy++;
          else if (decision.action === 'sell') sell++;
          else hold++;
        }
      });
    });
    
    return { buy, sell, hold, total: buy + sell + hold };
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-white text-2xl mb-2">决策时序分析</h1>
        <p className="text-gray-400">24小时AI决策矩阵热力图 · 每小时观察决策循环</p>
      </div>

      {/* Timeline */}
      <div className="glass-card rounded-lg p-6">
        <h3 className="text-white mb-4">24小时决策时间轴</h3>
        
        {/* Hour Timeline */}
        <div className="flex gap-2 mb-6 overflow-x-auto pb-2 scrollbar-thin">
          {Array.from({ length: 24 }, (_, i) => {
            const stats = getHourStats(i);
            const isSelected = selectedHour === i;
            const currentHour = new Date().getHours();
            const isCurrent = i === currentHour;
            
            return (
              <TooltipProvider key={i}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      onClick={() => handleHourClick(i)}
                      className={`flex-shrink-0 w-20 p-3 rounded-lg transition-all ${
                        isSelected
                          ? 'bg-[#1E40AF] scale-110 shadow-lg'
                          : 'glass-card hover:bg-gray-700'
                      } ${isCurrent ? 'ring-2 ring-yellow-500' : ''}`}
                    >
                      <div className="flex items-center justify-center gap-1 mb-2">
                        <Clock className="w-3 h-3 text-gray-400" />
                        <span className="text-white text-sm">{i}:00</span>
                      </div>
                      <div className="flex gap-1 justify-center">
                        <div 
                          className="w-2 h-8 rounded"
                          style={{ 
                            backgroundColor: '#EF4444',
                            opacity: stats.buy / stats.total || 0.2,
                          }}
                        />
                        <div 
                          className="w-2 h-8 rounded"
                          style={{ 
                            backgroundColor: '#10B981',
                            opacity: stats.sell / stats.total || 0.2,
                          }}
                        />
                        <div 
                          className="w-2 h-8 rounded"
                          style={{ 
                            backgroundColor: '#6B7280',
                            opacity: stats.hold / stats.total || 0.2,
                          }}
                        />
                      </div>
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <div className="text-xs">
                      <div>买入: {stats.buy}次</div>
                      <div>卖出: {stats.sell}次</div>
                      <div>持有: {stats.hold}次</div>
                    </div>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            );
          })}
        </div>

        {/* Decision Matrix */}
        {selectedHour !== null && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-white">
                {selectedHour}:00 时刻决策矩阵
              </h4>
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-red-500 rounded" />
                  <span className="text-gray-400">买入</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded" />
                  <span className="text-gray-400">卖出</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-gray-500 rounded" />
                  <span className="text-gray-400">持有</span>
                </div>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left text-gray-400 p-2 sticky left-0 bg-[#111827]">AI模型</th>
                    {DEFAULT_STOCK_SYMBOLS.slice(0, 10).map(stock => (
                      <th key={stock.code} className="text-center text-gray-400 p-2 text-xs">
                        <div>{stock.name}</div>
                        <div className="text-gray-500">{stock.code}</div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {AI_MODELS.map(ai => (
                    <tr key={ai.id} className="border-b border-gray-700">
                      <td className="p-2 sticky left-0 bg-[#111827]">
                        <div className="flex items-center gap-2">
                          <div 
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: ai.color }}
                          />
                          <span className="text-white text-sm">{ai.name}</span>
                        </div>
                      </td>
                      {DEFAULT_STOCK_SYMBOLS.slice(0, 10).map(stock => {
                        const key = `${selectedHour}-${ai.id}-${stock.code}`;
                        const decision = decisionMatrix.get(key);
                        
                        return (
                          <td key={stock.code} className="p-1">
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <div
                                    className="w-12 h-12 rounded flex items-center justify-center cursor-pointer hover:scale-110 transition-transform"
                                    style={{ 
                                      backgroundColor: getActionColor(decision?.action),
                                      opacity: decision ? 0.8 : 0.3,
                                    }}
                                  >
                                    <span className="text-white">
                                      {getActionLabel(decision?.action)}
                                    </span>
                                  </div>
                                </TooltipTrigger>
                                {decision && (
                                  <TooltipContent>
                                    <div className="text-xs space-y-1">
                                      <div className="font-semibold">{ai.name}</div>
                                      <div>{stock.name} ({stock.code})</div>
                                      <div>操作: {getActionLabel(decision.action)}</div>
                                      <div>价格: ¥{decision.price.toFixed(2)}</div>
                                      <div>置信度: {decision.confidence.toFixed(1)}%</div>
                                    </div>
                                  </TooltipContent>
                                )}
                              </Tooltip>
                            </TooltipProvider>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Hourly Decisions Detail */}
      {selectedHour !== null && hourlyDecisions.length > 0 && (
        <div className="glass-card rounded-lg p-6">
          <h3 className="text-white mb-4">{selectedHour}:00 详细决策记录</h3>
          <div className="grid grid-cols-2 gap-4">
            {hourlyDecisions.slice(0, 6).map((decision, idx) => {
              const ai = AI_MODELS.find(m => m.id === decision.aiModelId);
              const actionConfig = {
                buy: { label: '买入', color: 'bg-red-600', icon: TrendingUp },
                sell: { label: '卖出', color: 'bg-green-600', icon: TrendingDown },
                hold: { label: '持有', color: 'bg-gray-600', icon: Activity },
              }[decision.action];
              const ActionIcon = actionConfig.icon;

              return (
                <div key={idx} className="p-4 bg-gray-700 bg-opacity-30 rounded-lg">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div 
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: ai?.color }}
                      />
                      <span className="text-white text-sm">{ai?.name}</span>
                    </div>
                    <Badge className={`${actionConfig.color} text-white`}>
                      <ActionIcon className="w-3 h-3 mr-1" />
                      {actionConfig.label}
                    </Badge>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">股票</span>
                      <span className="text-white">{decision.stockCode}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">价格</span>
                      <span className="text-white">¥{decision.price.toFixed(2)}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">置信度</span>
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-gray-700 rounded overflow-hidden">
                          <div 
                            className="h-full bg-[#1E40AF]"
                            style={{ width: `${decision.confidence}%` }}
                          />
                        </div>
                        <span className="text-white">{decision.confidence.toFixed(0)}%</span>
                      </div>
                    </div>
                    <div className="pt-2 border-t border-gray-700">
                      <div className="text-xs text-gray-400 mb-1">决策理由:</div>
                      <p className="text-xs text-gray-300">{decision.reasoning}</p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Statistics */}
      <div className="grid grid-cols-4 gap-4">
        <div className="glass-card rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-2">今日总决策数</div>
          <div className="text-2xl text-white">{decisionMatrix.size}</div>
        </div>
        <div className="glass-card rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-2">买入信号</div>
          <div className="text-2xl price-up">
            {Array.from(decisionMatrix.values()).filter(d => d.action === 'buy').length}
          </div>
        </div>
        <div className="glass-card rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-2">卖出信号</div>
          <div className="text-2xl price-down">
            {Array.from(decisionMatrix.values()).filter(d => d.action === 'sell').length}
          </div>
        </div>
        <div className="glass-card rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-2">决策一致性</div>
          <div className="text-2xl text-white">
            {selectedHour !== null 
              ? ((getHourStats(selectedHour).buy / Math.max(getHourStats(selectedHour).total, 1)) * 100).toFixed(0)
              : 0}%
          </div>
        </div>
      </div>
    </div>
  );
}
