import React, { useEffect, useMemo, useState } from 'react';
import { TopStatusBar } from './components/sci/TopStatusBar';
import { MainNavigation } from './components/sci/MainNavigation';
import { LeftStockSidebar } from './components/sci/LeftStockSidebar';
import { RightInfoPanel } from './components/sci/RightInfoPanel';
import { DashboardView } from './components/sci/DashboardView';
import { StockDetailView } from './components/sci/StockDetailView';
import { AIModelDetailView } from './components/sci/AIModelDetailView';
import { SectorRotationView } from './components/sci/SectorRotationView';
import { DecisionTimelineView } from './components/sci/DecisionTimelineView';
import { DEFAULT_STOCK_SYMBOLS, AI_MODELS } from './utils/stockData';
import type { Stock, AIModel } from './utils/stockData';
import { useModelDataCache } from './context/modelData';

type View = 
  | 'dashboard' 
  | 'stock-list' 
  | 'stock-detail' 
  | 'ai-arena'
  | 'ai-detail' 
  | 'sector' 
  | 'timeline' 
  | 'settings';

export default function App() {
  const [currentView, setCurrentView] = useState<View>('dashboard');
  const [selectedStock, setSelectedStock] = useState<Stock | null>(null);
  const [selectedAIModel, setSelectedAIModel] = useState<AIModel | null>(null);
  const [quotes, setQuotes] = useState<Record<string, Partial<Stock>>>({});
  const [activeSignature, setActiveSignature] = useState<string>(
    AI_MODELS[0]?.config.signature ?? ''
  );
  const { getStats } = useModelDataCache();

  const enrichedModels = useMemo(() => {
    return AI_MODELS.map((model) => {
      const stats = getStats(model.config.signature);
      if (!stats) {
        return model;
      }
      return {
        ...model,
        totalReturn:
          typeof stats.total_return_pct === 'number'
            ? stats.total_return_pct
            : model.totalReturn,
        sharpeRatio:
          typeof stats.sharpe_ratio === 'number'
            ? stats.sharpe_ratio
            : model.sharpeRatio,
        maxDrawdown:
          typeof stats.max_drawdown_pct === 'number'
            ? stats.max_drawdown_pct
            : model.maxDrawdown,
        positionCount:
          typeof stats.position_count === 'number'
            ? stats.position_count
            : model.positionCount,
        totalTrades:
          typeof stats.trade_count === 'number'
            ? stats.trade_count
            : typeof stats.total_records === 'number'
            ? stats.total_records
            : model.totalTrades,
        status: stats.total_records && stats.total_records > 0 ? 'active' : model.status,
      };
    });
  }, [getStats]);

  const handleStockSelect = (stock: Stock) => {
    setSelectedStock(stock);
    setCurrentView('stock-detail');
  };

  const handleAIModelClick = (model: AIModel) => {
    setSelectedAIModel(model);
    setActiveSignature(model.config.signature);
    setCurrentView('ai-detail');
  };

  const handleViewChange = (view: string) => {
    setCurrentView(view as View);
    // 清除选中的股票和AI模型（除非是详情页）
    if (view !== 'stock-detail') setSelectedStock(null);
    if (view !== 'ai-detail') setSelectedAIModel(null);
  };

  const handleBackToDashboard = () => {
    setCurrentView('dashboard');
    setSelectedStock(null);
    setSelectedAIModel(null);
  };

  const handleBackToStockList = () => {
    setCurrentView('stock-list');
    setSelectedStock(null);
  };

  const handleBackToAIArena = () => {
    setCurrentView('ai-arena');
    setSelectedAIModel(null);
  };

  const renderMainContent = () => {
    switch (currentView) {
      case 'dashboard':
        return (
          <DashboardView
            onAIModelClick={handleAIModelClick}
            activeSignature={activeSignature}
            onActiveSignatureChange={setActiveSignature}
          />
        );
      
      case 'stock-list':
        return (
          <div className="p-6 space-y-6">
            <div>
              <h1 className="text-white text-2xl mb-2">科创板股票池</h1>
              <p className="text-gray-400">10只科创板龙头股票实时监控</p>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              {DEFAULT_STOCK_SYMBOLS.map((stock) => {
                const overlay = quotes[stock.code] || {};
                const merged = { ...stock, ...overlay } as Stock;
                const isPositive = (merged.changePercent ?? 0) >= 0;
                const aiFocus = merged.aiHoldingCount ?? merged.aiAttention ?? 0;
                const aiTurnover = merged.aiTurnover ?? 0;
                const aiTrades = merged.aiTradeVolume ?? 0;
                const hot = aiFocus >= 3;
                const attentionClass = aiFocus > 0 ? 'text-[#1E40AF]' : 'text-gray-500';
                const aiVolumeClass = aiTrades > 0 ? 'text-blue-200' : 'text-gray-500';
                const aiTurnoverClass = aiTurnover > 0 ? 'text-blue-200' : 'text-gray-500';

                const formatVolume = (value: number) => {
                  if (!value) return '0';
                  if (value >= 10000) return `${(value / 10000).toFixed(0)}万`;
                  return value.toFixed(0);
                };
                return (
                  <div
                    key={merged.code}
                    onClick={() => handleStockSelect(merged)}
                    className="glass-card rounded-lg p-6 cursor-pointer hover:bg-opacity-80 transition-all hover:scale-[1.02] border border-gray-700 hover:border-[#1E40AF]"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-white text-xl mb-1">{merged.name}</h3>
                        <p className="text-gray-400 text-sm flex items-center gap-2">
                          {merged.code}
                          {hot && (
                            <span className="text-xs text-amber-400 border border-amber-400 px-1 py-0.5 rounded">
                              热
                            </span>
                          )}
                        </p>
                      </div>
                      <div className="text-right">
                        <div className="text-2xl text-white">¥{merged.price.toFixed(2)}</div>
                        <div className={`text-sm flex items-center justify-end gap-1 ${isPositive ? 'price-up' : 'price-down'}`}>
                          <span>{isPositive ? '+' : ''}{merged.changePercent.toFixed(2)}%</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <div className="text-gray-400 mb-1">成交量</div>
                        <div className="text-white">{(merged.volume / 10000).toFixed(0)}万</div>
                      </div>
                      <div>
                        <div className="text-gray-400 mb-1">换手率</div>
                        <div className="text-white">{merged.turnover.toFixed(2)}%</div>
                      </div>
                      <div>
                        <div className="text-gray-400 mb-1">AI关注</div>
                        <div className={attentionClass}>{aiFocus}</div>
                      </div>
                      <div>
                        <div className="text-gray-400 mb-1">AI成交量</div>
                        <div className={aiVolumeClass}>{formatVolume(aiTrades)}股</div>
                      </div>
                      <div>
                        <div className="text-gray-400 mb-1">AI换手</div>
                        <div className={aiTurnoverClass}>{aiTurnover.toFixed(2)}%</div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      
      case 'stock-detail':
        return selectedStock ? (
          <StockDetailView 
            stock={selectedStock} 
            onBack={handleBackToStockList}
          />
        ) : null;
      
      case 'ai-arena':
        return (
          <div className="p-6 space-y-6">
            <div>
              <h1 className="text-white text-2xl mb-2">AI竞技场</h1>
              <p className="text-gray-400">6个AI交易模型实时对比</p>
            </div>

            <div className="grid grid-cols-3 gap-4">
              {enrichedModels.map((model) => (
                <div
                  key={model.id}
                  onClick={() => handleAIModelClick(model)}
                  className="glass-card rounded-lg p-6 cursor-pointer hover:bg-opacity-80 transition-all hover:scale-[1.02] border-l-4"
                  style={{ borderLeftColor: model.color }}
                >
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-white text-lg">{model.name}</h3>
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: model.color }}
                    />
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400 text-sm">总收益率</span>
                      <span className={`text-xl ${model.totalReturn >= 0 ? 'price-up' : 'price-down'}`}>
                        {model.totalReturn >= 0 ? '+' : ''}{model.totalReturn.toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400 text-sm">夏普比率</span>
                      <span className="text-white">{model.sharpeRatio.toFixed(2)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400 text-sm">总交易</span>
                      <span className="text-white">{model.totalTrades ?? 0}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400 text-sm">持仓数</span>
                      <span className="text-white">{model.positionCount} / 10</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <DashboardView
              onAIModelClick={handleAIModelClick}
              activeSignature={activeSignature}
              onActiveSignatureChange={setActiveSignature}
            />
          </div>
        );
      
      case 'ai-detail':
        return selectedAIModel ? (
          <AIModelDetailView
            model={selectedAIModel}
            onBack={handleBackToAIArena}
          />
        ) : null;
      
      case 'sector':
        return <SectorRotationView />;
      
      case 'timeline':
        return <DecisionTimelineView />;
      
      case 'settings':
        return (
          <div className="p-6 space-y-6">
            <div>
              <h1 className="text-white text-2xl mb-2">系统配置</h1>
              <p className="text-gray-400">平台参数设置和管理</p>
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div className="glass-card rounded-lg p-6">
                <h3 className="text-white mb-4">交易规则配置</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-gray-700 bg-opacity-30 rounded">
                    <span className="text-gray-400">T+1限制</span>
                    <div className="w-12 h-6 bg-green-600 rounded-full relative">
                      <div className="w-5 h-5 bg-white rounded-full absolute right-0.5 top-0.5" />
                    </div>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-700 bg-opacity-30 rounded">
                    <span className="text-gray-400">手续费率</span>
                    <span className="text-white">0.03%</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-700 bg-opacity-30 rounded">
                    <span className="text-gray-400">印花税</span>
                    <span className="text-white">0.1%</span>
                  </div>
                </div>
              </div>

              <div className="glass-card rounded-lg p-6">
                <h3 className="text-white mb-4">AI模型配置</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-gray-700 bg-opacity-30 rounded">
                    <span className="text-gray-400">决策频率</span>
                    <span className="text-white">每小时</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-700 bg-opacity-30 rounded">
                    <span className="text-gray-400">运行模型数</span>
                    <span className="text-white">6 个</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-700 bg-opacity-30 rounded">
                    <span className="text-gray-400">自动交易</span>
                    <div className="w-12 h-6 bg-green-600 rounded-full relative">
                      <div className="w-5 h-5 bg-white rounded-full absolute right-0.5 top-0.5" />
                    </div>
                  </div>
                </div>
              </div>

              <div className="glass-card rounded-lg p-6">
                <h3 className="text-white mb-4">股票池管理</h3>
                <div className="space-y-2">
                  <div className="text-sm text-gray-400 mb-2">当前股票池 ({DEFAULT_STOCK_SYMBOLS.length}只)</div>
                  {DEFAULT_STOCK_SYMBOLS.map(stock => (
                    <div key={stock.code} className="flex items-center justify-between p-2 bg-gray-700 bg-opacity-30 rounded text-sm">
                      <span className="text-white">{stock.name}</span>
                      <span className="text-gray-400">{stock.code}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="glass-card rounded-lg p-6">
                <h3 className="text-white mb-4">数据管理</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-gray-700 bg-opacity-30 rounded">
                    <span className="text-gray-400">数据源</span>
                    <span className="text-green-500">已连接</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-700 bg-opacity-30 rounded">
                    <span className="text-gray-400">更新频率</span>
                    <span className="text-white">实时</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-700 bg-opacity-30 rounded">
                    <span className="text-gray-400">历史数据</span>
                    <span className="text-white">30天</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      
      default:
        return (
          <DashboardView
            onAIModelClick={handleAIModelClick}
            activeSignature={activeSignature}
            onActiveSignatureChange={setActiveSignature}
          />
        );
    }
  };

  // Fetch quotes periodically for the stock-list cards as well
  useEffect(() => {
    let cancelled = false;
    const codes = DEFAULT_STOCK_SYMBOLS.map((s) => s.code).join(',');
    const fetchQuotes = async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/market/quotes?codes=${encodeURIComponent(codes)}`);
        if (!res.ok) throw new Error('failed');
        const payload = await res.json();
        const map: Record<string, Partial<Stock>> = {};
        (payload.quotes || []).forEach((q: any) => {
          map[q.code] = {
            price: Number(q.price) || 0,
            changePercent: Number(q.changePercent) || 0,
            volume: Number(q.volume) || 0,
            turnover: Number(q.turnover) || 0,
            aiHoldingCount: Number(q.aiHoldingCount) || 0,
            aiTradeVolume: Number(q.aiTradeVolume) || 0,
            aiTurnover: Number(q.aiTurnoverPercent) || 0,
            aiAttention: Number(q.aiAttentionScore) || 0,
          };
        });
        if (!cancelled) setQuotes(map);
      } catch (e) {}
    };
    fetchQuotes();
    const t = setInterval(fetchQuotes, 30000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  return (
    <>
      <link rel="preconnect" href="https://fonts.googleapis.com" />
      <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
      
      <div className="flex flex-col h-screen bg-[#1F2937] overflow-hidden">
        {/* Top Status Bar */}
        <TopStatusBar />
        
        {/* Main Navigation */}
        <MainNavigation activeView={currentView} onViewChange={handleViewChange} />
        
        {/* Main Layout */}
        <div className="flex flex-1 overflow-hidden">
          {/* Left Stock Sidebar */}
          <LeftStockSidebar 
            onStockSelect={handleStockSelect}
            selectedStock={selectedStock}
          />
          
          {/* Main Content Area */}
          <div className="flex-1 overflow-y-auto scrollbar-thin">
            {renderMainContent()}
          </div>

          {/* Right Info Panel */}
          <RightInfoPanel
            activeSignature={activeSignature}
            onSignatureChange={setActiveSignature}
          />
        </div>
      </div>
    </>
  );
}
