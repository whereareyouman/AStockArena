import React, { useEffect, useMemo, useRef, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { AIModelCard } from './AIModelCard';
import { TradingControl } from './TradingControl';
import { AI_MODELS, type AIModel } from '../../utils/stockData';
import { TrendingUp } from 'lucide-react';
import { useModelDataCache } from '../../context/modelData';
import type { ModelStatsResponse } from '../../types/modelStats';

interface DashboardViewProps {
  onAIModelClick: (model: AIModel) => void;
  activeSignature: string;
  onActiveSignatureChange?: (signature: string) => void;
}

const DEFAULT_ACTIVE_SIGNATURE = AI_MODELS[0]?.config.signature ?? '';

const buildChartData = (seriesMap: Record<string, any[]>) => {
  const dateMap = new Map<string, Record<string, number | null>>();
  AI_MODELS.forEach((model) => {
    const signature = model.config.signature;
    const items = seriesMap[signature] || [];
    const lookup = new Map(items.map((item: any) => [item.date, item.returnPct ?? 0]));
    lookup.forEach((value, date) => {
      if (!date) return;
      if (!dateMap.has(date)) {
        dateMap.set(date, { time: date });
      }
      dateMap.get(date)![model.id] = value;
    });
  });
  return Array.from(dateMap.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([, data]) => data);
};

const formatValuationSource = (source?: string) => {
  switch (source) {
    case 'cache-hourly':
      return '共享快照（小时）';
    case 'cache-daily':
      return '共享快照（日）';
    case 'tinystock':
      return '实时行情';
    case 'fallback':
      return '成本均价';
    case 'mixed':
      return '混合估值';
    case 'cash-only':
      return '仅现金';
    default:
      return '—';
  }
};

export function DashboardView({
  onAIModelClick,
  activeSignature = DEFAULT_ACTIVE_SIGNATURE,
  onActiveSignatureChange,
}: DashboardViewProps) {
  const [performanceData, setPerformanceData] = useState<Array<any>>([]);
  const [modelStats, setModelStats] = useState<Record<string, ModelStatsResponse>>({});
  const [loading, setLoading] = useState(false);
  const seriesRef = useRef<Record<string, any[]>>({});
  const { getStats, getPnlSeries, updateStats, updatePnlSeries } = useModelDataCache();

  useEffect(() => {
    let cancelled = false;

    const hydrateFromCache = () => {
      const cachedStats: Record<string, ModelStatsResponse> = {};
      const cachedSeries: Record<string, any[]> = {};
      let hasCache = false;

      AI_MODELS.forEach((model) => {
        const signature = model.config.signature;
        const stats = getStats(signature);
        if (stats) {
          cachedStats[signature] = stats;
          hasCache = true;
        }
        const series = getPnlSeries(signature);
        if (series) {
          cachedSeries[signature] = series;
          hasCache = true;
        }
      });

      if (hasCache && !cancelled) {
        seriesRef.current = { ...seriesRef.current, ...cachedSeries };
        setModelStats((prev) => ({ ...prev, ...cachedStats }));
        setPerformanceData(buildChartData(seriesRef.current));
        setLoading(false);
      }
    };

    const fetchAllModels = async () => {
      setLoading((prev) => prev || Object.keys(seriesRef.current).length === 0);
      const statsAccumulator: Record<string, ModelStatsResponse> = {};
      const seriesAccumulator: Record<string, any[]> = {};

      await Promise.all(
        AI_MODELS.map(async (model) => {
          const signature = model.config.signature;
          try {
            const [pnlRes, statsRes] = await Promise.all([
              fetch(
                `http://localhost:8000/api/live/pnl-series?signature=${signature}&days=30&valuation=equity`
              ),
              fetch(`http://localhost:8000/api/live/model-stats?signature=${signature}`),
            ]);

            if (pnlRes.ok) {
              const payload = await pnlRes.json();
              const items = Array.isArray(payload.items) ? payload.items : [];
              seriesAccumulator[signature] = items;
              updatePnlSeries(signature, items);
            }

            if (statsRes.ok) {
              const statsPayload = await statsRes.json();
              statsAccumulator[signature] = statsPayload;
              updateStats(signature, statsPayload);
            }
          } catch (err) {
            console.warn(`Failed to fetch metrics for ${signature}:`, err);
          }
        })
      );

      if (!cancelled) {
        if (Object.keys(statsAccumulator).length > 0) {
          setModelStats((prev) => ({ ...prev, ...statsAccumulator }));
        }
        if (Object.keys(seriesAccumulator).length > 0) {
          seriesRef.current = { ...seriesRef.current, ...seriesAccumulator };
          setPerformanceData(buildChartData(seriesRef.current));
        }
        setLoading(false);
      }
    };

    hydrateFromCache();
    fetchAllModels();
    const interval = setInterval(fetchAllModels, 60_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [getStats, getPnlSeries, updateStats, updatePnlSeries]);

  const activeModel = useMemo(
    () => AI_MODELS.find((model) => model.config.signature === activeSignature) || AI_MODELS[0],
    [activeSignature]
  );
  const activeStats = activeModel ? modelStats[activeModel.config.signature] : undefined;
  const activeReturn = activeStats?.total_return_pct ?? null;

  return (
    <div className="p-6 space-y-6">
      {/* Trading Control */}
      <TradingControl />

      {/* Performance Chart */}
      <div className="glass-card rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-white text-xl mb-1">AI模型收益对比</h2>
            <p className="text-sm text-gray-400">实时收益率走势</p>
          </div>
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-green-500" />
            <span className="text-white">实时更新</span>
            {activeStats && (
              <span className="text-sm text-gray-400 ml-2">
                当前模型 Sharpe: {activeStats.sharpe_ratio?.toFixed(2) ?? 'N/A'}
              </span>
            )}
            {activeStats?.valuation_source && (
              <span className="text-xs text-gray-400 border border-gray-700 rounded px-2 py-0.5">
                估值:
                <span className="ml-1 text-white">{formatValuationSource(activeStats.valuation_source)}</span>
              </span>
            )}
          </div>
        </div>

        <div className="h-[320px]">
          {performanceData.length === 0 && loading ? (
            <div className="flex items-center justify-center h-full text-gray-400 text-sm">
              正在加载模型收益曲线...
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={performanceData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="time" 
              stroke="#9CA3AF"
              style={{ fontSize: '12px' }}
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis 
              stroke="#9CA3AF"
              style={{ fontSize: '12px' }}
                  tickFormatter={(value) => `${Number(value).toFixed(1)}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#fff',
              }}
                  formatter={(value: any, key: string) => {
                    const model = AI_MODELS.find((m) => m.id === key);
                    return [`${Number(value).toFixed(2)}%`, model?.name ?? key];
                  }}
            />
                <Legend wrapperStyle={{ paddingTop: '12px' }} />
                {AI_MODELS.map((model) => (
                  <Line
                    key={model.id}
              type="monotone"
                    dataKey={model.id}
                    stroke={model.color}
              strokeWidth={2}
                    dot={false}
                    name={model.name}
            />
                ))}
              </LineChart>
        </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* AI Model Cards Grid */}
      <div>
        <h2 className="text-white text-xl mb-4">AI模型监控面板</h2>
        <div className="grid grid-cols-3 gap-4">
          {AI_MODELS.map((model) => {
            const stats = modelStats[model.config.signature];
            const displayModel: AIModel = stats
              ? {
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
                }
              : model;
            
            return (
              <AIModelCard
                key={model.id}
                model={displayModel}
                onClick={() => {
                  onActiveSignatureChange?.(model.config.signature);
                  onAIModelClick(displayModel);
                }}
              />
            );
          })}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="glass-card rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-2">当前收益率</div>
          <div
            className={`text-2xl ${
              activeReturn !== null && (activeReturn ?? 0) >= 0 ? 'text-green-500' : 'text-red-500'
            }`}
          >
            {activeReturn !== null
              ? `${activeReturn >= 0 ? '+' : ''}${activeReturn.toFixed(2)}%`
              : 'N/A'}
          </div>
        </div>
        <div className="glass-card rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-2">Sharpe比率</div>
          <div className="text-2xl text-white">
            {activeStats?.sharpe_ratio !== undefined ? activeStats.sharpe_ratio.toFixed(2) : 'N/A'}
          </div>
        </div>
        <div className="glass-card rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-2">最大回撤</div>
          <div className="text-2xl text-red-400">
            {activeStats?.max_drawdown_pct !== undefined
              ? `${activeStats.max_drawdown_pct.toFixed(2)}%`
              : 'N/A'}
          </div>
        </div>
        <div className="glass-card rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-2">持仓数量</div>
          <div className="text-2xl text-white">{activeStats?.position_count ?? 0}</div>
        </div>
      </div>
    </div>
  );
}
