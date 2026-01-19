import React, { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceDot } from 'recharts';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Badge } from '../ui/badge';
import { TrendingUp, TrendingDown, Activity, BarChart3, AlertCircle } from 'lucide-react';
import type { Stock } from '../../utils/stockData';
import { AI_MODELS } from '../../utils/stockData';

interface StockDetailViewProps {
  stock: Stock;
  onBack: () => void;
}

interface HourlyPoint {
  time: string;
  price: number;
  volume?: number;
  amount?: number;
}

interface AIPostion {
  signature: string;
  shares: number;
  avg_price?: number;
  mark_price?: number;
  market_value?: number;
  pnl?: number;
  pnl_percent?: number;
  valuation_source?: string;
}

interface AITrade {
  signature: string;
  date?: string;
  decision_time?: string;
  action?: string;
  amount?: number;
  cash?: number;
  id?: number;
  price?: number;
}

interface AISummary {
  holding_count?: number;
  trade_volume?: number;
  turnover_percent?: number;
  holding_models?: string[];
}

interface StockDetailPayload {
  summary?: Record<string, any>;
  hourly_prices?: HourlyPoint[];
  indicators?: Record<string, any>;
  ai_positions?: AIPostion[];
  ai_trades?: AITrade[];
  ai_summary?: AISummary;
  news?: Array<{
    title: string;
    publish_time?: string;
    source?: string;
    url?: string;
  }>;
}

export function StockDetailView({ stock, onBack }: StockDetailViewProps) {
  const [priceData, setPriceData] = useState<Array<any>>([]);
  const [aiTradeMarks, setAiTradeMarks] = useState<Array<any>>([]);
  const [aiSummary, setAiSummary] = useState<AISummary | null>(null);
  const [detail, setDetail] = useState<StockDetailPayload | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const modelColorMap = useMemo(() => {
    const map = new Map<string, string>();
    AI_MODELS.forEach(model => {
      if (model.config?.signature) {
        map.set(model.config.signature, model.color);
      }
    });
    return map;
  }, []);

  const formatAxisLabel = (value?: string) => {
    if (!value) return '';
    if (value.includes(' ')) {
      const [datePart, timePart] = value.split(' ');
      const date = datePart?.slice(5) ?? datePart;
      const time = timePart?.slice(0, 5) ?? timePart;
      return `${date} ${time}`;
    }
    return value;
  };

  const formatAiVolume = (value?: number) => {
    if (!value) return '0';
    if (value >= 10000) return `${(value / 10000).toFixed(1)}万`;
    return value.toFixed(0);
  };

  useEffect(() => {
    let cancelled = false;
    const fetchDetail = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`http://localhost:8000/api/live/stock-detail?symbol=${stock.code}`);
        if (!res.ok) {
          throw new Error(await res.text());
        }
        const payload: StockDetailPayload = await res.json();
        if (cancelled) return;
        setDetail(payload);
        setAiSummary(payload.ai_summary ?? null);

        const hourly = (payload.hourly_prices || []).map((point: any) => {
          const timestamp = point.time || point.date;
          return {
            time: timestamp,
            price: Number(point.price ?? stock.price) || 0,
            volume: Number(point.volume ?? point.amount ?? 0) || 0,
          };
        });
        setPriceData(hourly);

        const marks = (payload.ai_trades || []).map((trade) => {
          const color = modelColorMap.get(trade.signature) || '#F59E0B';
          const price = Number(trade.price ?? payload.summary?.latest_price ?? stock.price) || stock.price;
          return {
            time: trade.decision_time || trade.date,
            price,
            action: trade.action,
            aiModel: {
              color,
              name: trade.signature,
            },
          };
        });
        setAiTradeMarks(marks);
      } catch (err) {
        console.warn('Failed to fetch stock detail:', err);
        if (!cancelled) {
          setError('暂时无法获取最新数据，请稍后再试。');
          setDetail(null);
          setPriceData([]);
          setAiTradeMarks([]);
          setAiSummary(null);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    fetchDetail();
    const interval = setInterval(fetchDetail, 60_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [modelColorMap, stock.code, stock.price]);

  const summary = detail?.summary;
  const isPositive = (summary?.change_percent ?? stock.changePercent) >= 0;
  const sectorColor = stock.sector === 'semiconductor' ? '#3B82F6' : 
                       stock.sector === 'solar' ? '#10B981' : '#8B5CF6';
  const attentionPercent = aiSummary?.attention_percent ?? stock.aiAttention ?? 0;

  const renderLoading = () => (
    <div className="p-6">
      <div className="glass-card rounded-lg p-6 flex items-center justify-center text-gray-400">
        加载中...
      </div>
    </div>
  );

  if (loading && !detail) {
    return renderLoading();
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={onBack}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-white transition-colors"
          >
            ← 返回
          </button>
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-white text-3xl">{stock.name}</h1>
              <span className="text-gray-400 text-xl">{stock.code}</span>
              <Badge style={{ backgroundColor: sectorColor }} className="text-white">
                {stock.sectorName}
              </Badge>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-white text-2xl">
                ¥{(summary?.latest_price ?? stock.price).toFixed(2)}
              </span>
              <div className={`flex items-center gap-2 text-xl ${isPositive ? 'price-up' : 'price-down'}`}>
                {isPositive ? <TrendingUp className="w-6 h-6" /> : <TrendingDown className="w-6 h-6" />}
                <span>{isPositive ? '+' : ''}{stock.change.toFixed(2)}</span>
                <span>({isPositive ? '+' : ''}{stock.changePercent.toFixed(2)}%)</span>
              </div>
            </div>
          </div>
        </div>

        <div className="glass-card rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">AI关注度</div>
          <div className="flex items-center gap-3">
            <div className="flex-1 w-32">
              <div className="h-2 bg-gray-700 rounded overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-[#1E40AF] to-[#3B82F6]"
                  style={{ width: `${Math.min(100, Math.max(0, attentionPercent ?? 0))}%` }}
                />
              </div>
            </div>
            <span className="text-2xl text-white">
              {attentionPercent !== undefined ? `${attentionPercent.toFixed(1)}%` : '--'}
            </span>
          </div>
        </div>
      </div>

      <Tabs defaultValue="chart" className="space-y-4">
        <TabsList className="glass-card">
          <TabsTrigger value="chart">价格走势</TabsTrigger>
          <TabsTrigger value="technical">技术指标</TabsTrigger>
          <TabsTrigger value="ai-positions">AI持仓</TabsTrigger>
          <TabsTrigger value="news">相关新闻</TabsTrigger>
        </TabsList>

        {/* Price Chart Tab */}
        <TabsContent value="chart" className="space-y-4">
          <div className="glass-card rounded-lg p-6">
            <h3 className="text-white mb-4">日内分时图（叠加AI买卖点）</h3>
            {error ? (
              <div className="flex items-center gap-2 text-red-400">
                <AlertCircle className="w-4 h-4" />
                {error}
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={priceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="time" 
                    stroke="#9CA3AF"
                    style={{ fontSize: '12px' }}
                  />
                  <YAxis 
                    stroke="#9CA3AF"
                    style={{ fontSize: '12px' }}
                    domain={['dataMin - 2', 'dataMax + 2']}
                    tickFormatter={(value) => `¥${Number(value).toFixed(2)}`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#fff',
                    }}
                    formatter={(value: any) => [`¥${Number(value).toFixed(2)}`, '价格']}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="price" 
                    stroke={isPositive ? '#EF4444' : '#10B981'}
                    strokeWidth={2}
                    dot={false}
                  />
                  {aiTradeMarks.map((mark, idx) => (
                    <ReferenceDot
                      key={`${mark.time}-${idx}`}
                      x={mark.time}
                      y={mark.price}
                      r={6}
                      fill={mark.aiModel.color}
                      stroke="#fff"
                      strokeWidth={2}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            )}

            {/* AI Legend */}
            <div className="flex items-center gap-4 mt-4 flex-wrap">
              {AI_MODELS.map(model => (
                <div key={model.id} className="flex items-center gap-2">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: model.color }}
                  />
                  <span className="text-sm text-gray-400">{model.name}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Volume Chart */}
          <div className="glass-card rounded-lg p-6">
            <h3 className="text-white mb-4">成交量</h3>
            <ResponsiveContainer width="100%" height={150}>
              <BarChart data={priceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="time" 
                  stroke="#9CA3AF"
                  style={{ fontSize: '12px' }}
                />
                <YAxis 
                  stroke="#9CA3AF"
                  style={{ fontSize: '12px' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
                />
                <Bar dataKey="volume" fill="#3B82F6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </TabsContent>

        {/* Technical Indicators Tab */}
        <TabsContent value="technical">
          <div className="grid grid-cols-2 gap-4">
            <div className="glass-card rounded-lg p-6">
              <h3 className="text-white mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                技术指标
              </h3>
              <div className="space-y-3">
                {detail?.indicators ? (
                  Object.entries(detail.indicators)
                    .slice(0, 6)
                    .map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between p-3 bg-gray-700 bg-opacity-30 rounded">
                        <span className="text-gray-400">{key}</span>
                        <span className="text-white">{String(value)}</span>
                      </div>
                    ))
                ) : (
                  <div className="text-gray-400 text-sm">暂无指标数据</div>
                )}
              </div>
            </div>

            <div className="glass-card rounded-lg p-6">
              <h3 className="text-white mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5" />
                市场数据
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-gray-700 bg-opacity-30 rounded">
                  <span className="text-gray-400">总市值</span>
                  <span className="text-white">523.8亿</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-gray-700 bg-opacity-30 rounded">
                  <span className="text-gray-400">流通市值</span>
                  <span className="text-white">421.6亿</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-gray-700 bg-opacity-30 rounded">
                  <span className="text-gray-400">市盈率TTM</span>
                  <span className="text-white">45.6</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-gray-700 bg-opacity-30 rounded">
                  <span className="text-gray-400">市净率</span>
                  <span className="text-white">6.8</span>
                </div>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* AI Positions Tab */}
        <TabsContent value="ai-positions">
          <div className="glass-card rounded-lg p-6">
            <h3 className="text-white mb-4">AI持仓情况</h3>
            <div className="space-y-3">
              {(detail?.ai_positions || []).length === 0 ? (
                <div className="text-gray-400 text-sm">暂无模型持仓</div>
              ) : (
                detail?.ai_positions?.map((pos) => {
                  const model = AI_MODELS.find(m => m.config.signature === pos.signature);
                  const color = model?.color || '#14B8A6';
                  return (
                    <div key={pos.signature} className="p-4 bg-gray-700 bg-opacity-30 rounded-lg">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div 
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: color }}
                          />
                          <span className="text-white">{model?.name || pos.signature}</span>
                        </div>
                        <Badge className="bg-blue-600 text-white">持仓中</Badge>
                      </div>
                      <div className="grid grid-cols-4 gap-4 text-sm">
                        <div>
                          <div className="text-gray-400 mb-1">持仓股数</div>
                          <div className="text-white">{pos.shares?.toLocaleString()}</div>
                        </div>
                        <div>
                          <div className="text-gray-400 mb-1">成本价</div>
                          <div className="text-white">¥{Number(pos.avg_price ?? 0).toFixed(2)}</div>
                        </div>
                        <div>
                          <div className="text-gray-400 mb-1">当前价</div>
                          <div className="text-white">¥{Number(pos.mark_price ?? summary?.latest_price ?? stock.price).toFixed(2)}</div>
                        </div>
                        <div>
                          <div className="text-gray-400 mb-1">盈亏</div>
                          <div className={(pos.pnl ?? 0) >= 0 ? 'price-up' : 'price-down'}>
                            {pos.pnl_percent ? `${pos.pnl_percent.toFixed(2)}%` : '—'}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </TabsContent>

        {/* News Tab */}
        <TabsContent value="news">
          <div className="glass-card rounded-lg p-6">
            <h3 className="text-white mb-4">相关新闻时间线</h3>
            <div className="space-y-4">
              {(detail?.news || []).length === 0 ? (
                <div className="text-gray-400 text-sm">暂无相关新闻</div>
              ) : (
                detail?.news?.map((news, idx) => (
                  <div key={`${news.title}-${idx}`} className="p-4 bg-gray-700 bg-opacity-30 rounded-lg hover:bg-opacity-50 transition-colors cursor-pointer">
                    <div className="flex items-start justify-between mb-2">
                      <h4 className="text-white flex-1">{news.title}</h4>
                      <Badge className="bg-green-600 text-white ml-3">{news.source || '资讯'}</Badge>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">{news.publish_time || ''}</span>
                      {news.url && (
                        <a 
                          className="text-blue-400 hover:underline"
                          href={news.url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          查看详情 →
                        </a>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
