import React, { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Badge } from '../ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../ui/table';
import { TrendingDown, Activity, Target, Brain, DollarSign, AlertTriangle } from 'lucide-react';
import type { AIModel } from '../../utils/stockData';
import type { ModelStatsResponse } from '../../types/modelStats';
import { useModelDataCache } from '../../context/modelData';

interface AIModelDetailViewProps {
  model: AIModel;
  onBack: () => void;
}

interface PositionItem {
  symbol: string;
  shares: number;
  purchase_date?: string;
  avg_price?: number;
  entry_price?: number;
  current_price?: number;
  market_value?: number;
  pnl?: number;
  pnl_percent?: number;
}

interface DecisionItem {
  date?: string;
  time?: string;
  action?: string;
  symbol?: string;
  amount?: number;
  cash?: number;
  holdings?: number;
  id?: number;
}

const PIE_COLORS = ['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B', '#EC4899', '#14B8A6', '#F97316', '#06B6D4'];

const describeValuationSource = (source?: string) => {
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

export function AIModelDetailView({ model, onBack }: AIModelDetailViewProps) {
  const [stats, setStats] = useState<ModelStatsResponse | null>(null);
  const [equityCurve, setEquityCurve] = useState<Array<{ day: string; equity: number }>>([]);
  const [positionData, setPositionData] = useState<PositionItem[]>([]);
  const [positionMeta, setPositionMeta] =
    useState<{ cash: number; totalEquity: number; valuationSource?: string }>({
    cash: 0,
    totalEquity: 0,
  });
  const [tradeHistory, setTradeHistory] = useState<DecisionItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { getStats, getPnlSeries, updateStats, updatePnlSeries } = useModelDataCache();

  useEffect(() => {
    let cancelled = false;
    const signature = model.config.signature;

    const hydrateFromCache = () => {
      const cachedStats = getStats(signature);
      if (cachedStats) {
        setStats(cachedStats);
      }
      const cachedSeries = getPnlSeries(signature);
      if (cachedSeries) {
        const series = cachedSeries.map((item: any) => ({
          day: item.date,
          equity: Number(item.returnPct ?? 0),
        }));
        setEquityCurve(series);
      }
    };

    const fetchDetailData = async () => {
      setLoading(true);
      setError(null);
      try {
        const [statsRes, pnlRes, positionsRes, decisionsRes] = await Promise.all([
          fetch(`http://localhost:8000/api/live/model-stats?signature=${signature}`),
          fetch(`http://localhost:8000/api/live/pnl-series?signature=${signature}&days=60&valuation=equity`),
          fetch(`http://localhost:8000/api/live/current-positions?signature=${signature}`),
          fetch(`http://localhost:8000/api/live/recent-decisions?signature=${signature}&limit=30`),
        ]);

        if (cancelled) {
          return;
        }

        if (statsRes.ok) {
          const payload = await statsRes.json();
          setStats(payload);
          updateStats(signature, payload);
        } else {
          setStats(null);
        }

        if (pnlRes.ok) {
          const payload = await pnlRes.json();
          const items = Array.isArray(payload.items) ? payload.items : [];
          const series = items.map((item: any) => ({
            day: item.date,
            equity: Number(item.returnPct ?? 0),
          }));
          setEquityCurve(series);
          updatePnlSeries(signature, items);
        } else {
          setEquityCurve([]);
        }

        if (positionsRes.ok) {
          const payload = await positionsRes.json();
          const items: PositionItem[] = Array.isArray(payload.positions) ? payload.positions : [];
          const normalized = items.map((pos) => {
            const shares = pos.shares ?? 0;
            const entryPrice = pos.avg_price ?? pos.entry_price ?? 0;
            const currentPrice = pos.current_price ?? entryPrice;
            const marketValue = pos.market_value ?? currentPrice * shares;
            return {
              ...pos,
              shares,
              current_price: currentPrice,
              avg_price: entryPrice,
              market_value: marketValue,
            };
          });
          setPositionData(normalized);
          setPositionMeta({
            cash: Number(payload.cash ?? 0),
            totalEquity: Number(payload.total_equity ?? 0),
            valuationSource: payload.valuation_source,
          });
        } else {
          setPositionData([]);
          setPositionMeta({ cash: 0, totalEquity: 0, valuationSource: undefined });
        }

        if (decisionsRes.ok) {
          const payload = await decisionsRes.json();
          setTradeHistory(Array.isArray(payload.decisions) ? payload.decisions : []);
        } else {
          setTradeHistory([]);
        }
      } catch (err) {
        console.warn('Failed to fetch model detail data:', err);
        if (!cancelled) {
          setError('暂时无法获取最新数据，请稍后重试。');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    hydrateFromCache();
    fetchDetailData();
    const interval = setInterval(fetchDetailData, 60_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [getPnlSeries, getStats, model, updatePnlSeries, updateStats]);

  const latestAction = stats?.last_action
    ? JSON.stringify(stats.last_action, null, 2)
    : '暂无最新动作';

  return (
    <div className="p-6 space-y-6">
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
              <h1 className="text-white text-3xl">{model.name}</h1>
              <Badge className="text-white" style={{ backgroundColor: model.color }}>
                {model.config.baseModel}
              </Badge>
              <Badge
                className={
                  model.status === 'active'
                    ? 'bg-green-600'
                    : model.status === 'deciding'
                    ? 'bg-yellow-600'
                    : 'bg-gray-600'
                }
              >
                {model.status === 'active' ? '运行中' : model.status === 'deciding' ? '决策中' : '空闲'}
              </Badge>
            </div>
            <p className="text-gray-400">
              {model.config.signature} · 风险等级:
              {model.config.riskLevel === 'high' ? ' 高' : model.config.riskLevel === 'medium' ? ' 中' : ' 低'}
            </p>
          </div>
        </div>

        <div>
          <div className="glass-card rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-1">总收益率</div>
            <div className={`text-3xl ${(stats?.total_return_pct ?? 0) >= 0 ? 'price-up' : 'price-down'}`}>
              {stats?.total_return_pct !== undefined
                ? `${stats.total_return_pct >= 0 ? '+' : ''}${stats.total_return_pct.toFixed(2)}%`
                : 'N/A'}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-5 gap-4">
        <div className="glass-card rounded-lg p-4">
          <div className="flex items-center gap-2 text-gray-400 mb-2">
            <Target className="w-4 h-4" />
            <span className="text-sm">夏普比率</span>
          </div>
          <div className="text-2xl text-white">
            {stats?.sharpe_ratio !== undefined ? stats.sharpe_ratio.toFixed(2) : 'N/A'}
          </div>
        </div>
        <div className="glass-card rounded-lg p-4">
          <div className="flex items-center gap-2 text-gray-400 mb-2">
            <TrendingDown className="w-4 h-4" />
            <span className="text-sm">最大回撤</span>
          </div>
          <div className="text-2xl price-down">
            {stats?.max_drawdown_pct !== undefined ? `${stats.max_drawdown_pct.toFixed(2)}%` : 'N/A'}
          </div>
        </div>
        <div className="glass-card rounded-lg p-4">
          <div className="flex items-center gap-2 text-gray-400 mb-2">
            <Activity className="w-4 h-4" />
            <span className="text-sm">总交易</span>
          </div>
          <div className="text-2xl text-white">{stats?.total_records ?? '—'}</div>
        </div>
        <div className="glass-card rounded-lg p-4">
          <div className="flex items-center gap-2 text-gray-400 mb-2">
            <DollarSign className="w-4 h-4" />
            <span className="text-sm">当前现金</span>
          </div>
          <div className="text-2xl text-white">
            {stats?.cash !== undefined ? `¥${stats.cash.toLocaleString()}` : 'N/A'}
          </div>
        </div>
        <div className="glass-card rounded-lg p-4">
          <div className="flex items-center gap-2 text-gray-400 mb-2">
            <Brain className="w-4 h-4" />
            <span className="text-sm">持仓数</span>
          </div>
          <div className="text-2xl text-white">{stats?.position_count ?? 0} / 10</div>
        </div>
      </div>

      {(stats?.valuation_source || positionMeta.valuationSource) && (
        <div className="glass-card rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-2">估值来源</div>
          <div className="text-xl text-white">
            {describeValuationSource(positionMeta.valuationSource || stats?.valuation_source)}
          </div>
        </div>
      )}

      <Tabs defaultValue="equity" className="space-y-4">
        <TabsList className="glass-card">
          <TabsTrigger value="equity">权益曲线</TabsTrigger>
          <TabsTrigger value="positions">持仓明细</TabsTrigger>
          <TabsTrigger value="trades">交易历史</TabsTrigger>
          <TabsTrigger value="analysis">绩效分析</TabsTrigger>
        </TabsList>

        <TabsContent value="equity">
          <div className="glass-card rounded-lg p-6">
            <h3 className="text-white mb-4">权益曲线</h3>
            {equityCurve.length === 0 ? (
              <div className="text-gray-400 text-sm">暂无权益数据。</div>
            ) : (
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={equityCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="day" stroke="#9CA3AF" style={{ fontSize: '12px' }} />
                <YAxis 
                  stroke="#9CA3AF"
                  style={{ fontSize: '12px' }}
                    tickFormatter={(value) => `${value.toFixed(1)}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
                    formatter={(value: any) => [`${Number(value).toFixed(2)}%`, '收益率']}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="equity" 
                  stroke={model.color}
                  strokeWidth={3}
                  dot={false}
                    name="模型收益"
                />
              </LineChart>
            </ResponsiveContainer>
            )}
          </div>
        </TabsContent>

        <TabsContent value="positions">
          <div className="grid grid-cols-2 gap-4">
            <div className="glass-card rounded-lg p-6">
              <h3 className="text-white mb-4">持仓分布</h3>
              {positionData.length === 0 ? (
                <div className="text-gray-400 text-sm">暂无持仓。</div>
              ) : (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                      data={positionData.map((pos) => ({
                        name: pos.symbol,
                        value: pos.market_value ?? 0,
                      }))}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                      {positionData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#fff',
                    }}
                      formatter={(value: any) => [`¥${Number(value).toLocaleString()}`, '市值']}
                  />
                </PieChart>
              </ResponsiveContainer>
              )}
            </div>

            <div className="glass-card rounded-lg p-6">
              <h3 className="text-white mb-4">持仓详情</h3>
              {positionData.length === 0 ? (
                <div className="text-gray-400 text-sm">暂无持仓记录。</div>
              ) : (
              <div className="space-y-2 max-h-[300px] overflow-y-auto scrollbar-thin">
                {positionData.map((pos, idx) => {
                    const cost = pos.avg_price ?? 0;
                    const current = pos.current_price ?? cost;
                    const profit = cost > 0 ? ((current - cost) / cost) * 100 : 0;

                  return (
                      <div key={`${pos.symbol}-${idx}`} className="p-3 bg-gray-700 bg-opacity-30 rounded">
                      <div className="flex items-center justify-between mb-1">
                          <span className="text-white">{pos.symbol}</span>
                        <span className={profit >= 0 ? 'price-up' : 'price-down'}>
                            {profit >= 0 ? '+' : ''}
                            {profit.toFixed(2)}%
                        </span>
                      </div>
                      <div className="text-xs text-gray-400">
                          持仓: {pos.shares.toLocaleString()}股 · 成本: ¥{cost.toFixed(2)} · 现价: ¥{current.toFixed(2)}
                      </div>
                    </div>
                  );
                })}
                </div>
              )}
              <div className="mt-3 text-xs text-gray-400 space-y-1">
                <div>现金: ¥{positionMeta.cash.toLocaleString()} · 总权益: ¥{positionMeta.totalEquity.toLocaleString()}</div>
                <div>
                  估值来源：{describeValuationSource(positionMeta.valuationSource || stats?.valuation_source)}
                </div>
              </div>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="trades">
          <div className="glass-card rounded-lg p-6">
            <h3 className="text-white mb-4">交易历史记录</h3>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="border-gray-700">
                    <TableHead className="text-gray-400">日期</TableHead>
                    <TableHead className="text-gray-400">股票</TableHead>
                    <TableHead className="text-gray-400">操作</TableHead>
                    <TableHead className="text-gray-400">数量</TableHead>
                    <TableHead className="text-gray-400">现金</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {tradeHistory.length === 0 ? (
                    <TableRow className="border-gray-700">
                      <TableCell colSpan={5} className="text-center text-gray-400">
                        暂无交易记录
                      </TableCell>
                    </TableRow>
                  ) : (
                    tradeHistory.map((trade, idx) => (
                      <TableRow key={`${trade.id}-${idx}`} className="border-gray-700">
                      <TableCell className="text-white">
                          {trade.date} {trade.time?.split(' ')[1] ?? ''}
                      </TableCell>
                        <TableCell className="text-white">{trade.symbol ?? '-'}</TableCell>
                      <TableCell>
                        <Badge className={trade.action === 'buy' ? 'bg-red-600' : 'bg-green-600'}>
                            {trade.action === 'buy' ? '买入' : trade.action === 'sell' ? '卖出' : '观望'}
                        </Badge>
                      </TableCell>
                        <TableCell className="text-white">{trade.amount ?? '-'}</TableCell>
                        <TableCell className="text-white">
                          {trade.cash !== undefined ? `¥${trade.cash.toLocaleString()}` : '—'}
                        </TableCell>
                    </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="analysis">
          <div className="grid grid-cols-2 gap-4">
            <div className="glass-card rounded-lg p-6 space-y-3">
              <h3 className="text-white mb-2">资产概览</h3>
              <div className="flex items-center justify-between text-sm text-gray-400">
                <span>当前现金</span>
                <span className="text-white">¥{positionMeta.cash.toLocaleString()}</span>
            </div>
              <div className="flex items-center justify-between text-sm text-gray-400">
                <span>总权益</span>
                <span className="text-white">¥{positionMeta.totalEquity.toLocaleString()}</span>
                  </div>
              <div className="flex items-center justify-between text-sm text-gray-400">
                <span>最新交易</span>
                <span className="text-white">{stats?.last_action ? '见下方' : '暂无'}</span>
                  </div>
                </div>
            <div className="glass-card rounded-lg p-6 space-y-3">
              <h3 className="text-white mb-2">最新指令</h3>
              <div className="bg-gray-900/60 rounded p-3 text-xs text-gray-300 whitespace-pre-wrap">
                {latestAction}
                  </div>
              {error && (
                <div className="flex items-center gap-2 text-sm text-yellow-400">
                  <AlertTriangle className="w-4 h-4" />
                  {error}
                </div>
              )}
            </div>
          </div>
        </TabsContent>
      </Tabs>

      {loading && <div className="text-center text-gray-400 text-sm">正在更新数据...</div>}
    </div>
  );
}

