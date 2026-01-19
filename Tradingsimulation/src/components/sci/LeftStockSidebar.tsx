import React, { useEffect, useMemo, useState } from 'react';
import { TrendingUp, TrendingDown, Search } from 'lucide-react';
import { DEFAULT_STOCK_SYMBOLS, getSectorColor, type Stock } from '../../utils/stockData';
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';

interface LeftStockSidebarProps {
  onStockSelect: (stock: Stock) => void;
  selectedStock?: Stock | null;
}

export function LeftStockSidebar({ onStockSelect, selectedStock }: LeftStockSidebarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [quotes, setQuotes] = useState<Record<string, Partial<Stock>>>({});

  // periodically fetch live quotes and overlay onto defaults. This is a test change to see if Git detects it.
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
            aiAttention: Number(q.aiAttentionScore) || 0,
          };
        });
        if (!cancelled) setQuotes(map);
      } catch (e) {
        // ignore errors, keep previous
      }
    };
    fetchQuotes();
    const t = setInterval(fetchQuotes, 30000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  const mergedStocks = useMemo(() => {
    return DEFAULT_STOCK_SYMBOLS.map((s) => ({
      ...s,
      ...(quotes[s.code] || {}),
    }));
  }, [quotes]);

  // 按板块分组
  const semiconductorStocks = mergedStocks.filter(s => s.sector === 'semiconductor');
  const solarStocks = mergedStocks.filter(s => s.sector === 'solar');
  const techStocks = mergedStocks.filter(s => s.sector === 'tech');

  const filteredStocks = (stocks: Stock[]) => {
    if (!searchQuery) return stocks;
    return stocks.filter(s => 
      s.code.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.name.includes(searchQuery)
    );
  };

  const renderStockItem = (stock: Stock) => {
    const isSelected = selectedStock?.code === stock.code;
    const liveQuote = quotes[stock.code];
    const price = typeof liveQuote?.price === 'number' && liveQuote.price > 0 ? liveQuote.price : undefined;
    const changePercent = typeof liveQuote?.changePercent === 'number' ? liveQuote.changePercent : undefined;
    const aiFocusRaw =
      typeof liveQuote?.aiHoldingCount === 'number'
        ? liveQuote.aiHoldingCount
        : typeof liveQuote?.aiAttention === 'number'
        ? liveQuote.aiAttention
        : undefined;
    const aiTrades = typeof liveQuote?.aiTradeVolume === 'number' ? liveQuote.aiTradeVolume : undefined;
    const aiFocus = typeof aiFocusRaw === 'number' ? aiFocusRaw : undefined;
    const attentionClass = typeof aiFocus === 'number' ? 'text-[#1E40AF]' : 'text-gray-500';
    const aiVolumeClass = typeof aiTrades === 'number' && aiTrades > 0 ? 'text-blue-200' : 'text-gray-500';
    const isHot = typeof aiFocus === 'number' && aiFocus >= 3;
    const displayedPrice = typeof price === 'number' ? price : stock.price;
    const displayedChange =
      typeof changePercent === 'number' ? changePercent : stock.changePercent;
    const isPositive = typeof displayedChange === 'number' ? displayedChange >= 0 : false;

    const formatVolume = (value: number) => {
      if (!value) return '0';
      if (value >= 10000) return `${(value / 10000).toFixed(0)}万`;
      return value.toFixed(0);
    };

    const aiFocusDisplay = typeof aiFocus === 'number' ? aiFocus : '--';

    return (
      <div
        key={stock.code}
        onClick={() => onStockSelect(stock)}
        className={`p-3 cursor-pointer transition-all border-l-2 ${
          isSelected 
            ? 'bg-[#1E40AF] bg-opacity-20 border-l-[#1E40AF]' 
            : 'border-l-transparent hover:bg-gray-700 hover:bg-opacity-30'
        }`}
      >
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <span className="text-white text-sm">{stock.name}</span>
            <span className="text-gray-400 text-xs">{stock.code}</span>
          </div>
          {isHot && (
            <Badge className="bg-[#F59E0B] text-white text-xs px-1 py-0">热</Badge>
          )}
        </div>
        
        <div className="flex items-center justify-between">
          <span className="text-white">¥{displayedPrice.toFixed(2)}</span>
          <div className={`flex items-center gap-1 text-sm ${isPositive ? 'price-up' : 'price-down'}`}>
            {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            <span>{isPositive ? '+' : ''}{displayedChange.toFixed(2)}%</span>
          </div>
        </div>
        <div className={`flex items-center justify-between mt-1 text-xs ${aiVolumeClass}`}>
          <span>AI成交:{formatVolume(aiTrades)}股</span>
        </div>
        <div className={`mt-1 text-xs ${attentionClass}`}>
          AI关注: {aiFocusDisplay}
        </div>
      </div>
    );
  };

  const renderSectorGroup = (title: string, stocks: Stock[], color: string) => {
    const filtered = filteredStocks(stocks);
    if (filtered.length === 0 && searchQuery) return null;

    return (
      <div className="mb-4">
        <div 
          className="px-3 py-2 flex items-center gap-2 border-b border-gray-700"
          style={{ borderLeftColor: color, borderLeftWidth: '3px' }}
        >
          <div 
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: color }}
          />
          <span className="text-gray-300 text-sm">{title}</span>
          <span className="text-gray-500 text-xs">({filtered.length})</span>
        </div>
        <div className="space-y-0.5">
          {filtered.map(renderStockItem)}
        </div>
      </div>
    );
  };

  return (
    <div className="w-80 bg-[#111827] border-r border-gray-700 h-screen flex flex-col overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-white mb-3">科创板股票池</h3>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="搜索股票代码或名称..."
            className="pl-10 bg-gray-700 border-gray-600 text-white text-sm"
          />
        </div>
      </div>

      {/* Stock List */}
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        {renderSectorGroup('半导体板块', semiconductorStocks, getSectorColor('semiconductor'))}
        {renderSectorGroup('光伏板块', solarStocks, getSectorColor('solar'))}
        {renderSectorGroup('科技其他', techStocks, getSectorColor('tech'))}
      </div>

      {/* Footer Stats */}
      <div className="p-4 border-t border-gray-700 bg-gray-800">
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div>
            <div className="text-gray-400 mb-1">上涨</div>
            <div className="price-up text-lg">
              {mergedStocks.filter(s => s.changePercent > 0).length}只
            </div>
          </div>
          <div>
            <div className="text-gray-400 mb-1">下跌</div>
            <div className="price-down text-lg">
              {mergedStocks.filter(s => s.changePercent < 0).length}只
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

