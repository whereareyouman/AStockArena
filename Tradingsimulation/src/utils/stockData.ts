// 科创板股票池数据结构

export interface Stock {
  code: string;
  name: string;
  sector: 'semiconductor' | 'solar' | 'tech';
  sectorName: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  turnover: number;
  aiAttention: number; // 默认 AI 关注度
  aiHoldingCount?: number;
  aiTradeVolume?: number;
  aiTurnover?: number;
}

export interface AIModel {
  id: string;
  name: string;
  color: string;
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  positionCount: number;
  winRate?: number;
  totalTrades?: number;
  status: 'active' | 'idle' | 'deciding';
  config: {
    baseModel: string;
    signature: string;
    riskLevel: 'low' | 'medium' | 'high';
  };
}

export interface Decision {
  timestamp: Date;
  aiModelId: string;
  stockCode: string;
  action: 'buy' | 'sell' | 'hold';
  confidence: number;
  reasoning: string;
  price: number;
}

// 股票池配置
export const DEFAULT_STOCK_SYMBOLS: Stock[] = [
  {
    code: 'SH688008',
    name: '澜起科技',
    sector: 'semiconductor',
    sectorName: '半导体',
    price: 0,
    change: 0,
    changePercent: 0,
    volume: 0,
    turnover: 0,
    aiAttention: 0,
  },
  {
    code: 'SH688111',
    name: '金山办公',
    sector: 'tech',
    sectorName: '软件服务',
    price: 0,
    change: 0,
    changePercent: 0,
    volume: 0,
    turnover: 0,
    aiAttention: 0,
  },
  {
    code: 'SH688009',
    name: '中国通号',
    sector: 'tech',
    sectorName: '轨道交通',
    price: 0,
    change: 0,
    changePercent: 0,
    volume: 0,
    turnover: 0,
    aiAttention: 0,
  },
  {
    code: 'SH688981',
    name: '中芯国际',
    sector: 'semiconductor',
    sectorName: '芯片制造',
    price: 0,
    change: 0,
    changePercent: 0,
    volume: 0,
    turnover: 0,
    aiAttention: 0,
  },
  {
    code: 'SH688256',
    name: '寒武纪',
    sector: 'semiconductor',
    sectorName: 'AI芯片',
    price: 0,
    change: 0,
    changePercent: 0,
    volume: 0,
    turnover: 0,
    aiAttention: 0,
  },
  {
    code: 'SH688271',
    name: '联影医疗',
    sector: 'tech',
    sectorName: '医疗设备',
    price: 0,
    change: 0,
    changePercent: 0,
    volume: 0,
    turnover: 0,
    aiAttention: 0,
  },
  {
    code: 'SH688047',
    name: '龙芯中科',
    sector: 'semiconductor',
    sectorName: 'CPU芯片',
    price: 0,
    change: 0,
    changePercent: 0,
    volume: 0,
    turnover: 0,
    aiAttention: 0,
  },
  {
    code: 'SH688617',
    name: '惠泰医疗',
    sector: 'tech',
    sectorName: '医疗器械',
    price: 0,
    change: 0,
    changePercent: 0,
    volume: 0,
    turnover: 0,
    aiAttention: 0,
  },
  {
    code: 'SH688303',
    name: '大全能源',
    sector: 'solar',
    sectorName: '光伏材料',
    price: 0,
    change: 0,
    changePercent: 0,
    volume: 0,
    turnover: 0,
    aiAttention: 0,
  },
  {
    code: 'SH688180',
    name: '君实生物',
    sector: 'tech',
    sectorName: '生物医药',
    price: 0,
    change: 0,
    changePercent: 0,
    volume: 0,
    turnover: 0,
    aiAttention: 0,
  },
];

// AI模型配置
export const AI_MODELS: AIModel[] = [
  {
    id: 'ai-gemini',
    name: 'Gemini 2.5 Flash',
    color: '#60A5FA',
    totalReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    positionCount: 0,
    totalTrades: 0,
    status: 'active',
    config: {
      baseModel: 'Gemini-2.5-Flash',
      signature: 'gemini-2.5-flash',
      riskLevel: 'medium',
    },
  },
  {
    id: 'ai-qwen',
    name: 'Qwen3 235B',
    color: '#A855F7',
    totalReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    positionCount: 0,
    totalTrades: 0,
    status: 'active',
    config: {
      baseModel: 'Qwen3-235B-A22B',
      signature: 'qwen3-235b',
      riskLevel: 'high',
    },
  },
  {
    id: 'ai-gpt5',
    name: 'GPT-5.1',
    color: '#3B82F6',
    totalReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    positionCount: 0,
    totalTrades: 0,
    status: 'active',
    config: {
      baseModel: 'GPT-5.1',
      signature: 'gpt-5.1',
      riskLevel: 'high',
    },
  },
  {
    id: 'ai-claude',
    name: 'Claude Haiku 4.5',
    color: '#F97316',
    totalReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    positionCount: 0,
    totalTrades: 0,
    status: 'active',
    config: {
      baseModel: 'Claude-Haiku-4.5',
      signature: 'claude-haiku-4-5',
      riskLevel: 'medium',
    },
  },
  {
    id: 'ai-deepseek',
    name: 'DeepSeek Reasoner',
    color: '#10B981',
    totalReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    positionCount: 0,
    totalTrades: 0,
    status: 'active',
    config: {
      baseModel: 'DeepSeek-Reasoner',
      signature: 'deepseek-reasoner',
      riskLevel: 'medium',
    },
  },
];

// 获取板块股票
export const getStocksBySector = (sector: Stock['sector']) => {
  return DEFAULT_STOCK_SYMBOLS.filter(stock => stock.sector === sector);
};

// 获取板块名称
export const getSectorName = (sector: Stock['sector']) => {
  const names = {
    semiconductor: '半导体板块',
    solar: '光伏板块',
    tech: '科技其他',
  };
  return names[sector];
};

// 获取板块颜色
export const getSectorColor = (sector: Stock['sector']) => {
  const colors = {
    semiconductor: '#3B82F6',
    solar: '#10B981',
    tech: '#8B5CF6',
  };
  return colors[sector];
};

// 生成模拟决策数据
export const generateMockDecisions = (count: number = 20): Decision[] => {
  const decisions: Decision[] = [];
  const now = new Date();
  
  for (let i = 0; i < count; i++) {
    const stock = DEFAULT_STOCK_SYMBOLS[Math.floor(Math.random() * DEFAULT_STOCK_SYMBOLS.length)];
    const aiModel = AI_MODELS[Math.floor(Math.random() * AI_MODELS.length)];
    const actions: Decision['action'][] = ['buy', 'sell', 'hold'];
    
    decisions.push({
      timestamp: new Date(now.getTime() - i * 3600000), // 每小时一个决策
      aiModelId: aiModel.id,
      stockCode: stock.code,
      action: actions[Math.floor(Math.random() * actions.length)],
      confidence: 60 + Math.random() * 40,
      reasoning: '基于技术指标分析，MACD金叉信号出现，RSI处于超卖区域，建议买入',
      price: stock.price,
    });
  }
  
  return decisions.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
};
