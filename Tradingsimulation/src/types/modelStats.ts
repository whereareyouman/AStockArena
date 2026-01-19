export interface ModelStatsResponse {
  total_return_pct?: number;
  sharpe_ratio?: number;
  max_drawdown_pct?: number;
  position_count?: number;
  total_records?: number;
  trade_count?: number;
  cash?: number;
  equity?: number;
  last_action?: Record<string, unknown>;
  latest_date?: string;
  signature?: string;
  holdings?: Array<Record<string, unknown>>;
  valuation_source?: string;
}

