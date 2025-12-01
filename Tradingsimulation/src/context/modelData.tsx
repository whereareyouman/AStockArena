import { createContext, useContext, useMemo, useState, ReactNode } from 'react';
import type { ModelStatsResponse } from '../types/modelStats';

type CachedEntry<T> = {
  data: T;
  fetchedAt: number;
};

type StatsCache = Record<string, CachedEntry<ModelStatsResponse>>;
type SeriesCache = Record<string, CachedEntry<any[]>>;

interface ModelDataContextValue {
  getStats(signature: string): ModelStatsResponse | undefined;
  getPnlSeries(signature: string): any[] | undefined;
  updateStats(signature: string, stats: ModelStatsResponse): void;
  updatePnlSeries(signature: string, series: any[]): void;
  clearCache(): void;
}

const STALE_MS = 45_000;

const ModelDataContext = createContext<ModelDataContextValue | undefined>(undefined);

const getEntryData = <T,>(entry: CachedEntry<T> | undefined): T | undefined => {
  if (!entry) return undefined;
  if (Date.now() - entry.fetchedAt > STALE_MS) {
    return undefined;
  }
  return entry.data;
};

export const ModelDataProvider = ({ children }: { children: ReactNode }) => {
  const [statsCache, setStatsCache] = useState<StatsCache>({});
  const [seriesCache, setSeriesCache] = useState<SeriesCache>({});

  const value = useMemo<ModelDataContextValue>(
    () => ({
      getStats(signature) {
        return getEntryData(statsCache[signature]);
      },
      getPnlSeries(signature) {
        return getEntryData(seriesCache[signature]);
      },
      updateStats(signature, stats) {
        setStatsCache((prev) => ({
          ...prev,
          [signature]: { data: stats, fetchedAt: Date.now() },
        }));
      },
      updatePnlSeries(signature, series) {
        setSeriesCache((prev) => ({
          ...prev,
          [signature]: { data: series, fetchedAt: Date.now() },
        }));
      },
      clearCache() {
        setStatsCache({});
        setSeriesCache({});
      },
    }),
    [statsCache, seriesCache]
  );

  return <ModelDataContext.Provider value={value}>{children}</ModelDataContext.Provider>;
};

export const useModelDataCache = () => {
  const ctx = useContext(ModelDataContext);
  if (!ctx) {
    throw new Error('useModelDataCache must be used within ModelDataProvider');
  }
  return ctx;
};

