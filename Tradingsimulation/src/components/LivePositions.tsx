import React, { useEffect, useMemo, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

interface LineItem {
  date?: string;
  id?: number;
  cash?: number;
  positions_count?: number;
  action?: string;
  symbol?: string;
  amount?: number;
}

export default function LivePositions() {
  const [data, setData] = useState<LineItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function fetchData() {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch('http://localhost:8000/api/live/position-lines?limit=100');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json.items || []);
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchData();
    const t = setInterval(fetchData, 3000);
    return () => clearInterval(t);
  }, []);

  const chartData = useMemo(() => {
    return (data || []).map((d) => ({
      name: d.date || String(d.id ?? ''),
      cash: d.cash ?? null,
      count: d.positions_count ?? 0,
    }));
  }, [data]);

  return (
    <div style={{ padding: 12, border: '1px solid #eee', borderRadius: 8, marginTop: 12 }}>
      <h3 style={{ marginBottom: 8 }}>Live Positions (Cash & Count)</h3>
      {loading && <p>Loading...</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      <div style={{ width: '100%', height: 260 }}>
        <ResponsiveContainer>
          <LineChart data={chartData} margin={{ left: 12, right: 12, top: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2f3b51" />
            <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <YAxis yAxisId="left" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <YAxis yAxisId="right" orientation="right" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <Tooltip contentStyle={{ background: '#0b1220', border: '1px solid #334155', color: '#e2e8f0' }} />
            <Line yAxisId="left" type="monotone" dataKey="cash" stroke="#60a5fa" strokeWidth={2} dot={false} name="Cash" />
            <Line yAxisId="right" type="monotone" dataKey="count" stroke="#34d399" strokeWidth={2} dot={false} name="# Holdings" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Recent action summary */}
      {data && data.length > 0 && (
        <div style={{ marginTop: 8, color: '#cbd5e1' }}>
          <strong>Latest:</strong>{' '}
          {data[data.length - 1].action || 'no_trade'} {data[data.length - 1].symbol || ''} {data[data.length - 1].amount || ''}
        </div>
      )}
    </div>
  );
}
