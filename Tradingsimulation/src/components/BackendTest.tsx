import React, { useState } from 'react';

export default function BackendTest() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<any>(null);

  async function callHello() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/api/hello');
      const json = await res.json();
      setResult(json);
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function callPositions() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/api/positions?limit=5');
      const json = await res.json();
      setResult(json);
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function startTrading() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch('http://localhost:8000/api/run-trading', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_path: null }),
      });
      const json = await res.json();
      setJobId(json.job_id);
      setResult(json);
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function checkJob() {
    if (!jobId) {
      setError('No job id available');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`http://localhost:8000/api/job/${jobId}`);
      const json = await res.json();
      setJobStatus(json);
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function stopJob() {
    if (!jobId) {
      setError('No job id available');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`http://localhost:8000/api/stop/${jobId}`, { method: 'POST' });
      const json = await res.json();
      setJobStatus(json);
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: 12, border: '1px solid #eee', borderRadius: 8 }}>
      <h3>Backend test</h3>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button onClick={callHello} disabled={loading}>Call /api/hello</button>
        <button onClick={callPositions} disabled={loading}>Call /api/positions</button>
        <button onClick={startTrading} disabled={loading}>Start Trading</button>
        <button onClick={checkJob} disabled={loading || !jobId}>Check Job</button>
        <button onClick={stopJob} disabled={loading || !jobId}>Stop Job</button>
      </div>

      {loading && <p>Loading...</p>}
      {error && (
        <div style={{ color: 'red' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {jobId && (
        <div style={{ marginTop: 8 }}>
          <strong>Job ID:</strong> {jobId}
        </div>
      )}

      {jobStatus && (
        <pre style={{ maxHeight: 400, overflow: 'auto', background: '#fafafa', padding: 8 }}>
          {JSON.stringify(jobStatus, null, 2)}
        </pre>
      )}

      {result && !jobStatus && (
        <pre style={{ maxHeight: 400, overflow: 'auto', background: '#fafafa', padding: 8 }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}
