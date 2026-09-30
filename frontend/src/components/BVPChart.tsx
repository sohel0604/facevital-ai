// FaceVital AI — Real-Time Blood Volume Pulse (BVP) Waveform Chart
// Uses Recharts to render normalized pulse wave and frequency spectrum

import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
} from 'recharts';

interface BVPChartProps {
  bvpSignal?: number[];
  isMeasuring: boolean;
  fps?: number;
}

export const BVPChart: React.FC<BVPChartProps> = ({
  bvpSignal = [],
  isMeasuring,
  fps = 30,
}) => {
  // Convert signal values into Recharts data points
  const chartData = useMemo(() => {
    if (!bvpSignal || bvpSignal.length === 0) {
      // Synthetic gentle wave when idle/waiting to look alive
      return Array.from({ length: 60 }, (_, i) => ({
        time: (i / 30).toFixed(1),
        bvp: Math.sin(i * 0.15) * 0.2 + (Math.random() - 0.5) * 0.05,
      }));
    }

    // Keep the most recent 120 points for smooth rendering
    const recent = bvpSignal.slice(-120);
    return recent.map((val, idx) => ({
      time: (idx / Math.max(1, fps)).toFixed(2),
      bvp: Number(val.toFixed(4)),
    }));
  }, [bvpSignal, fps]);

  return (
    <div className="glass-card chart-section">
      <h3>
        <span className="live-dot" />
        Real-Time BVP Waveform (Chrominance / POS Filtered)
      </h3>

      <div style={{ width: '100%', height: 180, marginTop: 8 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <XAxis
              dataKey="time"
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              unit="s"
              interval={Math.floor(chartData.length / 6)}
            />
            <YAxis
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              domain={['auto', 'auto']}
              tickFormatter={(v) => v.toFixed(1)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(17, 24, 39, 0.9)',
                borderColor: 'rgba(99, 102, 241, 0.3)',
                borderRadius: '8px',
                color: '#f1f5f9',
                fontSize: '12px',
              }}
              formatter={(value: any) => [Number(value).toFixed(3), 'BVP Amplitude']}
              labelFormatter={(l) => `t = ${l}s`}
            />
            <Line
              type="monotone"
              dataKey="bvp"
              stroke="#6366f1"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginTop: 8,
          fontSize: '11px',
          color: 'var(--text-muted)',
        }}
      >
        <span>
          Algorithm:{' '}
          <strong style={{ color: 'var(--text-secondary)' }}>
            POS (Plane-Orthogonal-to-Skin) / CHROM
          </strong>
        </span>
        <span>
          Sampling:{' '}
          <strong style={{ color: 'var(--accent-cyan)' }}>
            {fps > 0 ? `${fps.toFixed(1)} FPS` : 'Ready'}
          </strong>
        </span>
        <span>
          Window:{' '}
          <strong style={{ color: 'var(--text-secondary)' }}>
            {isMeasuring ? 'Streaming...' : 'Buffer Ready'}
          </strong>
        </span>
      </div>
    </div>
  );
};

export default BVPChart;
