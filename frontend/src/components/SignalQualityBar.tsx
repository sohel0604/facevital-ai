// FaceVital AI — Signal Quality & Metrics Bar Component

import React from 'react';
import type { QualityMetrics } from '../hooks/useWebcam';

interface SignalQualityBarProps {
  quality: QualityMetrics;
  signalQualityScore?: number; // 0 to 1
  components?: Record<string, number>;
  isSufficient?: boolean;
}

export const SignalQualityBar: React.FC<SignalQualityBarProps> = ({
  quality,
  signalQualityScore = 0,
  components = {},
  isSufficient = true,
}) => {
  const percent = Math.min(100, Math.max(0, Math.round(signalQualityScore * 100)));

  const qualityColor =
    percent >= 70
      ? 'var(--accent-emerald)'
      : percent >= 45
      ? 'var(--accent-amber)'
      : 'var(--accent-rose)';

  return (
    <div className="glass-card signal-quality-section">
      <div className="signal-quality-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 16 }}>📊</span>
          <span style={{ fontWeight: 600, fontSize: 14 }}>
            Optical Signal Quality Index (SQI)
          </span>
          <span
            style={{
              fontSize: 12,
              padding: '2px 8px',
              borderRadius: 12,
              background: isSufficient
                ? 'rgba(16, 185, 129, 0.15)'
                : 'rgba(244, 63, 94, 0.15)',
              color: isSufficient ? '#6ee7b7' : '#fda4af',
              border: `1px solid ${
                isSufficient ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'
              }`,
            }}
          >
            {isSufficient ? 'Sufficient for rPPG' : 'Low Quality - Stabilize Pose'}
          </span>
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 16, fontWeight: 700, color: qualityColor }}>
          {percent}%
        </div>
      </div>

      <div className="signal-quality-bar-outer">
        <div
          className="signal-quality-bar-inner"
          style={{
            width: `${percent}%`,
            background: qualityColor,
          }}
        />
      </div>

      <div className="quality-components">
        <div className="quality-component">
          <div
            className="dot"
            style={{
              background:
                quality.lighting === 'good'
                  ? 'var(--accent-emerald)'
                  : quality.lighting === 'fair'
                  ? 'var(--accent-amber)'
                  : 'var(--accent-rose)',
            }}
          />
          <span>Lighting: <strong>{quality.lighting}</strong></span>
        </div>

        <div className="quality-component">
          <div
            className="dot"
            style={{
              background:
                quality.motion === 'stable'
                  ? 'var(--accent-emerald)'
                  : quality.motion === 'slight'
                  ? 'var(--accent-amber)'
                  : 'var(--accent-rose)',
            }}
          />
          <span>Motion: <strong>{quality.motion}</strong></span>
        </div>

        <div className="quality-component">
          <div
            className="dot"
            style={{
              background: quality.faceDetected
                ? 'var(--accent-emerald)'
                : 'var(--accent-rose)',
            }}
          />
          <span>Face Tracking: <strong>{quality.faceDetected ? 'Locked' : 'Searching'}</strong></span>
        </div>

        {components.snr !== undefined && (
          <div className="quality-component">
            <div className="dot" style={{ background: 'var(--accent-cyan)' }} />
            <span>SNR: <strong>{components.snr.toFixed(1)} dB</strong></span>
          </div>
        )}

        {components.spectral_peak_ratio !== undefined && (
          <div className="quality-component">
            <div className="dot" style={{ background: 'var(--accent-indigo)' }} />
            <span>Spectral Peak Ratio: <strong>{components.spectral_peak_ratio.toFixed(2)}</strong></span>
          </div>
        )}
      </div>
    </div>
  );
};

export default SignalQualityBar;
