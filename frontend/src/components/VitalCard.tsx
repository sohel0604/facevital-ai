// FaceVital AI — Vital Sign Card Component

import React from 'react';
import type { VitalEstimate } from '../api';

interface VitalCardProps {
  label: string;
  estimate: VitalEstimate | null | undefined;
  variant: 'hr' | 'sbp' | 'dbp' | 'glucose' | 'cholesterol';
  icon: string;
}

export const VitalCard: React.FC<VitalCardProps> = ({ label, estimate, variant, icon }) => {
  const hasValue = estimate && estimate.value !== null && estimate.value !== undefined;
  const isPlaceholder = estimate?.placeholder;
  const confidence = estimate?.confidence ?? 0;

  const confidenceClass = confidence >= 0.7 ? 'high' : confidence >= 0.4 ? 'medium' : 'low';

  return (
    <div className={`glass-card vital-card ${variant} animate-scale-in`}>
      <div className="vital-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span><span>{icon}</span> {label}</span>
        {isPlaceholder && (
          <span
            style={{
              fontSize: '10px',
              padding: '1px 6px',
              borderRadius: '10px',
              background: 'rgba(245, 158, 11, 0.15)',
              color: 'var(--accent-amber)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              textTransform: 'none',
              letterSpacing: '0px',
            }}
          >
            Research Model
          </span>
        )}
      </div>

      {hasValue ? (
        <>
          <div className="vital-value">{estimate!.value!.toFixed(1)}</div>
          <div className="vital-unit">{estimate!.unit}</div>
          <div className="vital-confidence">
            <div className="confidence-bar">
              <div
                className={`confidence-bar-fill ${confidenceClass}`}
                style={{ width: `${Math.max(15, confidence * 100)}%` }}
              />
            </div>
            <span className="confidence-text">{(confidence * 100).toFixed(0)}%</span>
          </div>
        </>
      ) : (
        <div className="vital-placeholder">
          — waiting for measurement
        </div>
      )}
    </div>
  );
};

export default VitalCard;
