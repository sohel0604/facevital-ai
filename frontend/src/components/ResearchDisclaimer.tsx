// FaceVital AI — Research Prototype Disclaimer Component

import React from 'react';

export const ResearchDisclaimer: React.FC = () => {
  return (
    <div className="disclaimer">
      <div className="disclaimer-icon">⚠️</div>
      <div>
        <p>
          <strong>RESEARCH PROTOTYPE ONLY — NOT FOR MEDICAL USE:</strong> FaceVital AI is an experimental computer vision and machine learning platform demonstrating remote photoplethysmography (rPPG) and physiological feature extraction. Heart rate (HR) is estimated optically from facial micro-vascular pulse dynamics. Systolic/diastolic blood pressure, blood glucose, and cholesterol estimations are experimental computational approximations requiring laboratory calibration against certified reference instruments (e.g., arterial cannulation, finger-prick spectrophotometry).
        </p>
        <p style={{ marginTop: 4, fontSize: '11px', color: 'var(--text-muted)' }}>
          Do not use this system for triage, diagnosis, treatment planning, or life-critical monitoring. Reference: Wang et al. (IEEE TBME 2017) POS algorithm; de Haan & Jeanne (IEEE TBME 2013) CHROM method.
        </p>
      </div>
    </div>
  );
};

export default ResearchDisclaimer;
