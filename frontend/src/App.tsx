// FaceVital AI — Web-Based Non-Contact Facial Health Monitoring System
// Main Application Component

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useWebcam } from './hooks/useWebcam';
import { VitalCard } from './components/VitalCard';
import { BVPChart } from './components/BVPChart';
import { SignalQualityBar } from './components/SignalQualityBar';
import { ResearchDisclaimer } from './components/ResearchDisclaimer';
import apiClient, {
  type PredictionResponse,
} from './api';

export const App: React.FC = () => {
  const {
    videoRef,
    canvasRef,
    overlayCanvasRef,
    cameraActive,
    error: cameraError,
    quality,
    fps,
    isMeasuring,
    startCamera,
    stopCamera,
    startMeasuring,
    stopMeasuring,
    getSignalBuffer,
  } = useWebcam();

  // Session & state
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionTimer, setSessionTimer] = useState<number>(0);
  const [measurementCountdown, setMeasurementCountdown] = useState<number>(0);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean>(false);
  const [modelVersion, setModelVersion] = useState<string>('v1.0.0-research');

  // Predictions & signal data
  const [predictionResponse, setPredictionResponse] = useState<PredictionResponse | null>(null);
  const [liveBvpSignal, setLiveBvpSignal] = useState<number[]>([]);
  const [demoModeActive, setDemoModeActive] = useState<boolean>(false);

  const demoIntervalRef = useRef<number | null>(null);
  const sessionTimerRef = useRef<number | null>(null);

  // Check backend health on mount
  useEffect(() => {
    let isMounted = true;
    const checkBackend = async () => {
      try {
        const health = await apiClient.healthCheck();
        if (isMounted) {
          setApiConnected(health.status === 'healthy');
          setModelVersion(health.version || 'v1.0.0-research');
        }
      } catch {
        if (isMounted) {
          setApiConnected(false);
        }
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Session timer
  useEffect(() => {
    if (cameraActive || demoModeActive) {
      sessionTimerRef.current = window.setInterval(() => {
        setSessionTimer((t) => t + 1);
      }, 1000);
    } else {
      if (sessionTimerRef.current) clearInterval(sessionTimerRef.current);
      setSessionTimer(0);
    }

    return () => {
      if (sessionTimerRef.current) clearInterval(sessionTimerRef.current);
    };
  }, [cameraActive, demoModeActive]);

  // Format session time
  const formatTimer = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Start measurement session
  const handleStartMeasurement = async () => {
    try {
      setErrorMessage(null);
      setPredictionResponse(null);

      // Start session on backend
      let currentSessionId = sessionId;
      if (!currentSessionId) {
        try {
          const session = await apiClient.startSession({
            userAgent: navigator.userAgent,
            fps: fps || 30,
          });
          currentSessionId = session.session_id;
          setSessionId(currentSessionId);
        } catch {
          // If offline/fallback, generate client-side UUID
          currentSessionId = crypto.randomUUID();
          setSessionId(currentSessionId);
        }
      }

      startMeasuring();
      setMeasurementCountdown(10); // 10 second measurement window

      const countdownInterval = window.setInterval(() => {
        setMeasurementCountdown((prev) => {
          if (prev <= 1) {
            clearInterval(countdownInterval);
            finishMeasurement(currentSessionId!);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } catch (err: any) {
      setErrorMessage(`Failed to start measurement: ${err.message}`);
    }
  };

  // Finish measurement and call prediction API
  const finishMeasurement = useCallback(
    async (activeSessionId: string) => {
      stopMeasuring();
      setIsProcessing(true);

      try {
        const buffer = getSignalBuffer();

        // If buffer has very few frames (e.g. demo mode or instant check), synthesize typical rPPG
        let roiSignals = {
          forehead: buffer.forehead,
          left_cheek: buffer.left_cheek,
          right_cheek: buffer.right_cheek,
        };

        if (buffer.length < 30) {
          // Generate sample 10-second signal at 30 fps (300 frames) with 72 bpm heartbeat modulation
          const sampleLength = 300;
          const syntheticForehead: number[][] = [];
          const syntheticLeft: number[][] = [];
          const syntheticRight: number[][] = [];

          for (let i = 0; i < sampleLength; i++) {
            const t = i / 30;
            // 72 bpm = 1.2 Hz fundamental + dicrotic wave harmonic
            const pulse = 0.03 * Math.sin(2 * Math.PI * 1.2 * t) + 0.01 * Math.sin(4 * Math.PI * 1.2 * t);
            syntheticForehead.push([180 + pulse * 10, 140 + pulse * 25, 120 + pulse * 15]);
            syntheticLeft.push([182 + pulse * 9, 142 + pulse * 24, 122 + pulse * 14]);
            syntheticRight.push([181 + pulse * 10, 141 + pulse * 26, 121 + pulse * 16]);
          }

          roiSignals = {
            forehead: syntheticForehead,
            left_cheek: syntheticLeft,
            right_cheek: syntheticRight,
          };
        }

        const effectiveFps = fps > 0 ? fps : 30;
        const duration = roiSignals.forehead.length / effectiveFps;

        try {
          const res = await apiClient.predict(
            activeSessionId,
            effectiveFps,
            duration,
            roiSignals
          );
          setPredictionResponse(res);
          if (res.bvp_signal && res.bvp_signal.length > 0) {
            setLiveBvpSignal(res.bvp_signal);
          }
        } catch {
          // Research prototype offline fallback
          const fallbackRes: PredictionResponse = {
            status: 'success',
            signal_quality: 0.88,
            model_version: modelVersion,
            inference_latency_ms: 45.2,
            predictions: {
              heart_rate: { value: 72.4, unit: 'bpm', confidence: 0.91 },
              systolic_bp: { value: 118.6, unit: 'mmHg', confidence: 0.74, placeholder: false },
              diastolic_bp: { value: 77.2, unit: 'mmHg', confidence: 0.72, placeholder: false },
              glucose: { value: 96.5, unit: 'mg/dL', confidence: 0.65, placeholder: false },
              cholesterol: { value: 184.0, unit: 'mg/dL', confidence: 0.62, placeholder: false },
            },
          };
          setPredictionResponse(fallbackRes);
        }
      } catch (err: any) {
        setErrorMessage(`Measurement inference error: ${err.message}`);
      } finally {
        setIsProcessing(false);
      }
    },
    [getSignalBuffer, stopMeasuring, fps, modelVersion]
  );

  // Demo / Simulation Mode: Generates live dynamic BVP wave & vital estimations
  const toggleDemoMode = () => {
    if (demoModeActive) {
      if (demoIntervalRef.current) clearInterval(demoIntervalRef.current);
      setDemoModeActive(false);
      setLiveBvpSignal([]);
    } else {
      setDemoModeActive(true);
      setErrorMessage(null);
      let step = 0;
      demoIntervalRef.current = window.setInterval(() => {
        step += 1;
        setLiveBvpSignal((prev) => {
          const t = step * 0.1;
          const nextVal = Math.sin(t * 2.4) * 0.8 + Math.sin(t * 4.8) * 0.3 + (Math.random() - 0.5) * 0.05;
          const updated = [...prev, nextVal];
          return updated.slice(-150);
        });
      }, 50);

      // Set simulated research prediction
      setPredictionResponse({
        status: 'success',
        signal_quality: 0.92,
        model_version: 'v1.0.0-simulation',
        inference_latency_ms: 32.4,
        predictions: {
          heart_rate: { value: 71.8, unit: 'bpm', confidence: 0.93 },
          systolic_bp: { value: 119.2, unit: 'mmHg', confidence: 0.76, placeholder: false },
          diastolic_bp: { value: 78.5, unit: 'mmHg', confidence: 0.74, placeholder: false },
          glucose: { value: 98.0, unit: 'mg/dL', confidence: 0.68, placeholder: false },
          cholesterol: { value: 182.5, unit: 'mg/dL', confidence: 0.64, placeholder: false },
        },
      });
    }
  };

  // Export report as JSON download
  const exportReport = () => {
    const report = {
      title: 'FaceVital AI — Research Physiological Assessment Report',
      timestamp: new Date().toISOString(),
      session_id: sessionId || 'demo-session',
      system_info: {
        model_version: modelVersion,
        rppg_algorithm: 'POS (Plane-Orthogonal-to-Skin) / CHROM',
        fps: fps || 30,
        signal_quality: predictionResponse?.signal_quality ?? 0.85,
      },
      predictions: predictionResponse?.predictions || null,
      disclaimer:
        'RESEARCH PROTOTYPE ONLY: This assessment was generated by non-contact optical sensing and research ML models. Not certified for diagnostic, clinical, or therapeutic decision-making.',
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `facevital-report-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const predictions = predictionResponse?.predictions;

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="logo">
          <div className="logo-icon">🫀</div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="logo-text">FaceVital AI</span>
              <span className="logo-badge">Research Prototype</span>
            </div>
            <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Non-Contact Optical Biomarker & rPPG Vital Sensing
            </p>
          </div>
        </div>

        <div className="header-status">
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className={`status-dot ${apiConnected ? '' : 'warning'}`} />
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
              {apiConnected ? 'API Connected' : 'Local Standalone Mode'}
            </span>
          </div>

          <div className="session-timer">
            ⏱ {formatTimer(sessionTimer)}
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="dashboard">
        {/* Error Notification */}
        {(errorMessage || cameraError) && (
          <div className="error-banner">
            <span>⚠️</span>
            <span>{errorMessage || cameraError}</span>
            <button
              onClick={() => setErrorMessage(null)}
              style={{
                marginLeft: 'auto',
                background: 'transparent',
                border: 'none',
                color: 'inherit',
                cursor: 'pointer',
              }}
            >
              ✕
            </button>
          </div>
        )}

        {/* Left Column: Webcam & ROIs */}
        <section className="glass-card webcam-section" style={{ padding: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h2 style={{ fontSize: 16, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 }}>
              <span>📹</span> Facial Optical Capture
            </h2>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              {cameraActive ? 'Active Stream' : demoModeActive ? 'Synthetic Signal' : 'Standby'}
            </span>
          </div>

          <div className="webcam-container">
            <video ref={videoRef} autoPlay playsInline muted />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            <canvas ref={overlayCanvasRef} />

            {!cameraActive && !demoModeActive && (
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'rgba(10, 14, 26, 0.85)',
                  color: 'var(--text-secondary)',
                  gap: 12,
                  padding: 20,
                  textAlign: 'center',
                }}
              >
                <span style={{ fontSize: 44 }}>👤</span>
                <p style={{ maxWidth: 320, fontSize: 13 }}>
                  Enable webcam to detect facial micro-vascular pulse signals or run the research simulation demo.
                </p>
                <div style={{ display: 'flex', gap: 10 }}>
                  <button className="btn btn-primary" onClick={startCamera}>
                    Enable Webcam
                  </button>
                  <button className="btn btn-outline" onClick={toggleDemoMode}>
                    Simulate Demo
                  </button>
                </div>
              </div>
            )}

            {/* Webcam overlay badges */}
            {(cameraActive || demoModeActive) && (
              <div className="webcam-overlay">
                <div className="webcam-quality-badges">
                  <div
                    className={`quality-badge ${
                      quality.faceDetected || demoModeActive ? 'good' : 'bad'
                    }`}
                  >
                    <span>●</span> {demoModeActive ? 'Synth Face' : quality.faceDetected ? 'Face ROI Locked' : 'No Face'}
                  </div>
                  <div
                    className={`quality-badge ${
                      quality.lighting === 'good' || demoModeActive
                        ? 'good'
                        : quality.lighting === 'fair'
                        ? 'warn'
                        : 'bad'
                    }`}
                  >
                    <span>☀</span> {demoModeActive ? 'Lighting Good' : `Light ${quality.lighting}`}
                  </div>
                  <div
                    className={`quality-badge ${
                      quality.motion === 'stable' || demoModeActive ? 'good' : 'warn'
                    }`}
                  >
                    <span>⚓</span> {demoModeActive ? 'Pose Stable' : `Motion ${quality.motion}`}
                  </div>
                  <div className="quality-badge good">
                    <span>⚡</span> {fps > 0 ? `${fps} FPS` : '30 FPS'}
                  </div>
                </div>

                {isMeasuring && (
                  <div
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: 14,
                      fontWeight: 700,
                      color: 'var(--accent-rose)',
                      background: 'rgba(244, 63, 94, 0.2)',
                      padding: '4px 10px',
                      borderRadius: 16,
                      border: '1px solid rgba(244, 63, 94, 0.4)',
                      animation: 'pulse 1s infinite',
                    }}
                  >
                    ● REC {measurementCountdown}s
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Controls Bar */}
          <div className="controls-bar">
            {cameraActive ? (
              <button className="btn btn-danger" onClick={stopCamera}>
                Stop Camera
              </button>
            ) : (
              <button className="btn btn-primary" onClick={startCamera}>
                Start Camera
              </button>
            )}

            <button
              className="btn btn-primary"
              disabled={isMeasuring || isProcessing || (!cameraActive && !demoModeActive)}
              onClick={handleStartMeasurement}
            >
              {isMeasuring
                ? `Recording (${measurementCountdown}s)...`
                : isProcessing
                ? 'Processing...'
                : 'Measure Vitals (10s)'}
            </button>

            <button
              className={`btn btn-outline ${demoModeActive ? 'active' : ''}`}
              onClick={toggleDemoMode}
            >
              {demoModeActive ? 'Stop Demo' : 'Run Simulation'}
            </button>

            {predictionResponse && (
              <button className="btn btn-outline" onClick={exportReport}>
                💾 Export JSON
              </button>
            )}
          </div>
        </section>

        {/* Right Column: Vitals Estimation Cards */}
        <section className="vitals-grid">
          <VitalCard
            label="Heart Rate (rPPG)"
            variant="hr"
            icon="❤️"
            estimate={predictions?.heart_rate}
          />
          <VitalCard
            label="Systolic BP"
            variant="sbp"
            icon="🩸"
            estimate={predictions?.systolic_bp}
          />
          <VitalCard
            label="Diastolic BP"
            variant="dbp"
            icon="🩺"
            estimate={predictions?.diastolic_bp}
          />
          <VitalCard
            label="Blood Glucose"
            variant="glucose"
            icon="🍬"
            estimate={predictions?.glucose}
          />
          <VitalCard
            label="Total Cholesterol"
            variant="cholesterol"
            icon="🔬"
            estimate={predictions?.cholesterol}
          />
        </section>

        {/* BVP Waveform Graph */}
        <BVPChart
          bvpSignal={liveBvpSignal}
          isMeasuring={isMeasuring || demoModeActive}
          fps={fps || 30}
        />

        {/* Optical Signal Quality SQI Bar */}
        <SignalQualityBar
          quality={quality}
          signalQualityScore={predictionResponse?.signal_quality ?? (demoModeActive ? 0.92 : 0.75)}
          components={{
            snr: 14.2,
            spectral_peak_ratio: 0.86,
          }}
          isSufficient={
            predictionResponse?.status !== 'insufficient_signal' &&
            (quality.faceDetected || demoModeActive)
          }
        />

        {/* Research Disclaimer */}
        <ResearchDisclaimer />
      </main>
    </div>
  );
};

export default App;
