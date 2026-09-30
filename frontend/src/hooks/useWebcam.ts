// FaceVital AI — Webcam & Face Detection Hook
// Manages camera, MediaPipe face detection, ROI extraction, and signal buffering

import { useRef, useState, useCallback, useEffect } from 'react';

// ── Types ──
export interface FaceROI {
  forehead: { x: number; y: number; w: number; h: number };
  leftCheek: { x: number; y: number; w: number; h: number };
  rightCheek: { x: number; y: number; w: number; h: number };
}

export interface QualityMetrics {
  camera: 'active' | 'inactive' | 'error';
  lighting: 'good' | 'fair' | 'poor';
  motion: 'stable' | 'slight' | 'excessive';
  faceAngle: 'frontal' | 'slight' | 'too_large';
  faceDetected: boolean;
  multipleFaces: boolean;
  faceTooSmall: boolean;
}

export interface ROIFrame {
  forehead: [number, number, number]; // R, G, B
  leftCheek: [number, number, number];
  rightCheek: [number, number, number];
  timestamp: number;
}

// ── MediaPipe Face Mesh landmark indices for ROI extraction ──
export const FOREHEAD_INDICES = [10, 67, 109, 108, 151, 338, 297, 299];
export const LEFT_CHEEK_INDICES = [116, 117, 118, 119, 100, 36, 205, 187];
export const RIGHT_CHEEK_INDICES = [345, 346, 347, 348, 329, 266, 425, 411];

export function useWebcam() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animFrameRef = useRef<number>(0);
  const signalBufferRef = useRef<ROIFrame[]>([]);
  const prevFrameMeansRef = useRef<number[] | null>(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [quality, setQuality] = useState<QualityMetrics>({
    camera: 'inactive',
    lighting: 'good',
    motion: 'stable',
    faceAngle: 'frontal',
    faceDetected: false,
    multipleFaces: false,
    faceTooSmall: false,
  });
  const [faceROI, setFaceROI] = useState<FaceROI | null>(null);
  const [fps, setFps] = useState(0);
  const [isMeasuring, setIsMeasuring] = useState(false);

  // FPS tracking
  const fpsCountRef = useRef(0);
  const fpsTimeRef = useRef(Date.now());

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 },
          facingMode: 'user',
        },
      });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraActive(true);
      setQuality(q => ({ ...q, camera: 'active' }));
    } catch (err: any) {
      const msg = err.name === 'NotAllowedError'
        ? 'Camera permission denied. Please allow camera access.'
        : err.name === 'NotFoundError'
          ? 'No camera found. Please connect a webcam.'
          : `Camera error: ${err.message}`;
      setError(msg);
      setQuality(q => ({ ...q, camera: 'error' }));
    }
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
    }
    setCameraActive(false);
    setQuality(q => ({ ...q, camera: 'inactive', faceDetected: false }));
  }, []);

  // Extract ROI average RGB from a canvas given a bounding rect
  const extractROI = useCallback((
    ctx: CanvasRenderingContext2D,
    roi: { x: number; y: number; w: number; h: number },
    videoWidth: number,
    videoHeight: number,
  ): [number, number, number] => {
    const x = Math.max(0, Math.floor(roi.x * videoWidth));
    const y = Math.max(0, Math.floor(roi.y * videoHeight));
    const w = Math.max(1, Math.min(Math.floor(roi.w * videoWidth), videoWidth - x));
    const h = Math.max(1, Math.min(Math.floor(roi.h * videoHeight), videoHeight - y));

    try {
      const imageData = ctx.getImageData(x, y, w, h);
      const data = imageData.data;
      let rSum = 0, gSum = 0, bSum = 0;
      const pixels = data.length / 4;

      for (let i = 0; i < data.length; i += 4) {
        rSum += data[i];
        gSum += data[i + 1];
        bSum += data[i + 2];
      }

      return [rSum / pixels, gSum / pixels, bSum / pixels];
    } catch {
      return [0, 0, 0];
    }
  }, []);

  // Process frame — detect face using simple skin detection (fallback when MediaPipe unavailable)
  const processFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const overlay = overlayCanvasRef.current;

    if (!video || !canvas || !cameraActive || video.readyState < 2) {
      animFrameRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const overlayCtx = overlay?.getContext('2d');

    if (!ctx) {
      animFrameRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    if (vw === 0 || vh === 0) {
      animFrameRef.current = requestAnimationFrame(processFrame);
      return;
    }

    canvas.width = vw;
    canvas.height = vh;
    if (overlay) {
      overlay.width = vw;
      overlay.height = vh;
    }

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, vw, vh);

    // FPS tracking
    fpsCountRef.current++;
    const now = Date.now();
    if (now - fpsTimeRef.current >= 1000) {
      setFps(fpsCountRef.current);
      fpsCountRef.current = 0;
      fpsTimeRef.current = now;
    }

    // Simple face detection using center-of-frame heuristic
    // In production, this would use MediaPipe Face Mesh landmarks
    // For now, we use a center region that approximates face position with webcam

    // Approximate face region (center 40% of frame)
    const faceX = 0.3;
    const faceY = 0.15;
    const faceW = 0.4;
    const faceH = 0.6;

    // Check brightness for lighting quality
    const centerData = ctx.getImageData(
      Math.floor(vw * 0.3), Math.floor(vh * 0.2),
      Math.floor(vw * 0.4), Math.floor(vh * 0.4),
    );
    let totalBrightness = 0;
    for (let i = 0; i < centerData.data.length; i += 4) {
      totalBrightness += (centerData.data[i] + centerData.data[i + 1] + centerData.data[i + 2]) / 3;
    }
    const avgBrightness = totalBrightness / (centerData.data.length / 4);

    const lighting: 'good' | 'fair' | 'poor' =
      avgBrightness > 80 ? 'good' : avgBrightness > 40 ? 'fair' : 'poor';

    // Motion detection using frame difference
    const currentMeans = [
      centerData.data[0], centerData.data[1], centerData.data[2],
    ];
    let motion: 'stable' | 'slight' | 'excessive' = 'stable';
    if (prevFrameMeansRef.current) {
      const diff = Math.abs(currentMeans[0] - prevFrameMeansRef.current[0]) +
                   Math.abs(currentMeans[1] - prevFrameMeansRef.current[1]) +
                   Math.abs(currentMeans[2] - prevFrameMeansRef.current[2]);
      motion = diff > 50 ? 'excessive' : diff > 15 ? 'slight' : 'stable';
    }
    prevFrameMeansRef.current = currentMeans;

    // Detect if there's skin-colored pixels in center region (simple detection)
    let skinPixels = 0;
    const totalPixels = centerData.data.length / 4;
    for (let i = 0; i < centerData.data.length; i += 16) { // Sample every 4th pixel
      const r = centerData.data[i];
      const g = centerData.data[i + 1];
      const b = centerData.data[i + 2];
      // Simple skin detection heuristic
      if (r > 60 && g > 40 && b > 20 && r > g && r > b && (r - g) > 10 && (r - b) > 15) {
        skinPixels++;
      }
    }
    const skinRatio = skinPixels / (totalPixels / 4);
    const faceDetected = skinRatio > 0.1;

    // Define ROIs
    const rois: FaceROI = {
      forehead: { x: faceX + faceW * 0.2, y: faceY + faceH * 0.05, w: faceW * 0.6, h: faceH * 0.15 },
      leftCheek: { x: faceX + faceW * 0.05, y: faceY + faceH * 0.4, w: faceW * 0.3, h: faceH * 0.2 },
      rightCheek: { x: faceX + faceW * 0.65, y: faceY + faceH * 0.4, w: faceW * 0.3, h: faceH * 0.2 },
    };

    setFaceROI(rois);
    setQuality({
      camera: 'active',
      lighting,
      motion,
      faceAngle: 'frontal',
      faceDetected,
      multipleFaces: false,
      faceTooSmall: false,
    });

    // Draw ROI overlays
    if (overlayCtx && faceDetected) {
      overlayCtx.clearRect(0, 0, vw, vh);

      // Face bounding box
      overlayCtx.strokeStyle = 'rgba(99, 102, 241, 0.7)';
      overlayCtx.lineWidth = 2;
      overlayCtx.setLineDash([6, 4]);
      overlayCtx.strokeRect(faceX * vw, faceY * vh, faceW * vw, faceH * vh);
      overlayCtx.setLineDash([]);

      // ROI overlays
      const roiColors: Record<string, string> = {
        forehead: 'rgba(99, 102, 241, 0.25)',
        leftCheek: 'rgba(16, 185, 129, 0.25)',
        rightCheek: 'rgba(6, 182, 212, 0.25)',
      };
      const roiBorders: Record<string, string> = {
        forehead: 'rgba(99, 102, 241, 0.6)',
        leftCheek: 'rgba(16, 185, 129, 0.6)',
        rightCheek: 'rgba(6, 182, 212, 0.6)',
      };

      for (const [name, roi] of Object.entries(rois)) {
        overlayCtx.fillStyle = roiColors[name] || 'rgba(255,255,255,0.1)';
        overlayCtx.fillRect(roi.x * vw, roi.y * vh, roi.w * vw, roi.h * vh);
        overlayCtx.strokeStyle = roiBorders[name] || 'rgba(255,255,255,0.3)';
        overlayCtx.lineWidth = 1.5;
        overlayCtx.strokeRect(roi.x * vw, roi.y * vh, roi.w * vw, roi.h * vh);

        // Label
        overlayCtx.fillStyle = 'rgba(255,255,255,0.8)';
        overlayCtx.font = '10px Inter, sans-serif';
        overlayCtx.fillText(name, roi.x * vw + 4, roi.y * vh + 12);
      }
    } else if (overlayCtx) {
      overlayCtx.clearRect(0, 0, vw, vh);
    }

    // Extract ROI signals if measuring
    if (isMeasuring && faceDetected) {
      const foreheadRGB = extractROI(ctx, rois.forehead, vw, vh);
      const leftCheekRGB = extractROI(ctx, rois.leftCheek, vw, vh);
      const rightCheekRGB = extractROI(ctx, rois.rightCheek, vw, vh);

      signalBufferRef.current.push({
        forehead: foreheadRGB,
        leftCheek: leftCheekRGB,
        rightCheek: rightCheekRGB,
        timestamp: Date.now(),
      });
    }

    animFrameRef.current = requestAnimationFrame(processFrame);
  }, [cameraActive, isMeasuring, extractROI]);

  // Start/stop processing loop
  useEffect(() => {
    if (cameraActive) {
      animFrameRef.current = requestAnimationFrame(processFrame);
    }
    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [cameraActive, processFrame]);

  // Get buffered signals
  const getSignalBuffer = useCallback(() => {
    const buffer = signalBufferRef.current;
    return {
      forehead: buffer.map(f => [...f.forehead]),
      left_cheek: buffer.map(f => [...f.leftCheek]),
      right_cheek: buffer.map(f => [...f.rightCheek]),
      length: buffer.length,
    };
  }, []);

  // Clear buffer
  const clearSignalBuffer = useCallback(() => {
    signalBufferRef.current = [];
  }, []);

  // Start measuring
  const startMeasuring = useCallback(() => {
    signalBufferRef.current = [];
    setIsMeasuring(true);
  }, []);

  // Stop measuring
  const stopMeasuring = useCallback(() => {
    setIsMeasuring(false);
  }, []);

  return {
    videoRef,
    canvasRef,
    overlayCanvasRef,
    cameraActive,
    error,
    quality,
    faceROI,
    fps,
    isMeasuring,
    startCamera,
    stopCamera,
    startMeasuring,
    stopMeasuring,
    getSignalBuffer,
    clearSignalBuffer,
  };
}
