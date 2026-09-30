// FaceVital AI — API Client
// Communicates with the FastAPI backend

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface VitalEstimate {
  value: number | null;
  unit: string;
  confidence: number;
  placeholder?: boolean;
  message?: string;
}

export interface PredictionResult {
  heart_rate?: VitalEstimate;
  systolic_bp?: VitalEstimate;
  diastolic_bp?: VitalEstimate;
  glucose?: VitalEstimate;
  cholesterol?: VitalEstimate;
}

export interface PredictionResponse {
  status: 'success' | 'insufficient_signal' | 'error';
  predictions?: PredictionResult;
  signal_quality: number;
  model_version: string;
  inference_latency_ms?: number;
  message?: string;
  bvp_signal?: number[];
}

export interface SignalQualityResponse {
  signal_quality: number;
  components: Record<string, number>;
  is_sufficient: boolean;
  message: string;
}

export interface SessionStartResponse {
  session_id: string;
  status: string;
  started_at: string;
  model_version: string;
}

export interface ModelInfoResponse {
  model_version: string;
  architecture: string;
  framework: string;
  is_placeholder: boolean;
  supported_targets: string[];
  rppg_algorithm: string;
  signal_quality_threshold: number;
  description: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  database: string;
  model_loaded: boolean;
  uptime_seconds: number;
}

export interface ROISignals {
  forehead: number[][];
  left_cheek: number[][];
  right_cheek: number[][];
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API error: ${response.status}`);
    }

    return response.json();
  }

  async healthCheck(): Promise<HealthResponse> {
    return this.request('/api/v1/health');
  }

  async startSession(deviceMetadata?: Record<string, unknown>): Promise<SessionStartResponse> {
    return this.request('/api/v1/session/start', {
      method: 'POST',
      body: JSON.stringify({ device_metadata: deviceMetadata }),
    });
  }

  async predict(
    sessionId: string,
    fps: number,
    durationSeconds: number,
    roiSignals: ROISignals,
  ): Promise<PredictionResponse> {
    return this.request('/api/v1/predict', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        fps,
        duration_seconds: durationSeconds,
        roi_signals: roiSignals,
      }),
    });
  }

  async checkSignalQuality(roiSignals: ROISignals, fps: number): Promise<SignalQualityResponse> {
    return this.request('/api/v1/signal-quality', {
      method: 'POST',
      body: JSON.stringify({ roi_signals: roiSignals, fps }),
    });
  }

  async getModelInfo(): Promise<ModelInfoResponse> {
    return this.request('/api/v1/model-info');
  }
}

export const apiClient = new ApiClient();
export default apiClient;
