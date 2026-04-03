// ---------------------------------------------------------------------------
// Types mirroring the FastAPI PredictResponse schema exactly.
// All image fields are base64-encoded PNG strings.
// ---------------------------------------------------------------------------

export interface PredictionItem {
  label_code: string;
  label_name: string;
  probability: number; // 0–1
  predicted: boolean;
}

export interface ClassVisuals {
  label_code: string;
  label_name: string;
  probability: number;
  /** Raw GradCAM activation heatmap */
  heatmap: string;
  /** GradCAM blended on the fundus image */
  overlay: string;
  /** Hot-region contour/polygon boundary */
  polygon_overlay: string;
  /** Axis-aligned bounding boxes drawn on the fundus */
  bounding_box: string;
  /** 2×2 comparison panel: original / heatmap / mask / overlay */
  panel: string;
}

export interface PredictResponse {
  device: string;
  threshold: number;
  predictions: PredictionItem[];
  /** Preprocessed input fundus image — base64 PNG */
  original_image: string;
  visuals: ClassVisuals[];
}

// ---------------------------------------------------------------------------
// UI-internal types
// ---------------------------------------------------------------------------

/** Which visualisation tab is currently selected */
export type VisualTab = "heatmap" | "overlay" | "polygon_overlay" | "bounding_box" | "panel";

export const VISUAL_TABS: { key: VisualTab; label: string }[] = [
  { key: "heatmap",         label: "Heatmap" },
  { key: "overlay",         label: "Overlay" },
  { key: "polygon_overlay", label: "Polygon" },
  { key: "bounding_box",    label: "Bounding Box" },
  { key: "panel",           label: "Panel" },
];

export type AppState =
  | { stage: "idle" }
  | { stage: "selected"; file: File; previewUrl: string }
  // Loading carries file info so SUCCESS can reconstruct the result state
  | { stage: "loading"; file: File; previewUrl: string }
  | { stage: "result"; file: File; previewUrl: string; data: PredictResponse }
  | { stage: "error"; message: string };
