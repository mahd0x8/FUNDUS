/**
 * api.ts — thin wrapper around the FastAPI /predict endpoint.
 *
 * The backend accepts multipart/form-data with:
 *   file          — the fundus image (JPEG / PNG / BMP / TIFF / WebP)
 *   threshold     — float 0–1 (default 0.5)
 *   top_k_if_none — int (default 3)
 *
 * All images in the response are base64-encoded PNG strings.
 */

import type { PredictResponse } from "../types";

const BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

export interface PredictOptions {
  threshold?: number;
  topKIfNone?: number;
}

export async function predict(
  file: File,
  options: PredictOptions = {}
): Promise<PredictResponse> {
  const { threshold = 0.5, topKIfNone = 3 } = options;

  const form = new FormData();
  form.append("file", file);
  form.append("threshold", String(threshold));
  form.append("top_k_if_none", String(topKIfNone));

  const response = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    // Try to surface the FastAPI error detail if present
    let detail = `HTTP ${response.status}`;
    try {
      const json = (await response.json()) as { detail?: string };
      if (json.detail) detail = json.detail;
    } catch {
      // ignore parse failures — use the status code message
    }
    throw new Error(detail);
  }

  return (await response.json()) as PredictResponse;
}

/** Convert a base64 PNG string (as returned by the API) to a data-URL. */
export function base64ToDataUrl(b64: string): string {
  return `data:image/png;base64,${b64}`;
}

/**
 * Trigger a browser download of a base64 image.
 * @param b64      Base64-encoded PNG string
 * @param filename Suggested file name
 */
export function downloadBase64Image(b64: string, filename: string): void {
  const a = document.createElement("a");
  a.href = base64ToDataUrl(b64);
  a.download = filename;
  a.click();
}
