import { useCallback, useReducer, useRef } from "react";
import Header from "./components/Header";
import UploadArea from "./components/UploadArea";
import ImagePreview from "./components/ImagePreview";
import ResultTabs from "./components/ResultTabs";
import PredictionList from "./components/PredictionList";
import LoadingState from "./components/LoadingState";
import ErrorState from "./components/ErrorState";
import { predict } from "./lib/api";
import type { AppState, PredictResponse } from "./types";

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

type Action =
  | { type: "SELECT_FILE"; file: File; previewUrl: string }
  | { type: "SUBMIT" }
  | { type: "SUCCESS"; data: PredictResponse }
  | { type: "ERROR"; message: string }
  | { type: "RESET" };

function appReducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "SELECT_FILE":
      return { stage: "selected", file: action.file, previewUrl: action.previewUrl };

    case "SUBMIT": {
      if (state.stage !== "selected") return state;
      return { stage: "loading", file: state.file, previewUrl: state.previewUrl };
    }

    case "SUCCESS": {
      if (state.stage !== "loading") return state;
      return { stage: "result", file: state.file, previewUrl: state.previewUrl, data: action.data };
    }

    case "ERROR":
      return { stage: "error", message: action.message };

    case "RESET":
      return { stage: "idle" };
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function App() {
  const [state, dispatch] = useReducer(appReducer, { stage: "idle" });

  // Refs for stable callbacks that don't close over stale state
  const fileRef = useRef<File | null>(null);
  const previewUrlRef = useRef<string | null>(null);

  const handleFileSelected = useCallback((file: File) => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const previewUrl = URL.createObjectURL(file);
    fileRef.current = file;
    previewUrlRef.current = previewUrl;
    dispatch({ type: "SELECT_FILE", file, previewUrl });
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!fileRef.current) return;
    const file = fileRef.current;
    dispatch({ type: "SUBMIT" });
    try {
      const data = await predict(file);
      dispatch({ type: "SUCCESS", data });
    } catch (err) {
      const message = err instanceof Error ? err.message : "An unknown error occurred.";
      dispatch({ type: "ERROR", message });
    }
  }, []);

  const handleReset = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    fileRef.current = null;
    dispatch({ type: "RESET" });
  }, []);

  // After an error, let the user retry with the same file
  const handleRetry = useCallback(() => {
    if (fileRef.current && previewUrlRef.current) {
      dispatch({
        type: "SELECT_FILE",
        file: fileRef.current,
        previewUrl: previewUrlRef.current,
      });
    } else {
      dispatch({ type: "RESET" });
    }
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <Header />

      <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8 flex flex-col gap-6">

        {/* ── IDLE: show upload area ───────────────────────────────── */}
        {state.stage === "idle" && (
          <section className="card">
            <h2 className="text-base font-semibold text-slate-700 mb-4">
              Upload a fundus image to begin
            </h2>
            <UploadArea onFileSelected={handleFileSelected} />
          </section>
        )}

        {/* ── SELECTED: preview + submit ───────────────────────────── */}
        {state.stage === "selected" && (
          <section className="card">
            <h2 className="text-base font-semibold text-slate-700 mb-4">Selected image</h2>
            <ImagePreview
              previewUrl={state.previewUrl}
              fileName={state.file.name}
              onReset={handleReset}
              onSubmit={handleSubmit}
              loading={false}
            />
          </section>
        )}

        {/* ── LOADING: spinner ─────────────────────────────────────── */}
        {state.stage === "loading" && (
          <section className="card">
            <LoadingState />
          </section>
        )}

        {/* ── ERROR ────────────────────────────────────────────────── */}
        {state.stage === "error" && (
          <section className="card">
            <ErrorState message={state.message} onRetry={handleRetry} />
          </section>
        )}

        {/* ── RESULT ───────────────────────────────────────────────── */}
        {state.stage === "result" && (
          <>
            {/* Summary card */}
            <section className="card">
              <div className="flex items-center justify-between mb-5">
                <div>
                  <h2 className="text-base font-semibold text-slate-700">Analysis complete</h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Device: {state.data.device} · Threshold: {state.data.threshold}
                  </p>
                </div>
                <button className="btn-secondary text-sm" onClick={handleReset}>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  New image
                </button>
              </div>

              <PredictionList
                predictions={state.data.predictions}
                threshold={state.data.threshold}
              />
            </section>

            {/* GradCAM visualisations */}
            {state.data.visuals.length > 0 ? (
              <section className="card">
                <h2 className="text-base font-semibold text-slate-700 mb-4">
                  GradCAM visualisations
                </h2>
                <ResultTabs
                  visuals={state.data.visuals}
                  originalImage={state.data.original_image}
                />
              </section>
            ) : (
              <section className="card flex flex-col items-center gap-2 py-10 text-center">
                <p className="text-slate-500 text-sm">
                  No GradCAM visualisations were generated.
                </p>
                <p className="text-xs text-slate-400">
                  This can happen when all class probabilities are very low.
                </p>
              </section>
            )}
          </>
        )}
      </main>

      <footer className="text-center text-xs text-slate-400 py-4 border-t border-slate-200 bg-white">
        Fundus Disease Classifier · Multi-label retinal AI · Research use only
      </footer>
    </div>
  );
}
