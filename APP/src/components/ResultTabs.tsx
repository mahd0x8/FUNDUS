import { useState } from "react";
import type { ClassVisuals, VisualTab } from "../types";
import { VISUAL_TABS } from "../types";
import { base64ToDataUrl, downloadBase64Image } from "../lib/api";

interface Props {
  visuals: ClassVisuals[];
  originalImage: string;
}

export default function ResultTabs({ visuals, originalImage }: Props) {
  const [activeTab, setActiveTab] = useState<VisualTab>("overlay");
  // Which class visual is selected (index into visuals array)
  const [activeClass, setActiveClass] = useState(0);

  const current = visuals[activeClass];

  // The base64 string for the currently visible image
  const currentB64: string = current
    ? (current[activeTab] as string)
    : originalImage;

  function handleDownload() {
    const name = current
      ? `${current.label_code}_${activeTab}.png`
      : `original.png`;
    downloadBase64Image(currentB64, name);
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Class selector — shown only when multiple visuals exist */}
      {visuals.length > 1 && (
        <div className="flex gap-2 flex-wrap">
          {visuals.map((v, i) => (
            <button
              key={v.label_code}
              onClick={() => setActiveClass(i)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors
                ${activeClass === i
                  ? "bg-brand-600 text-white border-brand-600"
                  : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50"
                }`}
            >
              {v.label_name}
              <span className="ml-1.5 opacity-70">
                {Math.round(v.probability * 100)}%
              </span>
            </button>
          ))}
        </div>
      )}

      {/* Visualisation type tabs */}
      <div className="flex gap-1 bg-slate-100 p-1 rounded-xl w-fit flex-wrap">
        {VISUAL_TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setActiveTab(t.key)}
            className={`tab-btn ${activeTab === t.key ? "tab-btn-active" : "tab-btn-inactive"}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Image */}
      {current ? (
        <div className="relative rounded-xl overflow-hidden bg-slate-900 border border-slate-200">
          <img
            key={`${activeClass}-${activeTab}`}
            src={base64ToDataUrl(currentB64)}
            alt={`${current.label_name} – ${activeTab}`}
            className="w-full object-contain max-h-96"
          />

          {/* Class + probability chip */}
          <div className="absolute top-3 left-3 px-3 py-1.5 bg-black/60 backdrop-blur-sm rounded-xl">
            <p className="text-white text-sm font-semibold leading-tight">{current.label_name}</p>
            <p className="text-white/70 text-xs">{Math.round(current.probability * 100)}% confidence</p>
          </div>

          {/* Download button */}
          <button
            onClick={handleDownload}
            className="absolute top-3 right-3 p-2 bg-black/50 backdrop-blur-sm rounded-xl
                       text-white hover:bg-black/70 transition-colors"
            title="Download image"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
          </button>
        </div>
      ) : (
        // Fallback: show original image if no visuals (shouldn't normally happen)
        <div className="relative rounded-xl overflow-hidden bg-slate-900 border border-slate-200">
          <img
            src={base64ToDataUrl(originalImage)}
            alt="Preprocessed fundus"
            className="w-full object-contain max-h-96"
          />
        </div>
      )}

      {/* Original image strip */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-slate-400 font-medium flex-shrink-0">Original:</span>
        <div className="rounded-lg overflow-hidden border border-slate-200 w-20 h-16 flex-shrink-0 bg-slate-100">
          <img
            src={base64ToDataUrl(originalImage)}
            alt="Original fundus"
            className="w-full h-full object-cover"
          />
        </div>
        <p className="text-xs text-slate-400">Preprocessed input used for inference</p>
      </div>
    </div>
  );
}
