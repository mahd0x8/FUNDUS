import type { PredictionItem } from "../types";

interface Props {
  predictions: PredictionItem[];
  threshold: number;
}

export default function PredictionList({ predictions, threshold }: Props) {
  // Sort: predicted first, then by probability descending
  const sorted = [...predictions].sort((a, b) => {
    if (a.predicted !== b.predicted) return a.predicted ? -1 : 1;
    return b.probability - a.probability;
  });

  const positives = sorted.filter((p) => p.predicted);
  const topNegatives = sorted.filter((p) => !p.predicted).slice(0, 5);

  return (
    <div className="flex flex-col gap-4">
      {/* Summary badge row */}
      <div className="flex items-center gap-3 flex-wrap">
        <span
          className={`
            inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium
            ${positives.length > 0
              ? "bg-red-50 text-red-700 border border-red-200"
              : "bg-green-50 text-green-700 border border-green-200"
            }
          `}
        >
          <span
            className={`w-2 h-2 rounded-full ${positives.length > 0 ? "bg-red-500" : "bg-green-500"}`}
          />
          {positives.length > 0
            ? `${positives.length} condition${positives.length > 1 ? "s" : ""} detected`
            : "No conditions above threshold"}
        </span>
        <span className="text-xs text-slate-400">threshold = {threshold}</span>
      </div>

      {/* Detected conditions */}
      {positives.length > 0 && (
        <div className="flex flex-col gap-2">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
            Detected
          </h3>
          {positives.map((p) => (
            <PredictionRow key={p.label_code} item={p} highlighted />
          ))}
        </div>
      )}

      {/* Top 5 negative / low-confidence conditions */}
      {topNegatives.length > 0 && (
        <div className="flex flex-col gap-2">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
            Top candidates (below threshold)
          </h3>
          {topNegatives.map((p) => (
            <PredictionRow key={p.label_code} item={p} highlighted={false} />
          ))}
        </div>
      )}
    </div>
  );
}

function PredictionRow({ item, highlighted }: { item: PredictionItem; highlighted: boolean }) {
  const pct = Math.round(item.probability * 100);

  return (
    <div
      className={`
        flex items-center gap-3 px-4 py-3 rounded-xl border transition-colors
        ${highlighted
          ? "bg-red-50 border-red-200"
          : "bg-slate-50 border-slate-200"
        }
      `}
    >
      {/* Label */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-slate-800 truncate">{item.label_name}</p>
        <p className="text-xs text-slate-400">{item.label_code}</p>
      </div>

      {/* Bar + percentage */}
      <div className="flex items-center gap-2 w-36 flex-shrink-0">
        <div className="flex-1 h-1.5 bg-slate-200 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500
              ${highlighted ? "bg-red-500" : "bg-slate-400"}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span
          className={`text-sm font-semibold tabular-nums w-10 text-right
            ${highlighted ? "text-red-700" : "text-slate-500"}`}
        >
          {pct}%
        </span>
      </div>
    </div>
  );
}
