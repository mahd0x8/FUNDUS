interface Props {
  previewUrl: string;
  fileName: string;
  onReset: () => void;
  onSubmit: () => void;
  loading: boolean;
}

export default function ImagePreview({ previewUrl, fileName, onReset, onSubmit, loading }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <div className="relative rounded-xl overflow-hidden bg-slate-100 border border-slate-200">
        <img
          src={previewUrl}
          alt="Selected fundus"
          className="w-full max-h-72 object-contain"
        />
        {/* File name badge */}
        <div className="absolute bottom-2 left-2 px-2.5 py-1 bg-black/50 backdrop-blur-sm rounded-lg">
          <span className="text-xs text-white font-medium truncate max-w-xs block">
            {fileName}
          </span>
        </div>
      </div>

      <div className="flex gap-3">
        <button className="btn-secondary" onClick={onReset} disabled={loading}>
          {/* Trash icon */}
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
          Change image
        </button>

        <button className="btn-primary flex-1 justify-center" onClick={onSubmit} disabled={loading}>
          {loading ? (
            <>
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor"
                  d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              Analysing…
            </>
          ) : (
            <>
              {/* Sparkle icon */}
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
              </svg>
              Run Analysis
            </>
          )}
        </button>
      </div>
    </div>
  );
}
