import { useRef, useState, type DragEvent, type ChangeEvent } from "react";

interface Props {
  onFileSelected: (file: File) => void;
}

const ACCEPTED = ["image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"];

export default function UploadArea({ onFileSelected }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function handleFile(file: File) {
    if (!ACCEPTED.includes(file.type) && !file.name.match(/\.(jpe?g|png|bmp|tiff?|webp)$/i)) {
      alert("Unsupported file type. Please upload a JPEG, PNG, BMP, TIFF, or WebP image.");
      return;
    }
    onFileSelected(file);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  function onChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    // reset so the same file can be re-selected after a reset
    e.target.value = "";
  }

  return (
    <div
      className={`
        flex flex-col items-center justify-center gap-4 p-12
        border-2 border-dashed rounded-2xl cursor-pointer transition-colors duration-150
        ${dragging
          ? "border-brand-500 bg-brand-50"
          : "border-slate-300 bg-white hover:border-brand-400 hover:bg-slate-50"
        }
      `}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      {/* Upload icon */}
      <div className="w-14 h-14 rounded-full bg-brand-50 flex items-center justify-center">
        <svg
          className="w-7 h-7 text-brand-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
          />
        </svg>
      </div>

      <div className="text-center">
        <p className="font-medium text-slate-700">
          Drag &amp; drop a fundus image, or{" "}
          <span className="text-brand-600 underline underline-offset-2">browse</span>
        </p>
        <p className="text-sm text-slate-400 mt-1">
          JPEG · PNG · BMP · TIFF · WebP
        </p>
      </div>

      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept=".jpg,.jpeg,.png,.bmp,.tiff,.tif,.webp"
        onChange={onChange}
      />
    </div>
  );
}
