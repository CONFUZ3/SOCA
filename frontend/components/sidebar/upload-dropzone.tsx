"use client";

import { useRef, useState } from "react";
import { Upload, Loader2, XCircle } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { apiUpload, ApiError } from "@/lib/api";
import { cn } from "@/lib/cn";

interface UploadResponse {
  loaded: Array<{ name: string }>;
  errors: Array<{ name: string; error: string }>;
}

/**
 * Drag-and-drop zone that POSTs files to /api/data/upload. On success,
 * invalidates the session query so the sidebar dataset list refreshes.
 */
export function UploadDropzone() {
  const inputRef = useRef<HTMLInputElement>(null);
  const qc = useQueryClient();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = async (files: FileList | File[] | null) => {
    if (!files || files.length === 0) return;
    const list = Array.from(files);
    setBusy(true);
    setError(null);
    try {
      const res = await apiUpload<UploadResponse>("/api/data/upload", list);
      if (res.errors?.length) {
        setError(
          res.errors.map((e) => `${e.name}: ${e.error}`).join("; "),
        );
      }
      await qc.invalidateQueries({ queryKey: ["session"] });
      await qc.invalidateQueries({ queryKey: ["map-state"] });
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? typeof e.detail === "string"
            ? e.detail
            : e.message
          : e instanceof Error
            ? e.message
            : "Upload failed";
      setError(msg);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        void handleFiles(e.dataTransfer.files);
      }}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-1 rounded border border-dashed px-2 py-3 text-2xs transition-colors",
        dragOver
          ? "border-accent bg-surface-2 text-text"
          : "border-border bg-bg text-text-faint hover:border-accent/60 hover:text-text-muted",
      )}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".geojson,.json,.zip,.csv,.shp"
        className="hidden"
        onChange={(e) => {
          void handleFiles(e.target.files);
          if (inputRef.current) inputRef.current.value = "";
        }}
      />
      {busy ? (
        <Loader2 className="h-3.5 w-3.5 animate-spin" strokeWidth={1.5} />
      ) : (
        <Upload className="h-3.5 w-3.5" strokeWidth={1.5} />
      )}
      <span>
        {busy
          ? "Uploading…"
          : dragOver
            ? "Drop to upload"
            : "Drag & drop or click · GeoJSON, CSV, SHP.zip"}
      </span>
      {error ? (
        <span className="flex items-center gap-1 text-err">
          <XCircle className="h-3 w-3" strokeWidth={1.75} />
          <span className="truncate">{error}</span>
        </span>
      ) : null}
    </div>
  );
}
