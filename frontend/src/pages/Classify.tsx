import clsx from "clsx";
import { AlertTriangle, CheckCircle2, ImageUp, RotateCcw, XCircle } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { ErrorBox, PageHeader, Spinner, UrgencyBadge } from "../components/ui";
import { useReport } from "../hooks/useReport";
import { api, type PredictionResponse, type Sample } from "../lib/api";
import { pct } from "../lib/format";

const ACCEPT = "image/png,image/jpeg,image/bmp,image/tiff,image/webp,image/gif";
const MAX_MB = 8;

function PixelCanvas({ pixels, size = 168 }: { pixels: number[][]; size?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const h = pixels.length;
    const w = pixels[0]?.length ?? 0;
    const img = ctx.createImageData(w, h);
    pixels.forEach((row, y) =>
      row.forEach((v, x) => {
        const o = (y * w + x) * 4;
        img.data[o] = img.data[o + 1] = img.data[o + 2] = v;
        img.data[o + 3] = 255;
      })
    );
    ctx.putImageData(img, 0, 0);
  }, [pixels]);
  return (
    <canvas
      ref={ref}
      width={pixels[0]?.length ?? 28}
      height={pixels.length || 28}
      className="pixelated rounded-lg ring-1 ring-stone-200 dark:ring-stone-700"
      style={{ width: size, height: size }}
      role="img"
      aria-label="28 by 28 pixel grayscale input the network receives"
    />
  );
}

function ProbabilityBars({ result }: { result: PredictionResponse }) {
  const sorted = [...result.probabilities].sort((a, b) => b.probability - a.probability);
  return (
    <ul className="space-y-3">
      {sorted.map((p) => {
        const top = p.code === result.prediction.code;
        return (
          <li key={p.code}>
            <div className="mb-1 flex items-baseline justify-between gap-3 text-sm">
              <span className={clsx(top ? "font-semibold" : "muted")}>
                {p.name} <span className="font-mono text-xs muted">({p.code})</span>
              </span>
              <span className="font-mono tabular-nums">{pct(p.probability)}</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-stone-100 dark:bg-stone-800">
              <div
                className={clsx("h-full rounded-full transition-[width] duration-500", top ? "bg-brand-600 dark:bg-brand-400" : "bg-stone-400 dark:bg-stone-600")}
                style={{ width: `${Math.max(p.probability * 100, 0.5)}%` }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}

const STEPS = ["Upload", "Grayscale + crop", "Resize 28×28", "Features", "MLP forward pass", "Softmax"];

export default function ClassifyPage() {
  const { report } = useReport();
  const [samples, setSamples] = useState<Sample[]>([]);
  const [preview, setPreview] = useState<string | null>(null);
  const [truth, setTruth] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.samples().then(setSamples).catch(() => setSamples([]));
  }, []);

  useEffect(() => () => void (preview?.startsWith("blob:") && URL.revokeObjectURL(preview)), [preview]);

  const classify = useCallback(async (blob: Blob, name: string, previewUrl: string, label: string | null) => {
    setError(null);
    setResult(null);
    setTruth(label);
    setPreview(previewUrl);
    if (blob.size > MAX_MB * 1024 * 1024) {
      setError(`That file is larger than ${MAX_MB} MB.`);
      return;
    }
    setBusy(true);
    try {
      setResult(await api.predict(blob, name));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setBusy(false);
    }
  }, []);

  const onFile = useCallback(
    (file: File | undefined | null) => {
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        setError("Please choose an image file (PNG, JPEG, BMP, TIFF or WebP).");
        return;
      }
      classify(file, file.name, URL.createObjectURL(file), null);
    },
    [classify]
  );

  const onSample = async (s: Sample) => {
    try {
      const url = api.assetUrl(s.url);
      const blob = await (await fetch(url)).blob();
      classify(blob, s.file, url, s.label);
    } catch {
      setError("Could not load the sample image.");
    }
  };

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = Array.from(e.clipboardData?.files ?? []).find((f) => f.type.startsWith("image/"));
      if (file) onFile(file);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [onFile]);

  const reset = () => {
    setResult(null);
    setPreview(null);
    setTruth(null);
    setError(null);
  };

  const testAcc = report?.final.test.accuracy;

  return (
    <div>
      <PageHeader eyebrow="Image classification" title="Classify a retinal OCT scan">
        Upload an optical coherence tomography (OCT) B-scan of the retina. A multilayer perceptron written from scratch in
        NumPy sorts it into one of four classes: <strong className="font-medium text-stone-900 dark:text-stone-100">CNV, DME, Drusen or Normal</strong>.
        {testAcc !== undefined && (
          <>
            {" "}
            On the held-out test set it is correct {pct(testAcc)} of the time.{" "}
            <Link to="/model" className="font-medium text-brand-700 underline-offset-2 hover:underline dark:text-brand-400">
              See the full evaluation
            </Link>
            .
          </>
        )}
      </PageHeader>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
        {/* ------------------------------------------------------------ input */}
        <div className="space-y-6">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              onFile(e.dataTransfer.files?.[0]);
            }}
            className={clsx(
              "card flex flex-col items-center justify-center border-2 border-dashed !p-8 text-center transition-colors",
              dragging ? "border-brand-500 bg-brand-50 dark:bg-brand-900/20" : "border-stone-300 dark:border-stone-700"
            )}
          >
            {preview ? (
              <img src={preview} alt="Selected scan" className="max-h-64 w-auto rounded-lg object-contain pixelated" />
            ) : (
              <ImageUp className="h-10 w-10 text-stone-400" aria-hidden="true" />
            )}
            <p className="mt-4 text-sm font-medium">Drag an image here, paste it, or</p>
            <div className="mt-3 flex flex-wrap justify-center gap-2">
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="rounded-full bg-brand-700 px-5 py-2 text-sm font-medium text-white hover:bg-brand-800 dark:bg-brand-500 dark:text-stone-950 dark:hover:bg-brand-400"
              >
                Choose image
              </button>
              {(preview || result) && (
                <button
                  type="button"
                  onClick={reset}
                  className="inline-flex items-center gap-1.5 rounded-full px-4 py-2 text-sm font-medium text-stone-700 ring-1 ring-stone-300 hover:bg-stone-100 dark:text-stone-300 dark:ring-stone-700 dark:hover:bg-stone-800"
                >
                  <RotateCcw className="h-4 w-4" aria-hidden="true" /> Reset
                </button>
              )}
            </div>
            <p className="mt-3 text-xs muted">PNG, JPEG, BMP, TIFF or WebP · up to {MAX_MB} MB</p>
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPT}
              className="sr-only"
              aria-label="Upload OCT image"
              onChange={(e) => {
                onFile(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
          </div>

          {samples.length > 0 && (
            <div className="card">
              <h2 className="text-sm font-semibold">No scan at hand? Try a test-set image</h2>
              <p className="mt-1 text-xs muted">
                These come from the held-out OCTMNIST test split, which the model never saw during training or tuning.
              </p>
              <div className="mt-4 grid grid-cols-4 gap-3 sm:grid-cols-6">
                {samples.map((s) => (
                  <button
                    key={s.file}
                    type="button"
                    onClick={() => onSample(s)}
                    className="group text-center"
                    aria-label={`Classify sample ${s.name}`}
                  >
                    <img
                      src={api.assetUrl(s.url)}
                      alt=""
                      loading="lazy"
                      className="pixelated aspect-square w-full rounded-lg ring-1 ring-stone-200 transition group-hover:ring-2 group-hover:ring-brand-500 dark:ring-stone-700"
                    />
                    <span className="mt-1 block font-mono text-[11px] muted">{s.label}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ------------------------------------------------------------ output */}
        <div className="card min-h-[320px]" aria-live="polite">
          {busy && <Spinner label="Running the network…" />}
          {error && <ErrorBox message={error} />}
          {!busy && !error && !result && (
            <div className="flex h-full flex-col justify-center">
              <p className="eyebrow">How a prediction is made</p>
              <ol className="mt-4 space-y-3">
                {STEPS.map((s, i) => (
                  <li key={s} className="flex items-center gap-3 text-sm">
                    <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-stone-100 font-mono text-xs dark:bg-stone-800">
                      {i + 1}
                    </span>
                    {s}
                  </li>
                ))}
              </ol>
              {report && (
                <p className="mt-6 text-sm muted">
                  Network: {report.final.layers.join(" → ")} neurons · {report.final.parameters.toLocaleString()} trained
                  weights.
                </p>
              )}
            </div>
          )}
          {result && !busy && (
            <div>
              <p className="eyebrow">Prediction</p>
              <div className="mt-2 flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h2 className="text-2xl font-semibold tracking-tight">{result.prediction.name}</h2>
                  <p className="mt-0.5 text-sm muted">
                    Confidence <span className="font-mono tabular-nums">{pct(result.prediction.probability)}</span> ·{" "}
                    {result.latency_ms.toFixed(1)} ms
                  </p>
                </div>
                <UrgencyBadge urgency={result.prediction.urgency} />
              </div>
              <p className="mt-3 text-sm leading-relaxed muted">{result.prediction.description}</p>

              {truth && (
                <p
                  className={clsx(
                    "mt-4 inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm",
                    truth === result.prediction.code
                      ? "bg-emerald-50 text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-200"
                      : "bg-red-50 text-red-800 dark:bg-red-950/50 dark:text-red-200"
                  )}
                >
                  {truth === result.prediction.code ? (
                    <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
                  ) : (
                    <XCircle className="h-4 w-4" aria-hidden="true" />
                  )}
                  Ground truth: {truth}. {truth === result.prediction.code ? "Correct." : "Misclassified."}
                </p>
              )}

              {result.warnings.map((w) => (
                <p key={w} className="mt-4 flex gap-2 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:bg-amber-950/50 dark:text-amber-200">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                  {w}
                </p>
              ))}

              <div className="mt-6 grid gap-6 sm:grid-cols-[auto_1fr] sm:items-start">
                <figure className="flex flex-col items-center">
                  <PixelCanvas pixels={result.input_pixels} />
                  <figcaption className="mt-2 max-w-[168px] text-center text-xs muted">
                    What the network sees: 28 × 28 grayscale, from a {result.image_size[0]} × {result.image_size[1]} upload
                  </figcaption>
                </figure>
                <div>
                  <h3 className="mb-3 text-sm font-semibold">Softmax output (class probabilities)</h3>
                  <ProbabilityBars result={result} />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
