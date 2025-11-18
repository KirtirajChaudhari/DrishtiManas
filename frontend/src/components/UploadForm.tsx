import { useEffect, useRef, useState, type ChangeEvent, type FormEvent } from "react";

import { fetchDiseaseCatalog } from "../lib/api";
import type { PredictionPayload } from "../types/prediction";

interface UploadFormProps {
  onSubmit: (payload: PredictionPayload) => Promise<void>;
}

const DEFAULT_DISEASE_LIBRARY = [
  "Glaucoma",
  "Cataract",
  "Diabetic Retinopathy",
  "Hypertensive Retinopathy",
  "Dry Eye",
  "AMD",
  "Optic Neuritis",
  "Neurodegeneration",
  "Stroke Risk"
];

const MODALITIES = ["fundus", "retina", "oct", "slit_lamp"];

export default function UploadForm({ onSubmit }: UploadFormProps) {
  const [patientId, setPatientId] = useState("");
  const [modality, setModality] = useState("fundus");
  const [diseaseLibrary, setDiseaseLibrary] = useState<string[]>(DEFAULT_DISEASE_LIBRARY);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [isCatalogLoading, setIsCatalogLoading] = useState(false);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imageFilename, setImageFilename] = useState<string | null>(null);
  const [notes, setNotes] = useState("");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [cameraSupported, setCameraSupported] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const payload: PredictionPayload = {
      patient_id: patientId || `patient-${Date.now()}`,
      modality,
      image_path: imageFilename || "uploaded_image.jpg",
      ...(imageBase64
        ? {
            image_base64: imageBase64,
            image_filename: imageFilename ?? undefined
          }
        : {})
    };
    await onSubmit(payload);
    setNotes("");
  };

  useEffect(() => {
    let ignore = false;
    const loadCatalog = async () => {
      setIsCatalogLoading(true);
      setCatalogError(null);
      try {
        const catalog = await fetchDiseaseCatalog();
        if (!ignore && catalog?.all?.length) {
          setDiseaseLibrary(catalog.all);
        }
      } catch (err) {
        if (!ignore) {
          const message = err instanceof Error ? err.message : "Unable to load disease catalog";
          setCatalogError(message);
        }
      } finally {
        if (!ignore) {
          setIsCatalogLoading(false);
        }
      }
    };

    loadCatalog();

    setCameraSupported(!!navigator.mediaDevices?.getUserMedia);

    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [cameraStream]);

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        setImageBase64(reader.result);
        setImageFilename(file.name);
      }
    };
    reader.readAsDataURL(file);
  };

  const clearLocalImage = () => {
    setImageBase64(null);
    setImageFilename(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
    }
    setCameraStream(null);
    setIsCameraActive(false);
  };

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Camera capture is not supported on this device");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false
      });
      setCameraStream(stream);
      setIsCameraActive(true);
      setCameraError(null);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to access camera";
      setCameraError(message);
    }
  };

  const captureFromCamera = () => {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const context = canvas.getContext("2d");
    if (!context) {
      setCameraError("Unable to capture frame from camera");
      return;
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
    const filename = `capture-${Date.now()}.jpg`;
    setImageBase64(dataUrl);
    setImageFilename(filename);
    stopCamera();
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6 rounded-3xl bg-white p-6 shadow-sm">
      <div className="grid gap-4 md:grid-cols-2">
        <label className="text-sm font-medium text-slate-600">
          Patient ID
          <input
            value={patientId}
            onChange={(event: ChangeEvent<HTMLInputElement>) => setPatientId(event.target.value)}
            placeholder="MRN / UID"
            className="mt-2 w-full rounded-xl border border-slate-200 px-4 py-2 focus:border-primary focus:outline-none"
          />
        </label>
        <label className="text-sm font-medium text-slate-600">
          Modality
          <select
            value={modality}
            onChange={(event: ChangeEvent<HTMLSelectElement>) => setModality(event.target.value)}
            className="mt-2 w-full rounded-xl border border-slate-200 px-4 py-2 focus:border-primary focus:outline-none"
          >
            {MODALITIES.map((item) => (
              <option key={item} value={item}>
                {item.toUpperCase()}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="rounded-3xl border border-slate-100 bg-slate-50 p-4">
        <p className="text-sm font-semibold text-slate-700">Screening coverage</p>
        <p className="mt-1 text-xs text-slate-500">
          Drishti automatically evaluates every supported disease in a single pass. Select a modality and the
          platform reports likelihood percentages for each ocular and neurological condition we track.
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          {diseaseLibrary.map((disease) => (
            <span
              key={disease}
              className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-600 shadow-sm"
            >
              {disease}
            </span>
          ))}
          {isCatalogLoading && (
            <span className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-500 shadow-sm">
              Syncing catalog...
            </span>
          )}
        </div>
        {catalogError && <p className="mt-2 text-xs text-red-500">{catalogError}</p>}
      </div>

      <div>
        <p className="text-sm font-medium text-slate-600">Upload or capture image</p>
        <label className="mt-2 flex flex-col rounded-2xl border border-dashed border-slate-300 p-4 text-center text-sm text-slate-500">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleFileChange}
            className="hidden"
          />
          <span className="text-slate-700">Browse files or scan from device</span>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="mt-3 rounded-full border border-primary px-4 py-1 text-primary transition hover:bg-primary/10"
          >
            Choose image
          </button>
        </label>
        {cameraSupported ? (
          <div className="mt-3 space-y-2">
            <button
              type="button"
              onClick={isCameraActive ? stopCamera : startCamera}
              className="w-full rounded-2xl border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-600 transition hover:border-primary hover:text-primary"
            >
              {isCameraActive ? "Close camera" : "Use camera"}
            </button>
            {cameraError && <p className="text-xs text-red-500">{cameraError}</p>}
          </div>
        ) : (
          <p className="mt-3 text-xs text-slate-500">Camera not available on this device</p>
        )}
      </div>

      {isCameraActive && (
        <div className="space-y-3 rounded-3xl border border-slate-200 p-4">
          <p className="text-sm font-semibold text-slate-600">Live camera preview</p>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="h-56 w-full rounded-2xl border border-slate-200 object-cover"
          />
          <div className="flex gap-3">
            <button
              type="button"
              onClick={captureFromCamera}
              className="flex-1 rounded-2xl bg-primary px-4 py-2 text-sm font-semibold text-white"
            >
              Capture photo
            </button>
            <button
              type="button"
              onClick={stopCamera}
              className="flex-1 rounded-2xl border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-600"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {imageBase64 && (
        <div className="space-y-3 rounded-3xl border border-slate-200 p-4">
          <div className="flex items-center justify-between">
            <p className="text-sm font-semibold text-slate-600">Selected study preview</p>
            {imageFilename && <span className="text-xs text-slate-500">{imageFilename}</span>}
          </div>
          <img
            src={imageBase64}
            alt="Selected imaging study preview"
            className="h-56 w-full rounded-2xl border border-slate-100 object-contain bg-slate-50"
          />
          <button
            type="button"
            onClick={clearLocalImage}
            className="w-full rounded-2xl border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-600"
          >
            Remove image
          </button>
        </div>
      )}

      <label className="block text-sm font-medium text-slate-600">
        Technician notes
        <textarea
          value={notes}
          onChange={(event: ChangeEvent<HTMLTextAreaElement>) => setNotes(event.target.value)}
          rows={3}
          className="mt-2 w-full rounded-2xl border border-slate-200 px-4 py-2 focus:border-primary focus:outline-none"
          placeholder="Visual acuity, symptoms, medication"
        />
      </label>

      <button
        type="submit"
        className="w-full rounded-2xl bg-primary py-3 text-center text-base font-semibold text-white shadow-sm transition hover:bg-primary-dark"
      >
        Run Screening
      </button>
    </form>
  );
}
