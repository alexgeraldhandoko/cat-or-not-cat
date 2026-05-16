"use client";

import { ChangeEvent, useMemo, useState } from "react";

type PredictionResult = {
  label: "cat" | "not_cat";
  confidence: number;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>();

  const resultText = useMemo(() => {
    if (result === null) {
      return null;
    }

    if (result.label === "cat") {
      return {
        emoji: "😺",
        title: "This is a cat!",
        message: "The avatar is happy because the model found a cat."
      };
    }

    return {
       emoji: "😾",
        title: "This is not a cat!",
        message: "The avatar is angry because this is not a cat."
    };
  }, [result])

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];

    setResult(null);
    setErrorMessage(null);

    if (!file) {
      setSelectedFile(null);
      setPreviewUrl(null);
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  }

  async function handlePrediction() {
    if (!selectedFile) {
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail ?? "Prediction failed");
      }

      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (error) {
      if (error instanceof Error) {
        setErrorMessage(error.message);
      } else {
        setErrorMessage("Something went wrong");
      }
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10 text-white">
      <section className="mx-auto max-w-5xl">
        <div className="mb-10">
          <p className="mb-3 text-sm font-semibold uppercase tracking-widest text-cyan-300">
            Cat or Not Cat
          </p>

          <h1 className="text-4xl font-bold tracking-tight sm:text-6xl">
            Upload an animal image.
          </h1>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-3xl border border-white/10 bg-white/5 p-6">
            <label
              htmlFor="image-upload"
              className="flex min-h-80 cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-900 p-6 text-center"
            >
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Uploaded animal preview"
                  className="max-h-80 rounded-2xl object-contain"
                />
              ) : (
                <div>
                  <div className="text-6xl">🖼️</div>
                  <p className="mt-4 text-xl font-semibold">
                    Click to choose an image
                  </p>
                  <p className="mt-2 text-sm text-slate-400">
                    Upload a cat or non-cat animal image.
                  </p>
                </div>
              )}
            </label>

            <input
              id="image-upload"
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileChange}
            />

            <button
              onClick={handlePrediction}
              disabled={!selectedFile || isLoading}
              className="mt-5 w-full rounded-xl bg-cyan-400 px-5 py-3 font-bold text-slate-950 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isLoading ? "Classifying..." : "Classify image"}
            </button>

            {selectedFile && (
              <p className="mt-4 text-sm text-slate-400">
                Selected file: {selectedFile.name}
              </p>
            )}

            {errorMessage && (
              <p className="mt-4 rounded-xl bg-red-500/10 p-4 text-sm text-red-300">
                {errorMessage}
              </p>
            )}
          </div>

          <div className="rounded-3xl bg-white p-6 text-slate-950">
            <p className="text-sm font-semibold uppercase tracking-widest text-slate-500">
              Result
            </p>

            {resultText === null ? (
              <div className="mt-8 flex min-h-80 flex-col items-center justify-center rounded-2xl bg-slate-100 text-center">
                <div className="text-7xl">🐾</div>
                <h2 className="mt-5 text-2xl font-bold">
                  Waiting for an image
                </h2>
                <p className="mt-3 max-w-sm text-slate-600">
                  Choose an image, then click the button to see the result card.
                </p>
              </div>
            ) : (
              <div className="mt-8 flex min-h-80 flex-col items-center justify-center rounded-2xl bg-slate-100 p-6 text-center">
                <div className="text-8xl">{resultText.emoji}</div>
                <h2 className="mt-6 text-3xl font-extrabold">
                  {resultText.title}
                </h2>
                <p className="mt-3 text-slate-600">{resultText.message}</p>
                <p className="mt-6 rounded-full bg-slate-950 px-4 py-2 text-sm font-bold text-white">
                  Confidence: {(result!.confidence * 100).toFixed(1)}%
                </p>
              </div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}