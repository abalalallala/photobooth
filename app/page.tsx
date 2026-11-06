"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { DropperFilterRenderer, FILTER_VERSION } from "./lib/dropperFilter";

type CameraState = "initializing" | "ready" | "denied" | "error";
type BoothPhase = "camera" | "capturing" | "results";

const TOTAL_SHOTS = 4;
const FRAME_WIDTH = 960;
const FRAME_HEIGHT = 720; // Classic 4:3 photo booth aspect ratio
const STRIP_BORDER = 56;
const STRIP_GAP = 36;
const STRIP_BOTTOM_BORDER = 96;

export default function PhotoBoothPage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const glCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<DropperFilterRenderer | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isMountedRef = useRef<boolean>(true);
  const captureCancelRef = useRef<boolean>(false);

  const [cameraState, setCameraState] = useState<CameraState>("initializing");
  const [phase, setPhase] = useState<BoothPhase>("camera");
  const [countdown, setCountdown] = useState<number | null>(null);
  const [countdownKey, setCountdownKey] = useState<number>(0);
  const [currentShot, setCurrentShot] = useState<number>(0);
  const [isFlashing, setIsFlashing] = useState<boolean>(false);
  const [stripUrl, setStripUrl] = useState<string | null>(null);
  const [stripFile, setStripFile] = useState<File | null>(null);
  const [saveFeedback, setSaveFeedback] = useState<boolean>(false);

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }, []);

  const requestCameraStream = useCallback(async (): Promise<boolean> => {
    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices ||
      !navigator.mediaDevices.getUserMedia
    ) {
      return false;
    }

    try {
      stopStream();

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "user" },
          width: { ideal: 1920 },
          height: { ideal: 1440 },
        },
        audio: false,
      });

      if (!isMountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return false;
      }

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }

      return true;
    } catch {
      return false;
    }
  }, [stopStream]);

  const handleRetryCamera = useCallback(async () => {
    setCameraState("initializing");
    const ok = await requestCameraStream();
    if (isMountedRef.current) {
      setCameraState(ok ? "ready" : "denied");
    }
  }, [requestCameraStream]);

  useEffect(() => {
    isMountedRef.current = true;
    requestCameraStream().then((ok) => {
      if (isMountedRef.current) {
        setCameraState(ok ? "ready" : "denied");
      }
    });

    return () => {
      isMountedRef.current = false;
      captureCancelRef.current = true;
      stopStream();
    };
  }, [requestCameraStream, stopStream]);

  // Initialize WebGL renderer and animation loop
  useEffect(() => {
    void FILTER_VERSION;
    const canvas = glCanvasRef.current;
    if (!canvas) return;

    const renderer = new DropperFilterRenderer(canvas);
    rendererRef.current = renderer;

    let rafId = 0;
    const renderLoop = () => {
      if (videoRef.current && videoRef.current.readyState >= 2) {
        renderer.render(videoRef.current);
      }
      rafId = requestAnimationFrame(renderLoop);
    };

    rafId = requestAnimationFrame(renderLoop);

    return () => {
      cancelAnimationFrame(rafId);
      renderer.dispose();
      rendererRef.current = null;
    };
  }, []);

  // Ensure video stays playing when returning to camera view
  useEffect(() => {
    if (phase === "camera" && videoRef.current && streamRef.current) {
      if (videoRef.current.srcObject !== streamRef.current) {
        videoRef.current.srcObject = streamRef.current;
      }
      if (videoRef.current.paused) {
        videoRef.current.play().catch(() => {});
      }
    }
  }, [phase]);

  // Revoke object URL on cleanup
  useEffect(() => {
    return () => {
      if (stripUrl && stripUrl.startsWith("blob:")) {
        URL.revokeObjectURL(stripUrl);
      }
    };
  }, [stripUrl]);

  const captureSingleFrame = useCallback((): HTMLCanvasElement | null => {
    const glCanvas = glCanvasRef.current;
    const video = videoRef.current;

    if (video && rendererRef.current && video.readyState >= 2) {
      rendererRef.current.render(video);
    }

    const canvas = document.createElement("canvas");
    canvas.width = FRAME_WIDTH;
    canvas.height = FRAME_HEIGHT;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    const targetAspect = FRAME_WIDTH / FRAME_HEIGHT;

    if (glCanvas && glCanvas.width > 0 && glCanvas.height > 0) {
      const swTotal = glCanvas.width;
      const shTotal = glCanvas.height;
      const srcAspect = swTotal / shTotal;

      let sx = 0;
      let sy = 0;
      let sw = swTotal;
      let sh = shTotal;

      if (srcAspect > targetAspect) {
        sw = shTotal * targetAspect;
        sx = (swTotal - sw) / 2;
      } else {
        sh = swTotal / targetAspect;
        sy = (shTotal - sh) / 2;
      }

      ctx.drawImage(glCanvas, sx, sy, sw, sh, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
      return canvas;
    }

    if (!video || video.readyState < 2) return null;
    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 960;
    const videoAspect = vw / vh;

    let sx = 0;
    let sy = 0;
    let sw = vw;
    let sh = vh;

    if (videoAspect > targetAspect) {
      sw = vh * targetAspect;
      sx = (vw - sw) / 2;
    } else {
      sh = vw / targetAspect;
      sy = (vh - sh) / 2;
    }

    ctx.save();
    ctx.translate(FRAME_WIDTH, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
    ctx.restore();

    return canvas;
  }, []);

  const buildStripFromFrames = useCallback(
    async (
      frames: HTMLCanvasElement[]
    ): Promise<{ url: string; file: File } | null> => {
      const stripWidth = FRAME_WIDTH + STRIP_BORDER * 2;
      const stripHeight =
        STRIP_BORDER +
        FRAME_HEIGHT * TOTAL_SHOTS +
        STRIP_GAP * (TOTAL_SHOTS - 1) +
        STRIP_BOTTOM_BORDER;

      const canvas = document.createElement("canvas");
      canvas.width = stripWidth;
      canvas.height = stripHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;

      ctx.fillStyle = "#FFFFFF";
      ctx.fillRect(0, 0, stripWidth, stripHeight);

      frames.forEach((frame, index) => {
        const x = STRIP_BORDER;
        const y = STRIP_BORDER + index * (FRAME_HEIGHT + STRIP_GAP);
        ctx.drawImage(frame, x, y, FRAME_WIDTH, FRAME_HEIGHT);
      });

      const blob: Blob | null = await new Promise((resolve) =>
        canvas.toBlob((b) => resolve(b), "image/png", 1.0)
      );

      if (!blob) return null;

      const file = new File([blob], "photo-strip.png", {
        type: "image/png",
        lastModified: Date.now(),
      });
      const url = URL.createObjectURL(blob);
      return { url, file };
    },
    []
  );

  const sleep = (ms: number) =>
    new Promise<void>((resolve) => {
      setTimeout(resolve, ms);
    });

  const handleShutterTap = useCallback(async () => {
    if (phase !== "camera" || cameraState !== "ready") return;

    captureCancelRef.current = false;
    setPhase("capturing");
    setCurrentShot(0);

    const capturedFrames: HTMLCanvasElement[] = [];

    for (let shot = 1; shot <= TOTAL_SHOTS; shot++) {
      if (captureCancelRef.current || !isMountedRef.current) return;
      setCurrentShot(shot);

      for (const count of [3, 2, 1]) {
        if (captureCancelRef.current || !isMountedRef.current) return;
        setCountdown(count);
        setCountdownKey((k) => k + 1);
        await sleep(850);
      }

      if (captureCancelRef.current || !isMountedRef.current) return;
      setCountdown(null);

      setIsFlashing(true);
      const frame = captureSingleFrame();
      if (frame) {
        capturedFrames.push(frame);
      }

      await sleep(140);
      if (!isMountedRef.current) return;
      setIsFlashing(false);

      if (shot < TOTAL_SHOTS) {
        await sleep(420);
      } else {
        await sleep(260);
      }
    }

    if (captureCancelRef.current || !isMountedRef.current) return;

    const result = await buildStripFromFrames(capturedFrames);
    if (!result || !isMountedRef.current) {
      setPhase("camera");
      return;
    }

    setStripUrl((prev) => {
      if (prev && prev.startsWith("blob:")) URL.revokeObjectURL(prev);
      return result.url;
    });
    setStripFile(result.file);
    setPhase("results");
  }, [phase, cameraState, captureSingleFrame, buildStripFromFrames]);

  const handleRetake = useCallback(() => {
    setSaveFeedback(false);
    setCountdown(null);
    setCurrentShot(0);
    setPhase("camera");
  }, []);

  const triggerDownload = useCallback((url: string) => {
    const link = document.createElement("a");
    link.href = url;
    link.download = `photo-strip-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setSaveFeedback(true);
    setTimeout(() => {
      if (isMountedRef.current) setSaveFeedback(false);
    }, 2000);
  }, []);

  const handleSave = useCallback(async () => {
    if (!stripUrl || !stripFile) return;

    const ua = typeof navigator !== "undefined" ? navigator.userAgent : "";
    const isTouchMac =
      /Macintosh/i.test(ua) &&
      typeof navigator !== "undefined" &&
      navigator.maxTouchPoints > 1;
    const isPhoneOrTablet =
      /iPhone|iPad|iPod|Android|Mobile/i.test(ua) || isTouchMac;

    if (
      isPhoneOrTablet &&
      typeof navigator !== "undefined" &&
      typeof navigator.share === "function"
    ) {
      const canShareFiles =
        typeof navigator.canShare === "function"
          ? navigator.canShare({ files: [stripFile] })
          : true;

      if (canShareFiles) {
        try {
          await navigator.share({
            files: [stripFile],
          });
          return;
        } catch (err) {
          if (err instanceof Error && err.name === "AbortError") {
            return;
          }
        }
      }
    }

    triggerDownload(stripUrl);
  }, [stripUrl, stripFile, triggerDownload]);

  return (
    <main className="relative h-dvh w-screen overflow-hidden bg-black text-white">
      {/* Hidden source video element feeding the WebGL shader */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="pointer-events-none fixed top-0 left-0 h-px w-px opacity-0"
      />

      {/* Live WebGL Canvas */}
      <canvas
        ref={glCanvasRef}
        className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-300 ${
          cameraState === "ready" && phase !== "results"
            ? "opacity-100"
            : "opacity-0 pointer-events-none"
        }`}
      />

      {/* Subtle top/bottom gradient vignette on camera view for native iOS legibility */}
      {cameraState === "ready" && phase !== "results" && (
        <>
          <div className="pointer-events-none absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-black/50 to-transparent" />
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-44 bg-gradient-to-t from-black/65 to-transparent" />
        </>
      )}

      {/* White Screen Flash Overlay */}
      <div
        aria-hidden="true"
        className={`pointer-events-none fixed inset-0 z-50 bg-white transition-opacity ${
          isFlashing ? "opacity-100 duration-0" : "opacity-0 duration-300"
        }`}
      />

      {/* Camera Permission Denied / Friendly Help State */}
      {cameraState === "denied" && (
        <div
          className="relative z-30 flex h-full w-full flex-col items-center justify-between px-6 text-center"
          style={{
            paddingTop: "max(calc(env(safe-area-inset-top, 0px) + 24px), 32px)",
            paddingBottom:
              "max(calc(env(safe-area-inset-bottom, 0px) + 24px), 32px)",
          }}
        >
          <div />

          <div className="flex max-w-sm flex-col items-center">
            <div className="mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-white/10 ring-1 ring-white/15 backdrop-blur-md">
              <svg
                className="h-9 w-9 text-white"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
                <line x1="2" y1="2" x2="22" y2="22" />
              </svg>
            </div>

            <h1 className="text-2xl font-semibold tracking-tight text-white">
              Camera Access Needed
            </h1>
            <p className="mt-2.5 text-[15px] leading-relaxed text-white/70">
              Photo Booth needs access to your front camera to take your 4-photo
              strip.
            </p>

            <div className="mt-6 w-full rounded-2xl bg-white/[0.08] p-4 text-left text-sm leading-relaxed text-white/85 ring-1 ring-white/10 backdrop-blur-md">
              <p className="mb-2.5 font-medium text-white">
                How to turn camera access back on:
              </p>
              <ol className="list-decimal space-y-2 pl-4 text-white/75">
                <li>
                  On iPhone, tap the{" "}
                  <span className="font-medium text-white">Page Settings</span>{" "}
                  icon (or <span className="font-medium text-white">aA</span>)
                  in the Safari address bar and select{" "}
                  <span className="font-medium text-white">
                    Website Settings
                  </span>
                  .
                </li>
                <li>
                  Set <span className="font-medium text-white">Camera</span> to{" "}
                  <span className="font-medium text-white">Allow</span> (or
                  check{" "}
                  <span className="font-medium text-white">
                    Settings → Safari → Camera
                  </span>
                  ).
                </li>
                <li>
                  Tap <span className="font-medium text-white">Try Again</span>{" "}
                  below.
                </li>
              </ol>
            </div>
          </div>

          <button
            type="button"
            onClick={handleRetryCamera}
            className="w-full max-w-xs rounded-full bg-white py-4 text-base font-semibold text-black shadow-lg transition-transform active:scale-95"
          >
            Try Again
          </button>
        </div>
      )}

      {/* Camera Live Preview & Countdown UI */}
      {cameraState !== "denied" && phase !== "results" && (
        <div
          className="relative z-20 flex h-full w-full flex-col items-center justify-between"
          style={{
            paddingTop: "max(calc(env(safe-area-inset-top, 0px) + 14px), 20px)",
            paddingBottom:
              "max(calc(env(safe-area-inset-bottom, 0px) + 20px), 28px)",
            paddingLeft: "max(env(safe-area-inset-left, 0px), 20px)",
            paddingRight: "max(env(safe-area-inset-right, 0px), 20px)",
          }}
        >
          {/* Top Status Pill */}
          <div className="flex flex-col items-center gap-2.5">
            <div className="flex items-center gap-2.5 rounded-full bg-black/45 px-4 py-1.5 text-xs font-medium tracking-wide text-white/95 ring-1 ring-white/15 backdrop-blur-md">
              {phase === "capturing" ? (
                <>
                  <span className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
                  <span>
                    Photo {currentShot} of {TOTAL_SHOTS}
                  </span>
                </>
              ) : (
                <span>4-POSE PHOTO STRIP</span>
              )}
            </div>

            {/* 4-Shot Progress Dots */}
            {phase === "capturing" && (
              <div className="flex items-center gap-2">
                {[1, 2, 3, 4].map((dot) => (
                  <span
                    key={dot}
                    className={`h-2 w-2 rounded-full transition-all duration-300 ${
                      dot < currentShot
                        ? "bg-white scale-100"
                        : dot === currentShot
                        ? "bg-yellow-400 scale-125"
                        : "bg-white/30"
                    }`}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Center 3-2-1 Countdown Display */}
          <div className="flex flex-1 items-center justify-center">
            {countdown !== null && (
              <span
                key={countdownKey}
                className="animate-countdown text-[128px] font-bold leading-none tracking-tighter text-white drop-shadow-[0_6px_28px_rgba(0,0,0,0.65)]"
              >
                {countdown}
              </span>
            )}
          </div>

          {/* Bottom Shutter Bar */}
          <div className="flex flex-col items-center gap-3">
            <button
              type="button"
              aria-label="Take 4 photos"
              disabled={phase === "capturing" || cameraState !== "ready"}
              onClick={handleShutterTap}
              className={`group relative flex h-20 w-20 items-center justify-center rounded-full border-4 border-white p-1.5 shadow-[0_4px_24px_rgba(0,0,0,0.45)] transition-all ${
                phase === "capturing" || cameraState !== "ready"
                  ? "opacity-45 cursor-not-allowed"
                  : "active:scale-95 cursor-pointer"
              }`}
            >
              <span
                className={`block h-full w-full rounded-full bg-white transition-transform duration-150 ${
                  phase === "capturing"
                    ? "scale-75 bg-white/80"
                    : "group-active:scale-90"
                }`}
              />
            </button>
          </div>
        </div>
      )}

      {/* Results Screen with Vertical Photo Strip */}
      {phase === "results" && stripUrl && (
        <div
          className="relative z-30 flex h-full w-full flex-col items-center justify-between bg-zinc-950/95 backdrop-blur-xl"
          style={{
            paddingTop: "max(calc(env(safe-area-inset-top, 0px) + 14px), 20px)",
            paddingBottom:
              "max(calc(env(safe-area-inset-bottom, 0px) + 18px), 24px)",
            paddingLeft: "max(env(safe-area-inset-left, 0px), 20px)",
            paddingRight: "max(env(safe-area-inset-right, 0px), 20px)",
          }}
        >
          {/* Top Header */}
          <div className="flex items-center justify-center">
            <span className="rounded-full bg-white/10 px-4 py-1.5 text-xs font-medium tracking-wide text-white/90 ring-1 ring-white/10">
              Your Photo Strip
            </span>
          </div>

          {/* Photo Strip Preview (scaled cleanly to fit between safe areas) */}
          <div className="my-3 flex min-h-0 flex-1 w-full items-center justify-center overflow-hidden">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={stripUrl}
              alt="4-photo vertical strip"
              className="animate-strip-reveal max-h-full w-auto max-w-full object-contain rounded-[2px] shadow-[0_20px_60px_rgba(0,0,0,0.75)] ring-1 ring-white/15"
            />
          </div>

          {/* Bottom Action Buttons: Retake & Save */}
          <div className="flex w-full max-w-sm items-center gap-3.5 pt-1">
            <button
              type="button"
              onClick={handleRetake}
              className="flex-1 rounded-full bg-white/12 py-4 text-base font-semibold text-white ring-1 ring-white/15 backdrop-blur-md transition-transform active:scale-95 cursor-pointer"
            >
              Retake
            </button>

            <button
              type="button"
              onClick={handleSave}
              className="flex-1 rounded-full bg-white py-4 text-base font-semibold text-black shadow-[0_4px_20px_rgba(255,255,255,0.2)] transition-transform active:scale-95 cursor-pointer"
            >
              {saveFeedback ? "Saved!" : "Save"}
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
