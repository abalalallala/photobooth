import {
  FaceDetector,
  FaceLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";

export const FILTER_VERSION = 4;

export interface TrackedFace {
  /** Center between the eyes (nasion, landmark 168) in canvas pixel coordinates */
  cx: number;
  cy: number;
  /** Horizontal half-face width (cheekbone to center) in canvas pixels */
  rx: number;
  /** Vertical lower-face height (eye center to chin tip) in canvas pixels */
  ry: number;
  /** Head roll angle in radians (0 = upright) */
  angle: number;
  /** Left eye in local face coordinates (normalized by rx, ry) */
  leftEyeLx: number;
  leftEyeLy: number;
  /** Right eye in local face coordinates (normalized by rx, ry) */
  rightEyeLx: number;
  rightEyeLy: number;
}

const MAX_FACES = 3;

const VERTEX_SHADER_SOURCE = `
attribute vec2 a_position;
varying vec2 v_uv;

void main() {
  v_uv = vec2(a_position.x * 0.5 + 0.5, 0.5 - a_position.y * 0.5);
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FRAGMENT_SHADER_SOURCE = `
precision highp float;

varying vec2 v_uv;
uniform sampler2D u_video;
uniform vec2 u_resolution;
uniform vec2 u_cropScale;
uniform int u_faceCount;
uniform vec4 u_faces[${MAX_FACES}];  // (cx, cy, rx, ry)
uniform vec4 u_eyes[${MAX_FACES}];   // (leftEyeLx, leftEyeLy, rightEyeLx, rightEyeLy)
uniform float u_angles[${MAX_FACES}];

struct WarpResult {
  vec2 pos;
  float faceSkinRegion;
  float underEyeOrJawShadow;
  float lipRegion;
  float cheekBlush;
};

WarpResult warpAndMask(vec2 pos) {
  WarpResult res;
  res.pos = pos;
  res.faceSkinRegion = 0.0;
  res.underEyeOrJawShadow = 0.0;
  res.lipRegion = 0.0;
  res.cheekBlush = 0.0;

  for (int i = 0; i < ${MAX_FACES}; i++) {
    if (i < u_faceCount) {
      vec4 face = u_faces[i];
      vec4 eyes = u_eyes[i];
      float angle = u_angles[i];

      vec2 center = face.xy;
      float rx = max(face.z, 1.0);
      float ry = max(face.w, 1.0);

      float c = cos(angle);
      float s = sin(angle);
      vec2 rightVec = vec2(c, s);
      vec2 downVec = vec2(-s, c);

      vec2 rel = res.pos - center;
      float lx = dot(rel, rightVec) / rx;
      float ly = dot(rel, downVec) / ry;

      if (abs(lx) <= 3.0 && ly >= -2.5 && ly <= 2.9) {
        float srcLx = lx;
        float srcLy = ly;

        // 1. Upper Hair/Cranium Bulb Contrast (gentle puff high on the hair crown)
        vec2 dCrown = vec2(lx, ly - (-0.72));
        float crownEnv = exp(-(dCrown.x * dCrown.x) / (1.45 * 1.45) - (dCrown.y * dCrown.y) / (0.95 * 0.95))
                       * smoothstep(0.05, -0.35, ly);
        srcLx -= dCrown.x * 0.18 * crownEnv;
        srcLy -= dCrown.y * 0.16 * crownEnv;

        // 2. Vertical Face Elongation & Long Razor-Pointed V-Chin
        float lengthenEnv = exp(-pow(lx / 1.35, 2.0) - pow((ly - 0.95) / 0.85, 2.0)) * smoothstep(-0.10, 0.20, ly);
        float centerTipWeight = exp(-pow(lx / 0.38, 2.0));
        srcLy -= (0.20 + 0.28 * centerTipWeight) * lengthenEnv;

        // 3. UNCANNY EXTREME V-CONE TAPER (Crushes lower cheeks, jaw, and chin narrower than the eyes,
        // while stretching the ears right outside the cheekbones into huge pointed elf ears!)
        float effLy = srcLy;
        float vStart = smoothstep(-0.14, 0.22, effLy);
        float vMidCheek = smoothstep(0.02, 0.52, effLy);
        float vJawChin = exp(-pow((effLy - 0.82) / 0.54, 2.0));
        float vNeck = exp(-pow((effLy - 1.38) / 0.55, 2.0));
        float vFadeBottom = 1.0 - smoothstep(1.45, 2.35, effLy);

        // Total horizontal pull-in displacement (in units of rx)
        float deltaMax = (0.42 * vStart + 0.52 * vMidCheek + 0.56 * vJawChin + 0.42 * vNeck) * vFadeBottom;

        // Core width narrows from 0.30 at cheekbones down to 0.15 at the chin tip
        float wCore = mix(0.30, 0.15, clamp(effLy, 0.0, 1.0));
        float coreRatio = lx / sqrt(wCore * wCore + lx * lx);
        float outerEnvelope = exp(-pow(lx / 2.40, 2.0));

        srcLx += deltaMax * coreRatio * outerEnvelope;

        // 4. Needle-Thin Nose Bridge & Pinched Nostrils
        float noseEnv = exp(-pow(lx / 0.35, 2.0) - pow((effLy - 0.36) / 0.26, 2.0)) * smoothstep(0.0, 0.14, effLy);
        srcLx += (lx / sqrt(0.12 * 0.12 + lx * lx)) * 0.28 * noseEnv;

        // 5. Tiny Doll Mouth
        vec2 dMouth = vec2(lx, effLy - 0.68);
        float mouthEnv = exp(-pow(dMouth.x / 0.32, 2.0) - pow(dMouth.y / 0.18, 2.0));
        srcLx += dMouth.x * 0.45 * mouthEnv;
        srcLy += dMouth.y * 0.32 * mouthEnv;

        // 6. Huge Alien/Anime Eyes
        vec2 dEyeL = vec2(lx - eyes.x, (ly - eyes.y) * (ry / rx));
        float eyeEnvL = exp(-dot(dEyeL, dEyeL) / (0.42 * 0.42));
        srcLx -= (lx - eyes.x) * 0.44 * eyeEnvL;
        srcLy -= (ly - eyes.y) * 0.48 * eyeEnvL;

        vec2 dEyeR = vec2(lx - eyes.z, (ly - eyes.w) * (ry / rx));
        float eyeEnvR = exp(-dot(dEyeR, dEyeR) / (0.42 * 0.42));
        srcLx -= (lx - eyes.z) * 0.44 * eyeEnvR;
        srcLy -= (ly - eyes.w) * 0.48 * eyeEnvR;

        res.pos = center + (srcLx * rx) * rightVec + (srcLy * ry) * downVec;

        // Anatomical masks in source face coordinates
        float foreheadOval = exp(-pow(srcLx / 1.25, 2.0) - pow((effLy - (-0.35)) / 0.65, 2.0));
        float faceOval = exp(-pow(srcLx / 1.28, 2.0) - pow((effLy - 0.35) / 0.95, 2.0));
        float neckArea = exp(-pow(srcLx / 1.20, 2.0) - pow((effLy - 1.40) / 0.90, 2.0));
        res.faceSkinRegion = max(
          res.faceSkinRegion,
          clamp(max(max(foreheadOval, faceOval), neckArea), 0.0, 1.0)
        );

        float underEye = exp(-pow(srcLx / 0.92, 2.0) - pow((effLy - 0.18) / 0.22, 2.0));
        float beardZone = exp(-pow(srcLx / 0.95, 2.0) - pow((effLy - 0.78) / 0.42, 2.0));
        res.underEyeOrJawShadow = max(
          res.underEyeOrJawShadow,
          clamp(underEye + beardZone, 0.0, 1.0)
        );

        float lips = exp(-pow(srcLx / 0.28, 2.0) - pow((effLy - 0.68) / 0.11, 2.0));
        res.lipRegion = max(res.lipRegion, lips);

        float blushL = exp(-pow((srcLx - (-0.50)) / 0.34, 2.0) - pow((effLy - 0.28) / 0.22, 2.0));
        float blushR = exp(-pow((srcLx - 0.50) / 0.34, 2.0) - pow((effLy - 0.28) / 0.22, 2.0));
        res.cheekBlush = max(res.cheekBlush, clamp(blushL + blushR, 0.0, 1.0));
      }
    }
  }

  return res;
}

float computeSkinLikelihood(vec3 rgb) {
  float luma = dot(rgb, vec3(0.299, 0.587, 0.114));
  float cb = -0.168736 * rgb.r - 0.331264 * rgb.g + 0.5 * rgb.b + 0.5;
  float cr = 0.5 * rgb.r - 0.418688 * rgb.g - 0.081312 * rgb.b + 0.5;

  float dCb = (cb - 0.44) / 0.145;
  float dCr = (cr - 0.56) / 0.155;
  float chromaSkin = exp(-0.5 * (dCb * dCb + dCr * dCr));

  float lumaGate = smoothstep(0.06, 0.18, luma);
  return chromaSkin * lumaGate;
}

void main() {
  vec2 startPos = v_uv * u_resolution;
  WarpResult wr = warpAndMask(startPos);

  vec2 warpedUV = wr.pos / u_resolution;
  vec2 videoUV = vec2(
    0.5 + (0.5 - warpedUV.x) * u_cropScale.x,
    0.5 + (warpedUV.y - 0.5) * u_cropScale.y
  );
  videoUV = clamp(videoUV, vec2(0.002), vec2(0.998));

  vec3 centerCol = texture2D(u_video, videoUV).rgb;

  // --- STEP 1: EXTREME Cartoon/Wax-Doll Skin Airbrushing ---
  float resScale = max(1.0, min(u_resolution.x, u_resolution.y) / 480.0);
  vec2 texel = (u_cropScale * resScale) / max(u_resolution, vec2(1.0));

  vec3 accum = centerCol * 0.8;
  float weightSum = 0.8;
  vec3 wideMean = centerCol;

  // Wide 18-tap 3-ring bilateral surface blur with large range tolerance so skin turns into wax/porcelain
  const float sigmaSq = 0.14;
  for (int i = 0; i < 18; i++) {
    float ang = float(i) * 1.04719755;
    float ring = (i < 6) ? 4.5 : ((i < 12) ? 11.0 : 22.0);
    float phase = (i >= 6 && i < 12) ? 0.5235988 : 0.0;
    vec2 offset = vec2(cos(ang + phase), sin(ang + phase)) * ring * texel;
    vec3 sCol = texture2D(u_video, clamp(videoUV + offset, vec2(0.002), vec2(0.998))).rgb;
    vec3 diff = sCol - centerCol;
    float distSq = dot(diff, diff);
    float w = exp(-distSq / sigmaSq);
    accum += sCol * w;
    weightSum += w;
    wideMean += sCol;
  }

  vec3 smoothCol = accum / weightSum;
  wideMean /= 19.0;

  float smoothLuma = dot(smoothCol, vec3(0.299, 0.587, 0.114));
  float centerLuma = dot(centerCol, vec3(0.299, 0.587, 0.114));

  float colorSkin = computeSkinLikelihood(smoothCol);
  float notBlackHairOrFrame = smoothstep(0.08, 0.22, smoothLuma);
  float skinMask = clamp(
    max(colorSkin * 0.95, wr.faceSkinRegion * notBlackHairOrFrame),
    0.0,
    1.0
  );

  // Obliterate all pores, stubble, wrinkles, and shadows into a flat boneless anime/porcelain surface
  vec3 poreErased = max(centerCol, smoothCol);
  vec3 porcelainSmooth = mix(poreErased, smoothCol, 0.99);
  porcelainSmooth = mix(porcelainSmooth, wideMean, 0.52 * skinMask);

  // Completely flatten dark eye bags, nasolabial folds, and beard shadow
  float darkDip = max(0.0, smoothLuma - centerLuma);
  porcelainSmooth += vec3(darkDip * 0.95) + vec3(0.065, 0.060, 0.070) * wr.underEyeOrJawShadow;

  vec3 col = mix(centerCol, porcelainSmooth, skinMask);

  // --- STEP 2: EXTREME Over-Exposed Cold-White Porcelain Skin ---
  // Triple-boost logarithmic + gamma whitening curve
  vec3 logWhite1 = log(1.0 + col * 9.0) / log(10.0);
  vec3 logWhite2 = log(1.0 + logWhite1 * 5.0) / log(6.0);
  vec3 gammaWhite = pow(clamp(logWhite2, 0.0, 1.0), vec3(0.68));
  vec3 whitened = mix(col, gammaWhite, 0.94 * skinMask + 0.26);

  // Strip warm/yellow skin tones and push strongly into glowing cold-porcelain white
  float wLuma = dot(whitened, vec3(0.299, 0.587, 0.114));
  vec3 deYellowed = mix(whitened, vec3(wLuma), 0.62 * skinMask);
  vec3 coldWhite = deYellowed * mix(vec3(1.0), vec3(1.08, 1.02, 1.18), skinMask)
                 + vec3(0.11, 0.095, 0.145) * skinMask;

  // Soft-focus over-exposed beauty glow
  vec3 finalCol = clamp(coldWhite, 0.0, 1.0);
  finalCol += vec3(0.065, -0.015, 0.030) * wr.cheekBlush * skinMask;

  // Vibrant doll-like rosy lips
  float lipMask = wr.lipRegion * smoothstep(0.10, 0.28, centerLuma);
  vec3 glossyLip = vec3(
    min(1.0, finalCol.r * 1.18 + 0.10),
    finalCol.g * 0.72,
    finalCol.b * 0.82 + 0.04
  );
  finalCol = mix(finalCol, glossyLip, 0.75 * lipMask);

  gl_FragColor = vec4(clamp(finalCol, 0.0, 1.0), 1.0);
}
`;

function compileShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

export class DropperFilterRenderer {
  private canvas: HTMLCanvasElement;
  private gl: WebGLRenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private texture: WebGLTexture | null = null;

  private uResolutionLoc: WebGLUniformLocation | null = null;
  private uCropScaleLoc: WebGLUniformLocation | null = null;
  private uFaceCountLoc: WebGLUniformLocation | null = null;
  private uFacesLoc: WebGLUniformLocation | null = null;
  private uEyesLoc: WebGLUniformLocation | null = null;
  private uAnglesLoc: WebGLUniformLocation | null = null;

  private landmarker: FaceLandmarker | null = null;
  private detector: FaceDetector | null = null;
  private modelLoading = false;
  private lastVideoTime = -1;
  private lastDetectTimestamp = 0;
  private smoothedFaces: TrackedFace[] = [];
  private lastFaceSeenAt = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.initWebGL();
    this.initVisionModels();
  }

  private initWebGL() {
    const gl =
      (this.canvas.getContext("webgl", {
        preserveDrawingBuffer: true,
        alpha: false,
        antialias: true,
      }) as WebGLRenderingContext | null) ||
      (this.canvas.getContext("experimental-webgl", {
        preserveDrawingBuffer: true,
      }) as WebGLRenderingContext | null);

    if (!gl) return;
    this.gl = gl;

    const vs = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_SOURCE);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE);
    if (!vs || !fs) return;

    const program = gl.createProgram();
    if (!program) return;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      return;
    }

    this.program = program;
    gl.useProgram(program);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );

    const aPos = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    this.texture = texture;

    this.uResolutionLoc = gl.getUniformLocation(program, "u_resolution");
    this.uCropScaleLoc = gl.getUniformLocation(program, "u_cropScale");
    this.uFaceCountLoc = gl.getUniformLocation(program, "u_faceCount");
    this.uFacesLoc = gl.getUniformLocation(program, "u_faces");
    this.uEyesLoc = gl.getUniformLocation(program, "u_eyes");
    this.uAnglesLoc = gl.getUniformLocation(program, "u_angles");
  }

  private async initVisionModels() {
    if (this.modelLoading) return;
    this.modelLoading = true;
    try {
      const vision = await FilesetResolver.forVisionTasks("/wasm");

      try {
        this.landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/models/face_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numFaces: MAX_FACES,
          minFaceDetectionConfidence: 0.45,
          minFacePresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        });
        return;
      } catch {
        this.landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/models/face_landmarker.task",
          },
          runningMode: "VIDEO",
          numFaces: MAX_FACES,
          minFaceDetectionConfidence: 0.45,
          minFacePresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        });
        return;
      }
    } catch {
      try {
        const vision = await FilesetResolver.forVisionTasks("/wasm");
        this.detector = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/models/blaze_face_short_range.tflite",
          },
          runningMode: "VIDEO",
          minDetectionConfidence: 0.45,
        });
      } catch {
        // Fallback center face will still work
      }
    } finally {
      this.modelLoading = false;
    }
  }

  private smoothUpdateFaces(newFaces: TrackedFace[], now: number) {
    if (newFaces.length === 0) return;
    this.lastFaceSeenAt = now;
    newFaces.sort((a, b) => a.cx - b.cx);

    if (this.smoothedFaces.length !== newFaces.length) {
      this.smoothedFaces = newFaces;
      return;
    }

    const alpha = 0.45;
    this.smoothedFaces = this.smoothedFaces.map((prev, idx) => {
      const next = newFaces[idx];
      let dAngle = next.angle - prev.angle;
      while (dAngle > Math.PI) dAngle -= Math.PI * 2;
      while (dAngle < -Math.PI) dAngle += Math.PI * 2;

      return {
        cx: prev.cx + (next.cx - prev.cx) * alpha,
        cy: prev.cy + (next.cy - prev.cy) * alpha,
        rx: prev.rx + (next.rx - prev.rx) * alpha,
        ry: prev.ry + (next.ry - prev.ry) * alpha,
        angle: prev.angle + dAngle * alpha,
        leftEyeLx: prev.leftEyeLx + (next.leftEyeLx - prev.leftEyeLx) * alpha,
        leftEyeLy: prev.leftEyeLy + (next.leftEyeLy - prev.leftEyeLy) * alpha,
        rightEyeLx:
          prev.rightEyeLx + (next.rightEyeLx - prev.rightEyeLx) * alpha,
        rightEyeLy:
          prev.rightEyeLy + (next.rightEyeLy - prev.rightEyeLy) * alpha,
      };
    });
  }

  private updateDetections(
    video: HTMLVideoElement,
    cw: number,
    ch: number,
    scaleX: number,
    scaleY: number
  ) {
    const now = performance.now();
    const vw = video.videoWidth || 1280;

    const mapVidUVToCanvas = (vidU: number, vidV: number) => {
      const u = 0.5 - (vidU - 0.5) / scaleX;
      const v = 0.5 + (vidV - 0.5) / scaleY;
      return { x: u * cw, y: v * ch };
    };

    if (
      (this.landmarker || this.detector) &&
      video.currentTime !== this.lastVideoTime &&
      now - this.lastDetectTimestamp >= 22
    ) {
      this.lastVideoTime = video.currentTime;
      this.lastDetectTimestamp = now;

      try {
        if (this.landmarker) {
          const res = this.landmarker.detectForVideo(video, now);
          if (res.faceLandmarks && res.faceLandmarks.length > 0) {
            const newFaces: TrackedFace[] = [];

            for (
              let i = 0;
              i < Math.min(res.faceLandmarks.length, MAX_FACES);
              i++
            ) {
              const lms = res.faceLandmarks[i];
              if (!lms || lms.length < 455) continue;

              const eyeCenter = mapVidUVToCanvas(lms[168].x, lms[168].y);
              const chinTip = mapVidUVToCanvas(lms[152].x, lms[152].y);
              const cheekA = mapVidUVToCanvas(lms[234].x, lms[234].y);
              const cheekB = mapVidUVToCanvas(lms[454].x, lms[454].y);

              const eyeA = mapVidUVToCanvas(
                (lms[33].x + lms[133].x) * 0.5,
                (lms[33].y + lms[133].y) * 0.5
              );
              const eyeB = mapVidUVToCanvas(
                (lms[362].x + lms[263].x) * 0.5,
                (lms[362].y + lms[263].y) * 0.5
              );

              const leftEye = eyeA.x <= eyeB.x ? eyeA : eyeB;
              const rightEye = eyeA.x <= eyeB.x ? eyeB : eyeA;

              const toChinX = chinTip.x - eyeCenter.x;
              const toChinY = chinTip.y - eyeCenter.y;
              const ry = Math.max(28, Math.hypot(toChinX, toChinY));

              const downX = toChinX / ry;
              const downY = toChinY / ry;
              let rightX = downY;
              let rightY = -downX;
              if (
                rightX * (rightEye.x - leftEye.x) +
                  rightY * (rightEye.y - leftEye.y) <
                0
              ) {
                rightX = -rightX;
                rightY = -rightY;
              }

              const angle = Math.atan2(rightY, rightX);

              const cheekWidth = Math.hypot(
                cheekB.x - cheekA.x,
                cheekB.y - cheekA.y
              );
              const rx = Math.max(24, cheekWidth * 0.5);

              const c = Math.cos(angle);
              const s = Math.sin(angle);
              const relLx = (px: number, py: number) =>
                ((px - eyeCenter.x) * c + (py - eyeCenter.y) * s) / rx;
              const relLy = (px: number, py: number) =>
                (-(px - eyeCenter.x) * s + (py - eyeCenter.y) * c) / ry;

              newFaces.push({
                cx: eyeCenter.x,
                cy: eyeCenter.y,
                rx,
                ry,
                angle,
                leftEyeLx: relLx(leftEye.x, leftEye.y),
                leftEyeLy: relLy(leftEye.x, leftEye.y),
                rightEyeLx: relLx(rightEye.x, rightEye.y),
                rightEyeLy: relLy(rightEye.x, rightEye.y),
              });
            }

            this.smoothUpdateFaces(newFaces, now);
          }
        } else if (this.detector) {
          const result = this.detector.detectForVideo(video, now);
          if (result.detections && result.detections.length > 0) {
            const newFaces: TrackedFace[] = [];

            for (
              let i = 0;
              i < Math.min(result.detections.length, MAX_FACES);
              i++
            ) {
              const det = result.detections[i];
              const kps = det.keypoints;
              const bbox = det.boundingBox;

              if (kps && kps.length >= 4) {
                const p0 = mapVidUVToCanvas(kps[0].x, kps[0].y);
                const p1 = mapVidUVToCanvas(kps[1].x, kps[1].y);
                const mouth = mapVidUVToCanvas(kps[3].x, kps[3].y);

                const leftEye = p0.x <= p1.x ? p0 : p1;
                const rightEye = p0.x <= p1.x ? p1 : p0;

                const cx = (leftEye.x + rightEye.x) * 0.5;
                const cy = (leftEye.y + rightEye.y) * 0.5;

                const dx = rightEye.x - leftEye.x;
                const dy = rightEye.y - leftEye.y;
                let angle = Math.atan2(dy, dx);

                const downX = -Math.sin(angle);
                const downY = Math.cos(angle);
                if (
                  downX * (mouth.x - cx) + downY * (mouth.y - cy) <
                  0
                ) {
                  angle += Math.PI;
                }

                const eyeDist = Math.hypot(dx, dy);
                const eyeToMouth = Math.hypot(mouth.x - cx, mouth.y - cy);
                const bboxCanvasW = bbox
                  ? (bbox.width / vw) * (cw / scaleX)
                  : eyeDist * 2.3;

                const rx = Math.max(24, 0.5 * bboxCanvasW);
                const ry = Math.max(28, eyeToMouth / 0.68);

                const c = Math.cos(angle);
                const s = Math.sin(angle);
                const relLx = (px: number, py: number) =>
                  ((px - cx) * c + (py - cy) * s) / rx;
                const relLy = (px: number, py: number) =>
                  (-(px - cx) * s + (py - cy) * c) / ry;

                newFaces.push({
                  cx,
                  cy,
                  rx,
                  ry,
                  angle,
                  leftEyeLx: relLx(leftEye.x, leftEye.y),
                  leftEyeLy: relLy(leftEye.x, leftEye.y),
                  rightEyeLx: relLx(rightEye.x, rightEye.y),
                  rightEyeLy: relLy(rightEye.x, rightEye.y),
                });
              }
            }

            this.smoothUpdateFaces(newFaces, now);
          }
        }
      } catch {
        // Ignore transient frame detection errors
      }
    }

    if (this.smoothedFaces.length === 0 || now - this.lastFaceSeenAt > 1800) {
      const defaultRx = Math.min(cw, ch) * 0.23;
      const targetFallback: TrackedFace = {
        cx: cw * 0.5,
        cy: ch * 0.44,
        rx: defaultRx,
        ry: defaultRx * 1.25,
        angle: 0,
        leftEyeLx: -0.44,
        leftEyeLy: 0,
        rightEyeLx: 0.44,
        rightEyeLy: 0,
      };

      if (this.smoothedFaces.length === 0) {
        this.smoothedFaces = [targetFallback];
      } else {
        const prev = this.smoothedFaces[0];
        const a = 0.12;
        this.smoothedFaces = [
          {
            cx: prev.cx + (targetFallback.cx - prev.cx) * a,
            cy: prev.cy + (targetFallback.cy - prev.cy) * a,
            rx: prev.rx + (targetFallback.rx - prev.rx) * a,
            ry: prev.ry + (targetFallback.ry - prev.ry) * a,
            angle: prev.angle * (1 - a),
            leftEyeLx: -0.44,
            leftEyeLy: 0,
            rightEyeLx: 0.44,
            rightEyeLy: 0,
          },
        ];
      }
    }
  }

  public render(video: HTMLVideoElement) {
    const gl = this.gl;
    if (!gl || !this.program || !this.texture || video.readyState < 2) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const displayW = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const displayH = Math.max(1, Math.round(this.canvas.clientHeight * dpr));

    if (this.canvas.width !== displayW || this.canvas.height !== displayH) {
      this.canvas.width = displayW;
      this.canvas.height = displayH;
    }

    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 960;
    const canvasAspect = displayW / displayH;
    const videoAspect = vw / vh;

    let scaleX = 1.0;
    let scaleY = 1.0;
    if (videoAspect > canvasAspect) {
      scaleX = canvasAspect / videoAspect;
    } else {
      scaleY = videoAspect / canvasAspect;
    }

    this.updateDetections(video, displayW, displayH, scaleX, scaleY);

    gl.viewport(0, 0, displayW, displayH);
    gl.useProgram(this.program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);

    gl.uniform2f(this.uResolutionLoc, displayW, displayH);
    gl.uniform2f(this.uCropScaleLoc, scaleX, scaleY);

    const faceCount = Math.min(this.smoothedFaces.length, MAX_FACES);
    gl.uniform1i(this.uFaceCountLoc, faceCount);

    const facesArray = new Float32Array(MAX_FACES * 4);
    const eyesArray = new Float32Array(MAX_FACES * 4);
    const anglesArray = new Float32Array(MAX_FACES);

    for (let i = 0; i < faceCount; i++) {
      const f = this.smoothedFaces[i];
      facesArray[i * 4 + 0] = f.cx;
      facesArray[i * 4 + 1] = f.cy;
      facesArray[i * 4 + 2] = f.rx;
      facesArray[i * 4 + 3] = f.ry;

      eyesArray[i * 4 + 0] = f.leftEyeLx;
      eyesArray[i * 4 + 1] = f.leftEyeLy;
      eyesArray[i * 4 + 2] = f.rightEyeLx;
      eyesArray[i * 4 + 3] = f.rightEyeLy;

      anglesArray[i] = f.angle;
    }

    gl.uniform4fv(this.uFacesLoc, facesArray);
    gl.uniform4fv(this.uEyesLoc, eyesArray);
    gl.uniform1fv(this.uAnglesLoc, anglesArray);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  public dispose() {
    if (this.landmarker) {
      this.landmarker.close();
      this.landmarker = null;
    }
    if (this.detector) {
      this.detector.close();
      this.detector = null;
    }
  }
}
