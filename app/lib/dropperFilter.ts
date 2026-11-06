import {
  FaceDetector,
  FaceLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";

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

      if (abs(lx) <= 2.8 && ly >= -2.3 && ly <= 2.7) {
        float srcLx = lx;
        float srcLy = ly;

        // 1. Sleek Upper Head / Temple Narrowing so the cranium never looks big
        float upperSleekEnv = exp(-pow(lx / 1.35, 2.0) - pow((ly - (-0.30)) / 0.75, 2.0)) * smoothstep(0.25, -0.25, ly);
        srcLx += lx * 0.16 * upperSleekEnv;

        // 2. EXTREME Vertical Face Elongation & Long Razor-Pointed V-Chin
        float lengthenEnv = exp(-pow(lx / 1.25, 2.0) - pow((ly - 0.92) / 0.78, 2.0)) * smoothstep(-0.08, 0.22, ly);
        float centerTipWeight = exp(-pow(lx / 0.46, 2.0));
        srcLy -= (0.22 + 0.24 * centerTipWeight) * lengthenEnv;

        // 3. EXTREME V-Cone Horizontal Face Slimming (Cranked to MAX)
        // Uses a saturated core (lx / sqrt(1 + (lx/w0)^2)) * wide outer decay so we can squeeze
        // the jaw, cheeks, and chin down to ~24% width without folding the background!
        float effLy = srcLy;
        float taperRamp = smoothstep(-0.08, 0.78, effLy) * (1.0 - smoothstep(1.18, 2.10, effLy));
        float jawChinFocus = exp(-pow((effLy - 0.85) / 0.48, 2.0));
        float neckSlim = exp(-pow((effLy - 1.30) / 0.45, 2.0));

        float squeezeStrength = 1.85 * taperRamp + 1.35 * jawChinFocus + 0.75 * neckSlim;
        float saturatedX = lx / sqrt(1.0 + pow(lx / 0.44, 2.0));
        float outerFade = exp(-pow(lx / 2.15, 2.0));
        srcLx += saturatedX * squeezeStrength * outerFade;

        // 4. Ultra-Slender Nose Bridge & Tiny Nostrils
        float noseEnv = exp(-pow(lx / 0.38, 2.0) - pow((effLy - 0.36) / 0.26, 2.0)) * smoothstep(0.02, 0.16, effLy);
        srcLx += lx * 0.85 * noseEnv;

        // 5. Petite Doll Mouth (shrinks mouth width & height slightly inside the narrow V-jaw)
        vec2 dMouth = vec2(lx, effLy - 0.68);
        float mouthEnv = exp(-pow(dMouth.x / 0.38, 2.0) - pow(dMouth.y / 0.18, 2.0));
        srcLx += dMouth.x * 0.35 * mouthEnv;
        srcLy += dMouth.y * 0.25 * mouthEnv;

        // 6. Big Anime/Doll Eyes (Cranked enlargement)
        vec2 dEyeL = vec2(lx - eyes.x, (ly - eyes.y) * (ry / rx));
        float eyeEnvL = exp(-dot(dEyeL, dEyeL) / (0.36 * 0.36));
        srcLx -= (lx - eyes.x) * 0.36 * eyeEnvL;
        srcLy -= (ly - eyes.y) * 0.40 * eyeEnvL;

        vec2 dEyeR = vec2(lx - eyes.z, (ly - eyes.w) * (ry / rx));
        float eyeEnvR = exp(-dot(dEyeR, dEyeR) / (0.36 * 0.36));
        srcLx -= (lx - eyes.z) * 0.36 * eyeEnvR;
        srcLy -= (ly - eyes.w) * 0.40 * eyeEnvR;

        res.pos = center + (srcLx * rx) * rightVec + (srcLy * ry) * downVec;

        // Anatomical masks in source face coordinates
        float foreheadOval = exp(-pow(srcLx / 1.12, 2.0) - pow((effLy - (-0.32)) / 0.55, 2.0));
        float faceOval = exp(-pow(srcLx / 1.15, 2.0) - pow((effLy - 0.32) / 0.88, 2.0));
        float neckArea = exp(-pow(srcLx / 1.10, 2.0) - pow((effLy - 1.35) / 0.80, 2.0));
        res.faceSkinRegion = max(
          res.faceSkinRegion,
          clamp(max(max(foreheadOval, faceOval), neckArea * 0.9), 0.0, 1.0)
        );

        float underEye = exp(-pow(srcLx / 0.85, 2.0) - pow((effLy - 0.18) / 0.18, 2.0));
        float beardZone = exp(-pow(srcLx / 0.85, 2.0) - pow((effLy - 0.78) / 0.36, 2.0));
        res.underEyeOrJawShadow = max(
          res.underEyeOrJawShadow,
          clamp(underEye + beardZone, 0.0, 1.0)
        );

        float lips = exp(-pow(srcLx / 0.28, 2.0) - pow((effLy - 0.68) / 0.11, 2.0));
        res.lipRegion = max(res.lipRegion, lips);

        float blushL = exp(-pow((srcLx - (-0.52)) / 0.32, 2.0) - pow((effLy - 0.30) / 0.20, 2.0));
        float blushR = exp(-pow((srcLx - 0.52) / 0.32, 2.0) - pow((effLy - 0.30) / 0.20, 2.0));
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

  // Wide skin-tone cluster so warm, cool, or dim indoor skin is always captured
  float dCb = (cb - 0.44) / 0.125;
  float dCr = (cr - 0.56) / 0.135;
  float chromaSkin = exp(-0.5 * (dCb * dCb + dCr * dCr));

  float lumaGate = smoothstep(0.08, 0.22, luma);
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

  // --- STEP 1: MAX-LEVEL Porcelain Doll Skin Smoothing ---
  // Scale blur radius proportionally to screen resolution so it melts all pores & stubble on Retina displays
  float resScale = max(1.0, min(u_resolution.x, u_resolution.y) / 540.0);
  vec2 texel = (u_cropScale * resScale) / max(u_resolution, vec2(1.0));

  vec3 accum = centerCol * 1.2;
  float weightSum = 1.2;
  vec3 wideMean = centerCol;

  // 18-tap 3-ring wide bilateral surface filter
  const float sigmaSq = 0.095;
  for (int i = 0; i < 18; i++) {
    float ang = float(i) * 1.04719755; // 60 deg steps (6 per ring)
    float ring = (i < 6) ? 3.5 : ((i < 12) ? 8.5 : 16.0);
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

  // Skin mask: strong on the entire face/neck (excluding only pitch-black hair/glasses/pupils) + global skin tones
  float colorSkin = computeSkinLikelihood(smoothCol);
  float notBlackHairOrFrame = smoothstep(0.10, 0.25, smoothLuma);
  float skinMask = clamp(
    max(colorSkin * 0.92, wr.faceSkinRegion * notBlackHairOrFrame),
    0.0,
    1.0
  );

  // Completely melt pores, blemishes, beard stubble, eye bags, and nasolabial folds into glossy porcelain
  vec3 poreErased = max(centerCol, smoothCol - vec3(0.002));
  vec3 porcelainSmooth = mix(poreErased, smoothCol, 0.97);
  // Blend in soft-focus wide glow so skin looks airbrushed & boneless
  porcelainSmooth = mix(porcelainSmooth, wideMean, 0.35 * wr.faceSkinRegion * notBlackHairOrFrame);

  // Erase dark eye circles and greenish/greyish beard shadow on upper lip & chin
  float darkDip = max(0.0, smoothLuma - centerLuma);
  porcelainSmooth += vec3(darkDip * 0.85) + vec3(0.04, 0.035, 0.045) * wr.underEyeOrJawShadow;

  vec3 col = mix(centerCol, porcelainSmooth, skinMask);

  // --- STEP 2: MAX-LEVEL Cold-White Porcelain Whitening ---
  // Stage A: Strong logarithmic beauty whitening curve
  vec3 logWhite1 = log(1.0 + col * 6.5) / log(7.5);
  vec3 logWhite2 = log(1.0 + logWhite1 * 3.5) / log(4.5);
  vec3 whitened = mix(col, logWhite2, 0.88 * skinMask + 0.22);

  // Stage B: Aggressive de-yellowing + Cold-Porcelain Milky-Pink tone shift
  float wLuma = dot(whitened, vec3(0.299, 0.587, 0.114));
  // Strip warm/yellow cast from skin
  vec3 deYellowed = mix(whitened, vec3(wLuma), 0.52 * skinMask);
  // Push into bright cold-porcelain white with a snowy pink-blue undertone
  vec3 coldWhite = deYellowed * mix(vec3(1.0), vec3(1.06, 1.01, 1.14), skinMask)
                 + vec3(0.075, 0.065, 0.105) * skinMask;

  // Stage C: Over-filtered Beauty-Cam Soft-Focus Glow & Rosy Makeup
  vec3 beautyGlow = pow(clamp(coldWhite, 0.0, 1.0), vec3(0.78));
  vec3 finalCol = mix(coldWhite, beautyGlow, 0.65 * skinMask);

  // Subtle pink blush on upper cheeks
  finalCol += vec3(0.055, -0.015, 0.025) * wr.cheekBlush * skinMask;

  // Glossy rosy-red lips so the mouth looks vibrant against the snow-white face
  float lipMask = wr.lipRegion * smoothstep(0.12, 0.30, centerLuma);
  vec3 glossyLip = vec3(
    min(1.0, finalCol.r * 1.16 + 0.08),
    finalCol.g * 0.78,
    finalCol.b * 0.86 + 0.03
  );
  finalCol = mix(finalCol, glossyLip, 0.68 * lipMask);

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
