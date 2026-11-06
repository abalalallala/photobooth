import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";

export interface TrackedFace {
  /** Center between the eyes in canvas pixel coordinates */
  cx: number;
  cy: number;
  /** Characteristic half-face radius in canvas pixels */
  radius: number;
  /** Head roll angle in radians (0 = upright) */
  angle: number;
  /** Left eye in local face coordinates (normalized by radius) */
  leftEyeLx: number;
  leftEyeLy: number;
  /** Right eye in local face coordinates (normalized by radius) */
  rightEyeLx: number;
  rightEyeLy: number;
}

const MAX_FACES = 3;

const VERTEX_SHADER_SOURCE = `
attribute vec2 a_position;
varying vec2 v_uv;

void main() {
  // Map clip space (-1..1) to top-left origin UV (0..1)
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
uniform float u_intensity;
uniform int u_faceCount;
uniform vec4 u_faces[${MAX_FACES}]; // (cx, cy, radius, angle)
uniform vec4 u_eyes[${MAX_FACES}];  // (leftEyeLx, leftEyeLy, rightEyeLx, rightEyeLy)

vec2 warpSingleFace(vec2 pos, vec4 face, vec4 eyes, float intensity) {
  vec2 center = face.xy;
  float R = max(face.z, 1.0);
  float angle = face.w;

  float c = cos(angle);
  float s = sin(angle);
  vec2 rightVec = vec2(c, s);
  vec2 downVec = vec2(-s, c);

  vec2 rel = pos - center;
  float lx = dot(rel, rightVec) / R;
  float ly = dot(rel, downVec) / R;

  // Quick bounding check so distant background pixels skip math
  if (abs(lx) > 2.6 || ly < -2.4 || ly > 2.5) {
    return pos;
  }

  float srcLx = lx;
  float srcLy = ly;

  // 1. Upper Head / Hair / Cranium Rubber Bulb Expansion (胶头滴管大球头)
  vec2 dBulb = vec2(lx, ly - (-0.64));
  float bulbDist2 = (dBulb.x * dBulb.x) / (1.48 * 1.48) + (dBulb.y * dBulb.y) / (1.22 * 1.22);
  float bulbEnv = exp(-bulbDist2) * smoothstep(0.28, -0.15, ly);
  srcLx -= dBulb.x * (0.42 * intensity) * bulbEnv;
  srcLy -= dBulb.y * (0.36 * intensity) * bulbEnv;

  // 2. Lower Face / Cheeks / Jaw / Chin Pipette Taper (胶头滴管尖嘴收窄)
  float lowerMask = smoothstep(-0.12, 0.32, ly) * (1.0 - smoothstep(1.15, 2.15, ly));
  float jawFocus = exp(-pow((ly - 0.82) / 0.62, 2.0));
  float xEnv = exp(-pow(lx / 0.92, 2.0));
  float squeeze = (1.18 * lowerMask + 0.52 * jawFocus) * xEnv * intensity;
  srcLx += lx * squeeze;

  // 3. Slender Nose Bridge & Tip Pinch (鼻梁鼻头收细)
  float noseEnv = exp(-pow(lx / 0.34, 2.0) - pow((ly - 0.36) / 0.34, 2.0));
  srcLx += lx * (0.52 * intensity) * noseEnv;

  // 4. Compact Pointed Chin Lift (尖下巴微收)
  float chinEnv = exp(-pow(lx / 0.85, 2.0) - pow((ly - 0.92) / 0.55, 2.0));
  srcLy += (ly - 0.45) * (0.16 * intensity) * chinEnv;

  // 5. Enlarged Expressive Eyes (大眼效果)
  vec2 dEyeL = vec2(lx - eyes.x, ly - eyes.y);
  float eyeEnvL = exp(-dot(dEyeL, dEyeL) / (0.36 * 0.36));
  srcLx -= dEyeL.x * (0.24 * intensity) * eyeEnvL;
  srcLy -= dEyeL.y * (0.28 * intensity) * eyeEnvL;

  vec2 dEyeR = vec2(lx - eyes.z, ly - eyes.w);
  float eyeEnvR = exp(-dot(dEyeR, dEyeR) / (0.36 * 0.36));
  srcLx -= dEyeR.x * (0.24 * intensity) * eyeEnvR;
  srcLy -= dEyeR.y * (0.28 * intensity) * eyeEnvR;

  return center + (srcLx * rightVec + srcLy * downVec) * R;
}

void main() {
  vec2 pos = v_uv * u_resolution;

  if (u_intensity > 0.001) {
    for (int i = 0; i < ${MAX_FACES}; i++) {
      if (i < u_faceCount) {
        pos = warpSingleFace(pos, u_faces[i], u_eyes[i], u_intensity);
      }
    }
  }

  vec2 warpedUV = pos / u_resolution;

  // Convert mirrored canvas UV (0..1) to cover-cropped source video UV (0..1)
  vec2 videoUV = vec2(
    0.5 + (0.5 - warpedUV.x) * u_cropScale.x,
    0.5 + (warpedUV.y - 0.5) * u_cropScale.y
  );

  videoUV = clamp(videoUV, vec2(0.001), vec2(0.999));
  gl_FragColor = texture2D(u_video, videoUV);
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
  private uIntensityLoc: WebGLUniformLocation | null = null;
  private uFaceCountLoc: WebGLUniformLocation | null = null;
  private uFacesLoc: WebGLUniformLocation | null = null;
  private uEyesLoc: WebGLUniformLocation | null = null;

  private detector: FaceDetector | null = null;
  private detectorLoading = false;
  private lastVideoTime = -1;
  private lastDetectTimestamp = 0;
  private smoothedFaces: TrackedFace[] = [];
  private lastFaceSeenAt = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.initWebGL();
    this.initFaceDetector();
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

    // Fullscreen quad (-1..1)
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

    // Video texture
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    this.texture = texture;

    this.uResolutionLoc = gl.getUniformLocation(program, "u_resolution");
    this.uCropScaleLoc = gl.getUniformLocation(program, "u_cropScale");
    this.uIntensityLoc = gl.getUniformLocation(program, "u_intensity");
    this.uFaceCountLoc = gl.getUniformLocation(program, "u_faceCount");
    this.uFacesLoc = gl.getUniformLocation(program, "u_faces");
    this.uEyesLoc = gl.getUniformLocation(program, "u_eyes");
  }

  private async initFaceDetector() {
    if (this.detector || this.detectorLoading) return;
    this.detectorLoading = true;
    try {
      const vision = await FilesetResolver.forVisionTasks("/wasm");
      this.detector = await FaceDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "/models/blaze_face_short_range.tflite",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        minDetectionConfidence: 0.45,
      });
    } catch {
      // Fallback to CPU delegate if GPU delegate is unavailable
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
        // Fallback center face will still work if detector fails
      }
    } finally {
      this.detectorLoading = false;
    }
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

    if (
      this.detector &&
      video.currentTime !== this.lastVideoTime &&
      now - this.lastDetectTimestamp >= 24
    ) {
      this.lastVideoTime = video.currentTime;
      this.lastDetectTimestamp = now;

      try {
        const result = this.detector.detectForVideo(video, now);
        if (result.detections && result.detections.length > 0) {
          const newFaces: TrackedFace[] = [];

          const mapVidUVToCanvas = (vidU: number, vidV: number) => {
            const u = 0.5 - (vidU - 0.5) / scaleX;
            const v = 0.5 + (vidV - 0.5) / scaleY;
            return { x: u * cw, y: v * ch };
          };

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
              const nose = mapVidUVToCanvas(kps[2].x, kps[2].y);
              const mouth = mapVidUVToCanvas(kps[3].x, kps[3].y);

              // Ensure leftEye is on the left side of the canvas (smaller x)
              const leftEye = p0.x <= p1.x ? p0 : p1;
              const rightEye = p0.x <= p1.x ? p1 : p0;

              const cx = (leftEye.x + rightEye.x) * 0.5;
              const cy = (leftEye.y + rightEye.y) * 0.5;

              const dx = rightEye.x - leftEye.x;
              const dy = rightEye.y - leftEye.y;
              let angle = Math.atan2(dy, dx);

              // Verify downVec points from eye midpoint toward mouth/nose
              const downX = -Math.sin(angle);
              const downY = Math.cos(angle);
              const toLowerX = (mouth.x + nose.x) * 0.5 - cx;
              const toLowerY = (mouth.y + nose.y) * 0.5 - cy;
              if (downX * toLowerX + downY * toLowerY < 0) {
                angle += Math.PI;
              }

              const eyeDist = Math.hypot(dx, dy);
              const bboxCanvasW = bbox
                ? (bbox.width / vw) * (cw / scaleX)
                : eyeDist * 2.4;

              const radius = Math.max(
                24,
                0.55 * (eyeDist * 1.25) + 0.45 * (bboxCanvasW * 0.52)
              );

              const c = Math.cos(angle);
              const s = Math.sin(angle);
              const relLx = (px: number, py: number) =>
                ((px - cx) * c + (py - cy) * s) / radius;
              const relLy = (px: number, py: number) =>
                (-(px - cx) * s + (py - cy) * c) / radius;

              newFaces.push({
                cx,
                cy,
                radius,
                angle,
                leftEyeLx: relLx(leftEye.x, leftEye.y),
                leftEyeLy: relLy(leftEye.x, leftEye.y),
                rightEyeLx: relLx(rightEye.x, rightEye.y),
                rightEyeLy: relLy(rightEye.x, rightEye.y),
              });
            }
          }

          if (newFaces.length > 0) {
            this.lastFaceSeenAt = now;
            // Sort left-to-right for consistent multi-face smoothing
            newFaces.sort((a, b) => a.cx - b.cx);

            if (this.smoothedFaces.length !== newFaces.length) {
              this.smoothedFaces = newFaces;
            } else {
              const alpha = 0.42;
              this.smoothedFaces = this.smoothedFaces.map((prev, idx) => {
                const next = newFaces[idx];
                // Shortest angle delta
                let dAngle = next.angle - prev.angle;
                while (dAngle > Math.PI) dAngle -= Math.PI * 2;
                while (dAngle < -Math.PI) dAngle += Math.PI * 2;

                return {
                  cx: prev.cx + (next.cx - prev.cx) * alpha,
                  cy: prev.cy + (next.cy - prev.cy) * alpha,
                  radius: prev.radius + (next.radius - prev.radius) * alpha,
                  angle: prev.angle + dAngle * alpha,
                  leftEyeLx:
                    prev.leftEyeLx + (next.leftEyeLx - prev.leftEyeLx) * alpha,
                  leftEyeLy:
                    prev.leftEyeLy + (next.leftEyeLy - prev.leftEyeLy) * alpha,
                  rightEyeLx:
                    prev.rightEyeLx +
                    (next.rightEyeLx - prev.rightEyeLx) * alpha,
                  rightEyeLy:
                    prev.rightEyeLy +
                    (next.rightEyeLy - prev.rightEyeLy) * alpha,
                };
              });
            }
          }
        }
      } catch {
        // Ignore transient frame detection errors
      }
    }

    // Fallback default center face if no face seen recently so the filter always works
    if (this.smoothedFaces.length === 0 || now - this.lastFaceSeenAt > 1800) {
      const targetFallback: TrackedFace = {
        cx: cw * 0.5,
        cy: ch * 0.44,
        radius: Math.min(cw, ch) * 0.24,
        angle: 0,
        leftEyeLx: -0.4,
        leftEyeLy: 0,
        rightEyeLx: 0.4,
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
            radius: prev.radius + (targetFallback.radius - prev.radius) * a,
            angle: prev.angle * (1 - a),
            leftEyeLx: -0.4,
            leftEyeLy: 0,
            rightEyeLx: 0.4,
            rightEyeLy: 0,
          },
        ];
      }
    }
  }

  public render(video: HTMLVideoElement, intensity = 1.0) {
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
    gl.uniform1f(this.uIntensityLoc, intensity);

    const faceCount = Math.min(this.smoothedFaces.length, MAX_FACES);
    gl.uniform1i(this.uFaceCountLoc, faceCount);

    const facesArray = new Float32Array(MAX_FACES * 4);
    const eyesArray = new Float32Array(MAX_FACES * 4);

    for (let i = 0; i < faceCount; i++) {
      const f = this.smoothedFaces[i];
      facesArray[i * 4 + 0] = f.cx;
      facesArray[i * 4 + 1] = f.cy;
      facesArray[i * 4 + 2] = f.radius;
      facesArray[i * 4 + 3] = f.angle;

      eyesArray[i * 4 + 0] = f.leftEyeLx;
      eyesArray[i * 4 + 1] = f.leftEyeLy;
      eyesArray[i * 4 + 2] = f.rightEyeLx;
      eyesArray[i * 4 + 3] = f.rightEyeLy;
    }

    gl.uniform4fv(this.uFacesLoc, facesArray);
    gl.uniform4fv(this.uEyesLoc, eyesArray);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  public dispose() {
    if (this.detector) {
      this.detector.close();
      this.detector = null;
    }
  }
}
