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

vec2 warpSingleFace(vec2 pos, vec4 face, vec4 eyes, float angle) {
  vec2 center = face.xy;
  float rx = max(face.z, 1.0);
  float ry = max(face.w, 1.0);

  float c = cos(angle);
  float s = sin(angle);
  vec2 rightVec = vec2(c, s);
  vec2 downVec = vec2(-s, c);

  vec2 rel = pos - center;
  // Anatomical local coordinates:
  // lx = +-1.0 at left/right outer cheekbones
  // ly = 0.0 at eye line, +0.42 at nose tip, +0.68 at mouth, +1.00 at chin tip, -0.55 at hairline
  float lx = dot(rel, rightVec) / rx;
  float ly = dot(rel, downVec) / ry;

  if (abs(lx) > 2.6 || ly < -2.2 || ly > 2.5) {
    return pos;
  }

  float srcLx = lx;
  float srcLy = ly;

  // 1. Keep upper head/forehead slightly sleek so the head never looks oversized
  float upperSleekEnv = exp(-pow(lx / 1.25, 2.0) - pow((ly - (-0.35)) / 0.65, 2.0)) * smoothstep(0.15, -0.25, ly);
  srcLx += lx * 0.08 * upperSleekEnv;

  // 2. Vertical Face Elongation & Long Pointed V-Chin
  // Subtracting from srcLy pulls mid/lower-face features downward to make the face noticeably longer,
  // with extra downward pull along the center line (lx = 0) for a sharp pointed chin tip.
  float lengthenEnv = exp(-pow(lx / 1.15, 2.0) - pow((ly - 0.88) / 0.68, 2.0)) * smoothstep(-0.04, 0.24, ly);
  float centerTipWeight = exp(-pow(lx / 0.52, 2.0));
  srcLy -= (0.14 + 0.15 * centerTipWeight) * lengthenEnv;

  // 3. Smooth Continuous V-Cone Horizontal Taper along the elongated lower face
  float effLy = srcLy;
  float taperRamp = smoothstep(0.02, 0.90, effLy) * (1.0 - smoothstep(1.12, 1.90, effLy));
  float jawChinFocus = exp(-pow((effLy - 0.86) / 0.44, 2.0));
  float xEnv = exp(-pow(lx / 1.18, 2.0));

  float horizSqueeze = (1.18 * taperRamp + 0.64 * jawChinFocus) * xEnv;
  srcLx += lx * horizSqueeze;

  // 4. Slender Nose Bridge & Nostrils
  float noseEnv = exp(-pow(lx / 0.36, 2.0) - pow((effLy - 0.36) / 0.24, 2.0)) * smoothstep(0.05, 0.18, effLy);
  srcLx += lx * 0.56 * noseEnv;

  // 5. Enlarged Expressive Eyes
  vec2 dEyeL = vec2(lx - eyes.x, (ly - eyes.y) * (ry / rx));
  float eyeEnvL = exp(-dot(dEyeL, dEyeL) / (0.30 * 0.30));
  srcLx -= (lx - eyes.x) * 0.22 * eyeEnvL;
  srcLy -= (ly - eyes.y) * 0.24 * eyeEnvL;

  vec2 dEyeR = vec2(lx - eyes.z, (ly - eyes.w) * (ry / rx));
  float eyeEnvR = exp(-dot(dEyeR, dEyeR) / (0.30 * 0.30));
  srcLx -= (lx - eyes.z) * 0.22 * eyeEnvR;
  srcLy -= (ly - eyes.w) * 0.24 * eyeEnvR;

  return center + (srcLx * rx) * rightVec + (srcLy * ry) * downVec;
}

void main() {
  vec2 pos = v_uv * u_resolution;

  for (int i = 0; i < ${MAX_FACES}; i++) {
    if (i < u_faceCount) {
      pos = warpSingleFace(pos, u_faces[i], u_eyes[i], u_angles[i]);
    }
  }

  vec2 warpedUV = pos / u_resolution;

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
