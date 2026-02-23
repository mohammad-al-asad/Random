import { useEffect, useMemo, useState } from "react";
import { View, Text, StyleSheet, Dimensions, Platform } from "react-native";
import Svg, { Rect, Text as SvgText, G } from "react-native-svg";
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from "react-native-vision-camera";
import { useRunOnJS, useSharedValue } from "react-native-worklets-core";
import { useResizePlugin } from "vision-camera-resize-plugin";
import { loadTensorflowModel, TensorflowModel } from "react-native-fast-tflite";

const LABELS = [
  "boots",
  "glasses",
  "gloves",
  "helmet",
  "no boots",
  "no glasses",
  "no gloves",
  "no helmet",
  "no vest",
  "person",
  "vest",
];

const INPUT_SIZE = 640;
const NUM_CLASSES = 11;
const NUM_CANDIDATES = 8400;
const INFERENCE_INTERVAL_MS = 300;

const CONF_THR = 0.15;
const IOU_THR = 0.45;

type Det = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  cls: number;
  label: string;
};

type FrameOrientation =
  | "portrait"
  | "portrait-upside-down"
  | "landscape-left"
  | "landscape-right"
  | string;

function sigmoid(x: number) {
  "worklet";
  return 1 / (1 + Math.exp(-x));
}

function normalizeProbability(raw: number) {
  "worklet";
  if (!Number.isFinite(raw)) return 0;
  if (raw < 0 || raw > 1) return sigmoid(raw);
  return raw;
}

function iou(a: Det, b: Det) {
  "worklet";
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-9);
}

function nms(boxes: Det[], iouThr: number) {
  "worklet";
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const keep: Det[] = [];

  for (const b of sorted) {
    let ok = true;
    for (const k of keep) {
      if (iou(b, k) > iouThr) {
        ok = false;
        break;
      }
    }
    if (ok) keep.push(b);
  }

  return keep;
}

function toFloatArray(data: ArrayLike<number | bigint>): Float32Array {
  "worklet";
  if (data instanceof Float32Array) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i += 1) {
    const value = data[i];
    out[i] = typeof value === "bigint" ? Number(value) : value;
  }
  return out;
}

function mapPointToUpright(
  x: number,
  y: number,
  frameW: number,
  frameH: number,
  orientation: FrameOrientation,
): { x: number; y: number } {
  if (orientation === "landscape-left") {
    return { x: frameH - y, y: x };
  }
  if (orientation === "landscape-right") {
    return { x: y, y: frameW - x };
  }
  if (orientation === "portrait-upside-down") {
    return { x: frameW - x, y: frameH - y };
  }
  return { x, y };
}

function getUprightFrameSize(
  frameW: number,
  frameH: number,
  orientation: FrameOrientation,
): { width: number; height: number } {
  if (orientation === "landscape-left" || orientation === "landscape-right") {
    return { width: frameH, height: frameW };
  }
  return { width: frameW, height: frameH };
}

// Output shape: [1, 15, 8400] channels-first
function postprocess(output: Float32Array): Det[] {
  "worklet";
  const N = NUM_CANDIDATES;
  const at = (c: number, i: number) => output[c * N + i];

  const dets: Det[] = [];

  for (let i = 0; i < N; i += 1) {
    let cx = at(0, i);
    let cy = at(1, i);
    let w = at(2, i);
    let h = at(3, i);

    if (
      !Number.isFinite(cx) ||
      !Number.isFinite(cy) ||
      !Number.isFinite(w) ||
      !Number.isFinite(h)
    ) {
      continue;
    }

    // Auto-handle both normalized [0..1] and pixel-scale [0..640] box outputs.
    if (Math.max(Math.abs(cx), Math.abs(cy), Math.abs(w), Math.abs(h)) <= 2) {
      cx *= INPUT_SIZE;
      cy *= INPUT_SIZE;
      w *= INPUT_SIZE;
      h *= INPUT_SIZE;
    }

    let best = 0;
    let cls = -1;
    for (let k = 0; k < NUM_CLASSES; k += 1) {
      const s = normalizeProbability(at(4 + k, i));
      if (s > best) {
        best = s;
        cls = k;
      }
    }

    if (best < CONF_THR) continue;

    dets.push({
      x1: cx - w / 2,
      y1: cy - h / 2,
      x2: cx + w / 2,
      y2: cy + h / 2,
      score: best,
      cls,
      label: LABELS[cls] ?? `cls_${cls}`,
    });
  }

  const final: Det[] = [];
  for (let c = 0; c < NUM_CLASSES; c += 1) {
    final.push(...nms(dets.filter((d) => d.cls === c), IOU_THR));
  }

  return final;
}

function mapToScreen(
  dets: Det[],
  screenW: number,
  screenH: number,
  frameW: number,
  frameH: number,
  orientation: FrameOrientation,
): Det[] {
  if (frameW <= 0 || frameH <= 0) return [];

  // resize-plugin center-crops frame to square before scaling to INPUT_SIZE.
  const cropSize = Math.min(frameW, frameH);
  const cropX = (frameW - cropSize) / 2;
  const cropY = (frameH - cropSize) / 2;

  const uprightSize = getUprightFrameSize(frameW, frameH, orientation);
  const previewScale = Math.max(
    screenW / uprightSize.width,
    screenH / uprightSize.height,
  );
  const previewOffsetX = (screenW - uprightSize.width * previewScale) / 2;
  const previewOffsetY = (screenH - uprightSize.height * previewScale) / 2;

  return dets.map((d) => {
    // Model-space (0..INPUT_SIZE) -> source frame coordinates.
    const fx1 = cropX + (d.x1 / INPUT_SIZE) * cropSize;
    const fy1 = cropY + (d.y1 / INPUT_SIZE) * cropSize;
    const fx2 = cropX + (d.x2 / INPUT_SIZE) * cropSize;
    const fy2 = cropY + (d.y2 / INPUT_SIZE) * cropSize;

    // Rotate into upright preview coordinates.
    const p1 = mapPointToUpright(fx1, fy1, frameW, frameH, orientation);
    const p2 = mapPointToUpright(fx2, fy1, frameW, frameH, orientation);
    const p3 = mapPointToUpright(fx2, fy2, frameW, frameH, orientation);
    const p4 = mapPointToUpright(fx1, fy2, frameW, frameH, orientation);

    const ux1 = Math.min(p1.x, p2.x, p3.x, p4.x);
    const uy1 = Math.min(p1.y, p2.y, p3.y, p4.y);
    const ux2 = Math.max(p1.x, p2.x, p3.x, p4.x);
    const uy2 = Math.max(p1.y, p2.y, p3.y, p4.y);

    const rawX1 = ux1 * previewScale + previewOffsetX;
    const rawY1 = uy1 * previewScale + previewOffsetY;
    const rawX2 = ux2 * previewScale + previewOffsetX;
    const rawY2 = uy2 * previewScale + previewOffsetY;

    const clampedX1 = Math.max(0, Math.min(screenW, rawX1));
    const clampedY1 = Math.max(0, Math.min(screenH, rawY1));
    const clampedX2 = Math.max(0, Math.min(screenW, rawX2));
    const clampedY2 = Math.max(0, Math.min(screenH, rawY2));

    return {
      ...d,
      x1: Math.min(clampedX1, clampedX2),
      y1: Math.min(clampedY1, clampedY2),
      x2: Math.max(clampedX1, clampedX2),
      y2: Math.max(clampedY1, clampedY2),
    };
  });
}

export default function LiveDetectScreen() {
  const device = useCameraDevice("back");
  const { hasPermission, requestPermission } = useCameraPermission();
  const { resize } = useResizePlugin();

  const [model, setModel] = useState<TensorflowModel | null>(null);
  const [dets, setDets] = useState<Det[]>([]);
  const lastRunMs = useSharedValue(0);

  const { width: screenW, height: screenH } = Dimensions.get("window");
  const sendDetectionsToJs = useRunOnJS(
    (
      next: Det[],
      frameW: number,
      frameH: number,
      orientation: FrameOrientation,
    ) => {
      setDets(mapToScreen(next, screenW, screenH, frameW, frameH, orientation));
    },
    [screenW, screenH],
  );

  useEffect(() => {
    (async () => {
      const loaded = await loadTensorflowModel(
        require("../../assets/models/best_float16.tflite"),
      );
      setModel(loaded);
    })();
  }, []);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  const ready = useMemo(() => model !== null, [model]);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      "worklet";

      if (!model) return;

      // Keep frame processing bounded.
      const now = performance.now();
      if (now - lastRunMs.value < INFERENCE_INTERVAL_MS) return;
      lastRunMs.value = now;

      const input = resize(frame, {
        scale: { width: INPUT_SIZE, height: INPUT_SIZE },
        pixelFormat: "rgb",
        dataType: "float32",
      });

      const outputs = model.runSync([input]);
      if (!outputs.length) return;

      const out = toFloatArray(outputs[0]);
      const parsed = postprocess(out);
      sendDetectionsToJs(parsed, frame.width, frame.height, frame.orientation);
    },
    [model, resize, lastRunMs, sendDetectionsToJs],
  );

  if (!device) {
    return (
      <View style={styles.center}>
        <Text style={styles.txt}>No camera device found.</Text>
      </View>
    );
  }

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.txt}>Camera permission is required.</Text>
      </View>
    );
  }

  if (!ready) {
    return (
      <View style={styles.center}>
        <Text style={styles.txt}>Loading model...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
      />

      <View style={styles.topPill}>
        <View style={styles.dot} />
        <Text style={styles.topPillText}>LIVE SCAN ACTIVE</Text>
      </View>

      <Svg style={StyleSheet.absoluteFill}>
        {dets.map((d, idx) => {
          const w = d.x2 - d.x1;
          const h = d.y2 - d.y1;
          if (w < 2 || h < 2) return null;
          return (
            <G key={idx}>
              <Rect
                x={d.x1}
                y={d.y1}
                width={w}
                height={h}
                stroke="lime"
                strokeWidth={2}
                fill="transparent"
                rx={6}
                ry={6}
              />
              <Rect
                x={Math.max(0, Math.min(screenW - 180, d.x1))}
                y={Math.max(0, d.y1 - 22)}
                width={180}
                height={20}
                fill="rgba(0,0,0,0.55)"
                rx={6}
                ry={6}
              />
              <SvgText
                x={Math.max(8, Math.min(screenW - 172, d.x1 + 8))}
                y={Math.max(14, d.y1 - 8)}
                fill="white"
                fontSize={12}
                fontWeight="700"
              >
                {d.label} {(d.score * 100).toFixed(0)}%
              </SvgText>
            </G>
          );
        })}
      </Svg>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "black" },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "black",
  },
  txt: { color: "white" },

  topPill: {
    position: "absolute",
    top: Platform.select({ ios: 60, android: 40, default: 40 }),
    alignSelf: "center",
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(0,0,0,0.55)",
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 999,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 8,
    backgroundColor: "lime",
    marginRight: 8,
  },
  topPillText: {
    color: "white",
    fontWeight: "800",
    letterSpacing: 0.6,
    fontSize: 12,
  },
});
