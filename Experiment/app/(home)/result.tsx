import { useLocalSearchParams, useRouter } from "expo-router";
import { useEffect, useMemo, useState } from "react";
import { Image, Pressable, StyleSheet, Text, View } from "react-native";
import { manipulateAsync, SaveFormat } from "expo-image-manipulator";
import { loadTensorflowModel, TensorflowModel } from "react-native-fast-tflite";
import { SafeAreaView } from "react-native-safe-area-context";
import Svg, { Path } from "react-native-svg";
import { toByteArray } from "base64-js";
import * as jpeg from "jpeg-js";

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
const CONF_THR = 0.5;
const IOU_THR = 0.45;

type Detection = {
  classId: number;
  className: string;
  score: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

type ModelDet = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  cls: number;
  label: string;
};

function toTitleCase(value: string): string {
  return value
    .split(" ")
    .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
    .join(" ");
}

function formatSummary(detections: Detection[]): string {
  if (!detections.length) return "No detections";
  const unique = Array.from(
    new Set(detections.map((d) => toTitleCase(d.className))),
  );
  return unique.join(", ");
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function normalizeProbability(raw: number): number {
  if (!Number.isFinite(raw)) return 0;
  if (raw < 0 || raw > 1) return sigmoid(raw);
  return raw;
}

function iou(a: ModelDet, b: ModelDet): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-9);
}

function nms(boxes: ModelDet[], iouThr: number): ModelDet[] {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const keep: ModelDet[] = [];

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
  if (data instanceof Float32Array) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i += 1) {
    const value = data[i];
    out[i] = typeof value === "bigint" ? Number(value) : value;
  }
  return out;
}

function postprocess(output: Float32Array): ModelDet[] {
  const N = NUM_CANDIDATES;
  const at = (c: number, i: number) => output[c * N + i];

  const dets: ModelDet[] = [];

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

  const final: ModelDet[] = [];
  for (let c = 0; c < NUM_CLASSES; c += 1) {
    final.push(...nms(dets.filter((d) => d.cls === c), IOU_THR));
  }
  return final;
}

async function imageToTensor(imageUri: string): Promise<Float32Array> {
  const resized = await manipulateAsync(
    imageUri,
    [{ resize: { width: INPUT_SIZE, height: INPUT_SIZE } }],
    {
      compress: 1,
      format: SaveFormat.JPEG,
      base64: true,
    },
  );

  if (!resized.base64) {
    throw new Error("Image preprocessing failed.");
  }

  const jpegBytes = toByteArray(resized.base64);
  const decoded = jpeg.decode(jpegBytes, { useTArray: true, formatAsRGBA: true });
  if (decoded.width !== INPUT_SIZE || decoded.height !== INPUT_SIZE) {
    throw new Error("Unexpected preprocessed image size.");
  }

  const out = new Float32Array(INPUT_SIZE * INPUT_SIZE * 3);
  const rgba = decoded.data;
  const totalPixels = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < totalPixels; i += 1) {
    const src = i * 4;
    const dst = i * 3;
    out[dst] = rgba[src] / 255;
    out[dst + 1] = rgba[src + 1] / 255;
    out[dst + 2] = rgba[src + 2] / 255;
  }

  return out;
}

async function detectImage(
  model: TensorflowModel,
  imageUri: string,
): Promise<Detection[]> {
  const input = await imageToTensor(imageUri);
  const outputs = model.runSync([input]);
  if (!outputs.length) return [];

  const out = toFloatArray(outputs[0]);
  const parsed = postprocess(out);
  return parsed.map((d) => ({
    classId: d.cls,
    className: d.label,
    score: d.score,
    x1: d.x1,
    y1: d.y1,
    x2: d.x2,
    y2: d.y2,
  }));
}

export default function ResultScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{
    imageUri?: string;
  }>();
  const imageUri = Array.isArray(params.imageUri)
    ? params.imageUri[0]
    : params.imageUri;

  const [model, setModel] = useState<TensorflowModel | null>(null);
  const [parsedDetections, setParsedDetections] = useState<Detection[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(true);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const loaded = await loadTensorflowModel(
          require("../../assets/models/best_float16.tflite"),
        );
        if (!cancelled) setModel(loaded);
      } catch (error) {
        if (!cancelled) {
          console.warn("model_load_failed", error);
          setAnalysisError("Failed to load detection model.");
          setIsAnalyzing(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!imageUri) {
      setParsedDetections([]);
      setAnalysisError("No image provided.");
      setIsAnalyzing(false);
      return;
    }

    if (!model) {
      setIsAnalyzing(true);
      return;
    }

    let cancelled = false;
    setIsAnalyzing(true);
    setAnalysisError(null);

    (async () => {
      try {
        const next = await detectImage(model, imageUri);
        if (!cancelled) setParsedDetections(next);
      } catch (error) {
        if (!cancelled) {
          console.warn("image_detect_failed", error);
          setParsedDetections([]);
          setAnalysisError("Failed to analyze image.");
        }
      } finally {
        if (!cancelled) setIsAnalyzing(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [imageUri, model]);

  const hasHazard = useMemo(
    () => parsedDetections.some((d) => d.className.startsWith("no ")),
    [parsedDetections],
  );
  const isError = analysisError !== null;

  const statusColor = isAnalyzing
    ? "#334155"
    : isError || hasHazard
      ? "#7f1d1d"
      : "#0f9d58";
  const summary = isAnalyzing
    ? "Analyzing image..."
    : isError
      ? analysisError
      : formatSummary(parsedDetections);
  const headerTitle = isAnalyzing
    ? "Analyzing..."
    : isError
      ? "Analysis Failed"
      : hasHazard
        ? "Hazard Detected"
        : "Safeguard";
  const cardTitle = isAnalyzing
    ? "Analyzing Image"
    : isError
      ? "Analysis Failed"
      : hasHazard
        ? "Missing PPE Detected"
        : "Congratulations!";
  const cardBody = isAnalyzing
    ? "Please wait while the image is being analyzed."
    : isError
      ? "We could not analyze this image. Try again with a clearer photo."
      : hasHazard
        ? "Some required PPE items were not detected."
        : "You have proper PPE. Keep going and be safe!";

  const now = new Date();
  const dateLabel = now.toLocaleDateString(undefined, {
    month: "short",
    day: "2-digit",
    year: "numeric",
  });
  const timeLabel = now.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: statusColor }]}>
      <View style={styles.screen}>
        <View style={[styles.header, { backgroundColor: statusColor }]}>
          <Pressable style={styles.backButton} onPress={() => router.back()}>
            <Svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <Path
                d="M15 18l-6-6 6-6"
                stroke="#000"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </Svg>
          </Pressable>
          <Text style={styles.headerTitle}>{headerTitle}</Text>
          <View style={styles.headerSpacer} />
        </View>

        <View style={styles.previewWrap}>
          {imageUri ? (
            <Image source={{ uri: imageUri }} style={styles.preview} />
          ) : (
            <View style={styles.previewEmpty}>
              <Text style={styles.previewEmptyText}>No image provided</Text>
            </View>
          )}
          <View
            style={[
              styles.frame,
              {
                borderColor: isAnalyzing
                  ? "rgba(148, 163, 184, 0.9)"
                  : isError || hasHazard
                    ? "rgba(239, 68, 68, 0.9)"
                    : "rgba(22, 163, 74, 0.9)",
              },
            ]}
          />
          <View
            style={[
              styles.summaryPill,
              {
                backgroundColor: isAnalyzing
                  ? "#334155"
                  : isError || hasHazard
                    ? "#b91c1c"
                    : "#16a34a",
              },
            ]}
          >
            <Text style={styles.summaryText}>{summary}</Text>
          </View>
        </View>

        <View style={styles.metaRow}>
          <Text style={styles.metaText}>{dateLabel}</Text>
          <View style={styles.metaDot} />
          <Text style={styles.metaText}>{timeLabel}</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>{cardTitle}</Text>
          <Text style={styles.cardBody}>{cardBody}</Text>
        </View>

        <Pressable
          style={[
            styles.primaryButton,
            { backgroundColor: isAnalyzing ? "#334155" : "#1f2937" },
          ]}
          onPress={() => router.back()}
        >
          <Text style={styles.primaryButtonText}>Done</Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
  },
  screen: {
    flex: 1,
    backgroundColor: "#0f172a",
    paddingBottom: 24,
  },
  header: {
    paddingHorizontal: 18,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 12,
  },
  backButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: "#ffffff",
    alignItems: "center",
    justifyContent: "center",
  },
  backIcon: {
    color: "#f8fafc",
    fontSize: 20,
  },
  headerTitle: {
    color: "#f8fafc",
    fontSize: 16,
    fontWeight: "700",
  },
  headerSpacer: {
    width: 36,
  },
  previewWrap: {
    borderRadius: 20,
    overflow: "hidden",
    marginTop: 10,
    backgroundColor: "#111827",
    height: 320,
  },
  preview: {
    width: "100%",
    height: "100%",
  },
  previewEmpty: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  previewEmptyText: {
    color: "#94a3b8",
  },
  frame: {
    position: "absolute",
    left: 18,
    right: 18,
    top: 24,
    bottom: 24,
    borderWidth: 2,
    borderRadius: 18,
  },
  summaryPill: {
    position: "absolute",
    top: 16,
    left: 18,
    right: 18,
    borderRadius: 16,
    paddingVertical: 8,
    paddingHorizontal: 12,
  },
  summaryText: {
    color: "#f8fafc",
    fontSize: 12,
    fontWeight: "700",
    textAlign: "center",
  },
  metaRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    marginTop: 14,
  },
  metaText: {
    color: "#cbd5f5",
    fontSize: 12,
  },
  metaDot: {
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: "#64748b",
    marginHorizontal: 8,
  },
  card: {
    marginTop: 16,
    backgroundColor: "#111827",
    borderRadius: 16,
    padding: 16,
  },
  cardTitle: {
    color: "#f8fafc",
    fontSize: 16,
    fontWeight: "700",
  },
  cardBody: {
    color: "#94a3b8",
    marginTop: 6,
    fontSize: 13,
  },
  primaryButton: {
    marginTop: 18,
    borderRadius: 14,
    paddingVertical: 12,
    alignItems: "center",
    marginHorizontal: 18,
  },
  primaryButtonText: {
    color: "#f8fafc",
    fontWeight: "700",
  },
});
