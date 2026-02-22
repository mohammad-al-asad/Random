import { toByteArray } from "base64-js";
import * as FileSystem from "expo-file-system/legacy";
import * as ImageManipulator from "expo-image-manipulator";
import * as ImagePicker from "expo-image-picker";
import { decode as decodeJpeg } from "jpeg-js";
import React, { useEffect, useMemo, useState } from "react";
import {
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import {
  loadTensorflowModel,
  type TensorflowModel,
} from "react-native-fast-tflite";
import { SafeAreaView } from "react-native-safe-area-context";

const MODEL_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.15;
const NMS_THRESHOLD = 0.45;

// Replace these names if your model class ordering is different.
const CLASS_NAMES = [
  "Hardhat",
  "Mask",
  "NO-Hardhat",
  "NO-Mask",
  "NO-Safety Vest",
  "Person",
  "Safety Cone",
  "Safety Vest",
  "Machinery",
  "Vehicle",
];

type Detection = {
  classId: number;
  className: string;
  score: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function iou(a: Detection, b: Detection): number {
  const interX1 = Math.max(a.x1, b.x1);
  const interY1 = Math.max(a.y1, b.y1);
  const interX2 = Math.min(a.x2, b.x2);
  const interY2 = Math.min(a.y2, b.y2);
  const interW = Math.max(0, interX2 - interX1);
  const interH = Math.max(0, interY2 - interY1);
  const intersection = interW * interH;
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = areaA + areaB - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function nms(detections: Detection[], threshold: number): Detection[] {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const selected: Detection[] = [];

  while (sorted.length > 0) {
    const best = sorted.shift();
    if (!best) break;
    selected.push(best);

    for (let i = sorted.length - 1; i >= 0; i -= 1) {
      if (
        best.classId === sorted[i].classId &&
        iou(best, sorted[i]) > threshold
      ) {
        sorted.splice(i, 1);
      }
    }
  }

  return selected;
}

function labelFromClassId(classId: number): string {
  return CLASS_NAMES[classId] ?? `Class-${classId}`;
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

function decodeYoloOutputWithMode(
  rawOutput: Float32Array,
  hasObjectness: boolean,
): Detection[] {
  const channels = 15;
  const candidates = 8400;
  const classStart = hasObjectness ? 5 : 4;
  const classCount = channels - classStart;
  const detections: Detection[] = [];

  const sample = rawOutput[0] ?? 0;
  const useSigmoid = sample < 0 || sample > 1;

  for (let i = 0; i < candidates; i += 1) {
    let cx = rawOutput[0 * candidates + i];
    let cy = rawOutput[1 * candidates + i];
    let w = rawOutput[2 * candidates + i];
    let h = rawOutput[3 * candidates + i];

    if (Math.max(Math.abs(cx), Math.abs(cy), Math.abs(w), Math.abs(h)) <= 2) {
      cx *= MODEL_SIZE;
      cy *= MODEL_SIZE;
      w *= MODEL_SIZE;
      h *= MODEL_SIZE;
    }

    const rawObjectness = hasObjectness ? rawOutput[4 * candidates + i] : 1;
    const objectness = useSigmoid ? sigmoid(rawObjectness) : rawObjectness;

    let bestClassId = 0;
    let bestClassScore = 0;
    for (let c = 0; c < classCount; c += 1) {
      const rawScore = rawOutput[(classStart + c) * candidates + i];
      const score = useSigmoid ? sigmoid(rawScore) : rawScore;
      if (score > bestClassScore) {
        bestClassScore = score;
        bestClassId = c;
      }
    }

    const confidence = objectness * bestClassScore;
    if (confidence < CONFIDENCE_THRESHOLD) continue;

    const x1 = clamp(cx - w / 2, 0, MODEL_SIZE);
    const y1 = clamp(cy - h / 2, 0, MODEL_SIZE);
    const x2 = clamp(cx + w / 2, 0, MODEL_SIZE);
    const y2 = clamp(cy + h / 2, 0, MODEL_SIZE);
    if (x2 <= x1 || y2 <= y1) continue;

    detections.push({
      classId: bestClassId,
      className: labelFromClassId(bestClassId),
      score: confidence,
      x1,
      y1,
      x2,
      y2,
    });
  }

  return nms(detections, NMS_THRESHOLD);
}

function decodeYoloOutput(rawOutput: Float32Array): Detection[] {
  const withObj = decodeYoloOutputWithMode(rawOutput, true);
  const noObj = decodeYoloOutputWithMode(rawOutput, false);
  return withObj.length >= noObj.length ? withObj : noObj;
}

export default function HomeScreen() {
  const [model, setModel] = useState<TensorflowModel | null>(null);
  const [modelState, setModelState] = useState<"loading" | "ready" | "error">(
    "loading",
  );
  const [error, setError] = useState("");
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const setupModel = async () => {
      setModelState("loading");
      setError("");
      try {
        const loadedModel = await loadTensorflowModel(
          require("../../assets/models/best_float16.tflite"),
        );
        if (cancelled) return;
        console.log("inputs:", loadedModel.inputs);
        console.log("outputs:", loadedModel.outputs);
        setModel(loadedModel);
        setModelState("ready");
      } catch (loadError) {
        if (cancelled) return;
        setModelState("error");
        setError(
          loadError instanceof Error
            ? loadError.message
            : "Failed to load model.",
        );
      }
    };

    setupModel();
    return () => {
      cancelled = true;
    };
  }, []);

  const pickAndDetect = async () => {
    if (!model) {
      setError("Model not ready.");
      return;
    }
    setError("");
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setError("Media library permission is required.");
      return;
    }

    const picked = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      quality: 1,
      allowsEditing: false,
    });

    if (picked.canceled || !picked.assets?.[0]?.uri) return;

    setRunning(true);
    setDetections([]);

    try {
      const processed = await ImageManipulator.manipulateAsync(
        picked.assets[0].uri,
        [{ resize: { width: MODEL_SIZE, height: MODEL_SIZE } }],
        { compress: 1, format: ImageManipulator.SaveFormat.JPEG },
      );

      setImageUri(processed.uri);

      const jpegBase64 = await FileSystem.readAsStringAsync(processed.uri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const jpegBytes = toByteArray(jpegBase64);
      const decoded = decodeJpeg(jpegBytes, {
        useTArray: true,
        formatAsRGBA: true,
      });
      const rgba = decoded.data;

      const input = new Float32Array(MODEL_SIZE * MODEL_SIZE * 3);
      for (let src = 0, dst = 0; src < rgba.length; src += 4, dst += 3) {
        input[dst] = rgba[src] / 255;
        input[dst + 1] = rgba[src + 1] / 255;
        input[dst + 2] = rgba[src + 2] / 255;
      }

      const outputData = await model.run([input]);
      
      if (!outputData?.length) {
        throw new Error("Model produced no outputs.");
      }

      const firstOutput = outputData[0];
      const firstOutputFloat = toFloatArray(firstOutput);
      console.log("outputs count:", outputData.length);
      console.log("output[0] length:", firstOutputFloat.length);
      console.log(
        "output[0] sample (first 30):",
        Array.from(firstOutputFloat.slice(0, 30)),
      );
      const parsed = decodeYoloOutput(firstOutputFloat);
      console.log("decoded detections:", parsed.length);
      if (parsed.length) {
        console.log(
          "detected objects:",
          parsed.map((d) => ({
            name: d.className,
            score: Number((d.score * 100).toFixed(1)),
            x: Number(d.x1.toFixed(1)),
            y: Number(d.y1.toFixed(1)),
          })),
        );
      } else {
        console.log("detected objects: []");
      }
      setDetections(parsed);
    } catch (runError) {
      setError(
        runError instanceof Error
          ? runError.message
          : "Failed to run inference.",
      );
    } finally {
      setRunning(false);
    }
  };

  const detectionRows = useMemo(
    () =>
      detections.map((d, idx) => ({
        id: `${d.classId}-${idx}`,
        text: `${d.className} (${(d.score * 100).toFixed(1)}%) x:${d.x1.toFixed(1)} y:${d.y1.toFixed(1)}`,
      })),
    [detections],
  );

  return (
    <SafeAreaView>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Local TFLite Detection</Text>
        <Text style={styles.subtle}>Model status: {modelState}</Text>
        {error ? <Text style={styles.error}>{error}</Text> : null}

        <Pressable
          style={[styles.button, running && styles.buttonDisabled]}
          onPress={pickAndDetect}
          disabled={running || modelState !== "ready"}
        >
          <Text style={styles.buttonText}>
            {running ? "Running..." : "Pick Image + Run Model"}
          </Text>
        </Pressable>

        <View style={styles.preview}>
          {imageUri ? (
            <>
              <Image
                source={{ uri: imageUri }}
                style={styles.previewImage}
                resizeMode="contain"
              />
              <View pointerEvents="none" style={styles.overlay}>
                {detections.map((d, idx) => (
                  <View
                    key={`${d.classId}-${idx}`}
                    style={[
                      styles.box,
                      {
                        left: `${(d.x1 / MODEL_SIZE) * 100}%`,
                        top: `${(d.y1 / MODEL_SIZE) * 100}%`,
                        width: `${((d.x2 - d.x1) / MODEL_SIZE) * 100}%`,
                        height: `${((d.y2 - d.y1) / MODEL_SIZE) * 100}%`,
                      },
                    ]}
                  >
                    <Text style={styles.boxLabel}>{d.className}</Text>
                  </View>
                ))}
              </View>
            </>
          ) : (
            <Text style={styles.subtle}>No image selected.</Text>
          )}
        </View>

        <View style={styles.results}>
          <Text style={styles.resultsTitle}>
            Detected Objects (x/y from model output)
          </Text>
          {detectionRows.length ? (
            detectionRows.map((row) => (
              <Text key={row.id} style={styles.resultRow}>
                {row.text}
              </Text>
            ))
          ) : (
            <Text style={styles.subtle}>No detections yet.</Text>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 16,
    gap: 12,
  },
  title: {
    fontSize: 22,
    fontWeight: "700",
  },
  subtle: {
    color: "#64748b",
  },
  error: {
    color: "#dc2626",
  },
  button: {
    backgroundColor: "#0f766e",
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: "center",
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: "#fff",
    fontWeight: "700",
  },
  preview: {
    width: "100%",
    aspectRatio: 1,
    borderWidth: 1,
    borderColor: "#cbd5e1",
    borderRadius: 12,
    overflow: "hidden",
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
    backgroundColor: "#f8fafc",
  },
  previewImage: {
    width: "100%",
    height: "100%",
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
  },
  box: {
    position: "absolute",
    borderWidth: 2,
    borderColor: "#ef4444",
  },
  boxLabel: {
    position: "absolute",
    top: -18,
    left: 0,
    backgroundColor: "#ef4444",
    color: "#fff",
    fontSize: 10,
    paddingHorizontal: 4,
    paddingVertical: 1,
  },
  results: {
    borderWidth: 1,
    borderColor: "#e2e8f0",
    borderRadius: 10,
    padding: 10,
    gap: 6,
  },
  resultsTitle: {
    fontWeight: "700",
  },
  resultRow: {
    color: "#0f172a",
    fontSize: 13,
  },
});
