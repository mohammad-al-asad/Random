import { useLocalSearchParams, useRouter } from "expo-router";
import React, { useMemo } from "react";
import { Image, Pressable, StyleSheet, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import Svg, { Path } from "react-native-svg";

type Detection = {
  classId: number;
  className: string;
  score: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
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

export default function ResultScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{
    imageUri?: string;
    detections?: string;
  }>();

  const parsedDetections = useMemo(() => {
    if (!params.detections) return [] as Detection[];
    try {
      return JSON.parse(params.detections) as Detection[];
    } catch {
      return [] as Detection[];
    }
  }, [params.detections]);

  const hasHazard = parsedDetections.some((d) => d.className.startsWith("no "));
  const summary = formatSummary(parsedDetections);
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
    <SafeAreaView
      style={[
        styles.safeArea,
        { backgroundColor: hasHazard ? "#7f1d1d" : "#0f9d58" },
      ]}
    >
      <View style={styles.screen}>
        <View
          style={[
            styles.header,
            { backgroundColor: hasHazard ? "#7f1d1d" : "#0f9d58" },
          ]}
        >
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
          <Text style={styles.headerTitle}>
            {hasHazard ? "Hazard Detected" : "Safeguard"}
          </Text>
          <View style={styles.headerSpacer} />
        </View>

        <View style={styles.previewWrap}>
          {params.imageUri ? (
            <Image source={{ uri: params.imageUri }} style={styles.preview} />
          ) : (
            <View style={styles.previewEmpty}>
              <Text style={styles.previewEmptyText}>No image provided</Text>
            </View>
          )}
          <View
            style={[
              styles.frame,
              {
                borderColor: hasHazard
                  ? "rgba(239, 68, 68, 0.9)"
                  : "rgba(22, 163, 74, 0.9)",
              },
            ]}
          />
          <View
            style={[
              styles.summaryPill,
              { backgroundColor: hasHazard ? "#b91c1c" : "#16a34a" },
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
          <Text style={styles.cardTitle}>
            {hasHazard ? "Missing PPE Detected" : "Congratulations!"}
          </Text>
          <Text style={styles.cardBody}>
            {hasHazard
              ? "Some required PPE items were not detected."
              : "You have proper PPE. Keep going and be safe!"}
          </Text>
        </View>

        <Pressable
          style={[
            styles.primaryButton,
            { backgroundColor: hasHazard ? "#111827" : "#1f2937" },
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
