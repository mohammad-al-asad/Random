import { Stack } from "expo-router";
import "react-native-reanimated";

import { useColorScheme } from "@/hooks/use-color-scheme";

export const unstable_settings = {
  anchor: "(tabs)",
};

export default function RootLayout() {

  return (
    <Stack>
      <Stack.Screen name="(home)" options={{ headerShown: false }} />
    </Stack>
  );
}
