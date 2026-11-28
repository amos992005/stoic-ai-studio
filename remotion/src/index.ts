import { registerRoot } from "remotion";
import { RemotionRoot } from "./Root";

/**
 * Entry point named on the command line by CI:
 *
 *   npx remotion render src/index.ts DemoReel ../outputs/demo_reel.mp4
 */
registerRoot(RemotionRoot);
