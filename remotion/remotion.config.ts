import { Config } from "@remotion/cli/config";

/**
 * Render defaults for the DemoReel.
 *
 * The ``--codec`` / ``--pixel-format`` flags on the CLI win over these, but
 * having them here means a bare ``npx remotion render`` still produces a
 * phone-safe file, and the Studio preview matches what CI ships.
 */
Config.setVideoImageFormat("jpeg");
Config.setJpegQuality(92);
Config.setCodec("h264");
Config.setPixelFormat("yuv420p");
// Let Remotion pick from the machine's cores instead of a fixed number: a
// hard-coded 4 fails on a 2-core box ("Maximum for --concurrency is 2").
// CI passes --concurrency explicitly.

/**
 * Chrome Headless Shell is the small, download-only browser Remotion uses for
 * server rendering. Fetched at install time by ``scripts/render_setup.sh`` so
 * the render itself needs no network access.
 */
// Remotion's internal name for the same binary the CLI flag calls
// "chrome-headless-shell".
Config.setChromeMode("headless-shell");
