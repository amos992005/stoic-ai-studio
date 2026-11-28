/**
 * One place for the reel's numbers, palette and type.
 *
 * The timeline is the contract: 15 s at 30 fps is 450 frames, and the four
 * scenes tile it exactly with no gaps and no overlap. CI verifies the
 * composition's own duration against these values, so scene boundaries can be
 * changed in one place without silently shipping a 14.2 s reel.
 */

export const FPS = 30;
export const WIDTH = 1080;
export const HEIGHT = 1920;
export const DURATION_IN_FRAMES = 450;

export const SCENES = {
  hook: { from: 0, duration: 105 }, // 0.0  – 3.5 s
  metric: { from: 105, duration: 120 }, // 3.5  – 7.5 s
  highlight: { from: 225, duration: 120 }, // 7.5  – 11.5 s
  cta: { from: 345, duration: 105 }, // 11.5 – 15.0 s
} as const;

export type SceneName = keyof typeof SCENES;

/** Dark-luxury palette: near-black base, one warm accent, one cool accent. */
export const COLORS = {
  ink: "#05060A",
  inkSoft: "#0B0E16",
  surface: "#101421",
  gold: "#F5C46B",
  goldDeep: "#C8963E",
  mint: "#5EEAD4",
  violet: "#8B7CF6",
  white: "#FFFFFF",
  muted: "rgba(255,255,255,0.62)",
  faint: "rgba(255,255,255,0.10)",
} as const;

export const FONTS = {
  /** Anton: very tight, very heavy — used for the big kinetic words. */
  display: "Anton",
  /** Inter: the reading type and the numerals. */
  text: "Inter",
  fallback:
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
} as const;

/**
 * Words in the hook, one per beat. Each entry is a line; inside a line, a word
 * written in ALL CAPS is rendered as an accent word.
 */
export const HOOK_LINES = [
  ["EVERYONE", "wants", "the", "view."],
  ["NOBODY", "wants", "the", "climb."],
] as const;

/** Scene 2's counting number. */
export const METRIC = {
  value: 1,
  suffix: "%",
  label: "of your day",
  caption: "compounding, quietly, for years",
  bars: [
    { label: "Year 1", value: 34 },
    { label: "Year 3", value: 58 },
    { label: "Year 5", value: 82 },
    { label: "Year 10", value: 100 },
  ],
} as const;

/** Scene 4's call to action. */
export const CTA = {
  kicker: "THE WHOLE SECRET",
  line: "SHOW UP",
  accent: "TOMORROW",
  footnote: "STAY STOIC",
} as const;
