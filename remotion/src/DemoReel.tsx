import React from "react";
import {
  AbsoluteFill,
  Audio,
  interpolate,
  Sequence,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import "./fonts";
import { SceneCta } from "./scenes/SceneCta";
import { SceneHighlight } from "./scenes/SceneHighlight";
import { SceneHook } from "./scenes/SceneHook";
import { SceneMetric } from "./scenes/SceneMetric";
import { COLORS, DURATION_IN_FRAMES, FONTS, FPS, SCENES } from "./theme";

/**
 * The reel itself: four visual scenes and one audio bed.
 *
 * Layout rule — every scene is a ``<Sequence>`` starting at its own frame, so
 * a scene's internal timeline always begins at 0. That keeps the springs in
 * each scene independent of where the scene sits on the master timeline.
 *
 * Audio rule — everything is an ``<Audio>`` element pinned to the frames it
 * belongs to; there is no single mixed-down track, so retiming a scene moves
 * its whoosh with it. ``deprecated`` is not used: the master voiceover runs
 * from frame 0 to the end.
 */

/** Where each SFX fires, in master-timeline frames. */
const CUES = {
  /** Impact under the first hook word. */
  hookImpact: 0,
  /** Second whoosh as the hook resolves into the number. */
  sceneChange: SCENES.metric.from - 6,
  /** Pop when the counter lands on its final value. */
  counterLand: SCENES.metric.from + 44,
  /** Pop as each badge arrives on the split screen. */
  badges: [SCENES.highlight.from + 44, SCENES.highlight.from + 55, SCENES.highlight.from + 66],
  /** Riser under the CTA. */
  riser: SCENES.cta.from - 14,
} as const;

const Whoosh: React.FC<{ at: number; volume?: number }> = ({ at, volume = 0.45 }) => (
  <Sequence from={at} durationInFrames={30} layout="none" name="whoosh">
    <Audio src={staticFile("whoosh.wav")} volume={volume} />
  </Sequence>
);

const Pop: React.FC<{ at: number; volume?: number }> = ({ at, volume = 0.5 }) => (
  <Sequence from={at} durationInFrames={12} layout="none" name="pop">
    <Audio src={staticFile("pop.wav")} volume={volume} />
  </Sequence>
);

/**
 * A full-frame flash used on the two hardest cuts (hook -> number, and into
 * the CTA). It is 4 frames of a very light overlay: enough to feel like an
 * edit point, not enough to wash out the frame.
 */
const Flash: React.FC<{ at: number; color?: string }> = ({ at, color = COLORS.white }) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame - at, [0, 4], [0.20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  if (frame < at || opacity <= 0) {
    return null;
  }
  return <AbsoluteFill style={{ backgroundColor: color, opacity }} />;
};

export const DemoReel: React.FC = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  // A 1-frame dip to black on the two internal cuts covers the scene swap
  // without needing a cross-fade (which would double the rasterisation cost).
  const cutDip = [
    SCENES.metric.from,
    SCENES.highlight.from,
    SCENES.cta.from,
  ].reduce((acc, cut) => {
    const near = Math.abs(frame - cut);
    return Math.max(acc, near === 0 ? 0.32 : near === 1 ? 0.14 : 0);
  }, 0);

  // The CTA holds to the final frame, so only the opening needs a fade-in.
  const fadeIn = interpolate(frame, [0, 6], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.ink, fontFamily: FONTS.text }}>
      {/* ---------------- visuals ---------------- */}
      <Sequence from={SCENES.hook.from} durationInFrames={SCENES.hook.duration} name="1 · hook">
        <SceneHook />
      </Sequence>
      <Sequence from={SCENES.metric.from} durationInFrames={SCENES.metric.duration} name="2 · metric">
        <SceneMetric />
      </Sequence>
      <Sequence
        from={SCENES.highlight.from}
        durationInFrames={SCENES.highlight.duration}
        name="3 · highlight"
      >
        <SceneHighlight />
      </Sequence>
      <Sequence from={SCENES.cta.from} durationInFrames={SCENES.cta.duration} name="4 · cta">
        <SceneCta />
      </Sequence>

      {/* ---------------- transitions ---------------- */}
      <Flash at={SCENES.metric.from - 1} />
      <Flash at={CUES.riser} color={COLORS.gold} />
      <AbsoluteFill style={{ backgroundColor: "#000", opacity: cutDip + (1 - fadeIn) }} />

      {/* ---------------- audio ---------------- */}
      {/* Voiceover: full timeline, eased out over the last 20 frames so the
          reel does not end on a clipped word. */}
      <Audio
        src={staticFile("voiceover.mp3")}
        volume={(f) =>
          interpolate(f, [0, 8, durationInFrames - 20, durationInFrames], [0, 1, 1, 0.55], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          })
        }
      />

      <Whoosh at={CUES.hookImpact} volume={0.5} />
      <Whoosh at={CUES.sceneChange} volume={0.42} />
      <Pop at={CUES.counterLand} volume={0.55} />
      {CUES.badges.map((at) => (
        <Pop key={at} at={at} volume={0.38} />
      ))}
      <Sequence from={CUES.riser} durationInFrames={62} layout="none" name="riser">
        <Audio
          src={staticFile("riser.wav")}
          volume={(f) => interpolate(f, [0, 50], [0.05, 0.4], { extrapolateRight: "clamp" })}
        />
      </Sequence>
    </AbsoluteFill>
  );
};

/**
 * Exported for the CI duration check: 450 frames at 30 fps is 15.000 s.
 * Kept as a constant so a change to either number is caught by the check
 * rather than silently producing a reel of the wrong length.
 */
export const REEL_SECONDS = DURATION_IN_FRAMES / FPS;
