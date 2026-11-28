import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { Grid } from "../components/Grid";
import { GlowOrb } from "../components/GlowOrb";
import { COLORS, FONTS, HOOK_LINES, SCENES } from "../theme";

/**
 * Scene 1 — the hook (0.0 – 3.5 s, frames 0–104).
 *
 * Word-by-word kinetic type: each word springs in from below with a slight
 * overshoot, staggered 4 frames apart, so the two lines land in under a
 * second. ALL-CAPS words render in Anton at 1.35x the body size with a gold
 * gradient; lowercase words sit in Inter. The whole block pushes in slowly
 * (1.0 -> 1.06) so the frame is never completely still.
 *
 * The whoosh is placed in DemoReel at the scene boundary; this component is
 * purely visual.
 */

const WORD_STAGGER = 4;

const HookWord: React.FC<{
  word: string;
  index: number;
  accent: boolean;
}> = ({ word, index, accent }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const delay = index * WORD_STAGGER;
  const enter = spring({
    frame: frame - delay,
    fps,
    config: { damping: 12, stiffness: 160, mass: 0.6 },
  });

  // A soft settle after the overshoot: the word drifts up 6px while it fades
  // in, then stops. interpolate() is clamped so it never runs backwards.
  const lift = interpolate(enter, [0, 1], [46, 0]);

  const blur = interpolate(enter, [0, 0.85], [10, 0], { extrapolateRight: "clamp" });

  return (
    <span
      style={{
        display: "inline-block",
        transform: `translateY(${lift}px) scale(${0.94 + enter * 0.06})`,
        opacity: Math.min(1, enter * 1.35),
        filter: blur > 0.2 ? `blur(${blur}px)` : undefined,
        fontFamily: accent ? FONTS.display : FONTS.text,
        fontSize: accent ? 168 : 104,
        fontWeight: accent ? 400 : 700,
        letterSpacing: accent ? "0.01em" : "-0.02em",
        lineHeight: accent ? 0.94 : 1.05,
        color: accent ? "transparent" : COLORS.white,
        backgroundImage: accent
          ? `linear-gradient(178deg, ${COLORS.white} 0%, ${COLORS.gold} 58%, ${COLORS.goldDeep} 100%)`
          : undefined,
        WebkitBackgroundClip: accent ? "text" : undefined,
        backgroundClip: accent ? "text" : undefined,
        textShadow: accent ? undefined : "0 12px 40px rgba(0,0,0,0.65)",
        marginRight: 30,
        marginBottom: 8,
      }}
    >
      {word}
    </span>
  );
};

export const SceneHook: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const scene = SCENES.hook;
  const local = frame;
  // IMPORTANT: this component lives inside a <Sequence>, so useCurrentFrame()
  // already returns the frame *relative to this scene*. Subtracting
  // SCENES.<x>.from here would offset it twice and leave every spring pinned
  // at its frame-0 value (invisible) — which is exactly what happened on the
  // first render: only the spring-free backgrounds appeared.

  // Slow push-in over the whole scene keeps the background alive.
  const push = interpolate(local, [0, scene.duration], [1.0, 1.06], {
    extrapolateRight: "clamp",
  });

  // Leave: the type drops away just before the cut, so scene 2 arrives onto a
  // clean frame instead of on top of the outgoing words.
  const exit = interpolate(
    local,
    [scene.duration - 14, scene.duration],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );

  const flash = spring({
    frame: local - 2,
    fps,
    config: { damping: 200 },
  });

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.ink }}>
      <Backdrop mood="gold" intensity={0.8 + flash * 0.4} />
      <Grid opacity={0.55} tint={COLORS.gold} />

      {/* Bloom behind the type, breathing on a slow sine. */}
      <AbsoluteFill
        style={{
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <GlowOrb
          size={1150}
          color={COLORS.gold}
          opacity={0.16 + Math.sin(local / 18) * 0.03}
          style={{ position: "absolute", top: 520 }}
        />
      </AbsoluteFill>

      <AbsoluteFill
        style={{
          justifyContent: "center",
          padding: "0 92px",
          transform: `scale(${push}) translateY(${exit * -70}px)`,
          opacity: 1 - exit,
        }}
      >
        {/* Kicker: the first thing on screen, and the quietest. */}
        <div
          style={{
            fontFamily: FONTS.text,
            fontSize: 30,
            fontWeight: 700,
            letterSpacing: "0.42em",
            color: COLORS.gold,
            opacity: interpolate(local, [0, 14], [0, 1], {
              extrapolateRight: "clamp",
            }),
            transform: `translateY(${interpolate(local, [0, 14], [16, 0], {
              extrapolateRight: "clamp",
            })}px)`,
            marginBottom: 44,
          }}
        >
          THE STOIC EDGE
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 30 }}>
          {HOOK_LINES.map((words, lineIndex) => (
            <div
              key={lineIndex}
              style={{
                display: "flex",
                flexWrap: "wrap",
                alignItems: "flex-end",
                // Line 2 starts after line 1 has finished arriving.
                marginLeft: lineIndex === 1 ? 40 : 0,
              }}
            >
              {words.map((word, wordIndex) => (
                <HookWord
                  key={word}
                  word={word}
                  index={lineIndex * words.length + wordIndex + 1}
                  accent={word === word.toUpperCase()}
                />
              ))}
            </div>
          ))}
        </div>

        {/* Underline that draws itself left-to-right under the second line. */}
        <div
          style={{
            marginTop: 40,
            marginLeft: 40,
            height: 6,
            width: 380,
            borderRadius: 3,
            background: `linear-gradient(90deg, ${COLORS.gold}, ${COLORS.violet})`,
            transform: `scaleX(${spring({
              frame: local - 26,
              fps,
              config: { damping: 22, stiffness: 90 },
            })})`,
            transformOrigin: "left center",
            boxShadow: `0 0 34px ${COLORS.gold}`,
          }}
        />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
