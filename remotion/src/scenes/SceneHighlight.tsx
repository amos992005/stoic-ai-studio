import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { GlowOrb } from "../components/GlowOrb";
import { COLORS, FONTS, SCENES } from "../theme";

/**
 * Scene 3 — the callout (7.5 – 11.5 s, local frames 0–119).
 *
 * A hard vertical split: "NOISE" on the left in dead grey, "SIGNAL" on the
 * right in gold with a gradient sweep across it. The two halves slide in from
 * opposite edges, then three badges arrive from the right, each with its own
 * spring so they read as separate beats rather than a list appearing at once.
 */

const BADGES = [
  { text: "DISCIPLINE", accent: COLORS.mint },
  { text: "PATIENCE", accent: COLORS.gold },
  { text: "REPETITION", accent: COLORS.violet },
] as const;

const Badge: React.FC<{
  text: string;
  accent: string;
  index: number;
  localFrame: number;
}> = ({ text, accent, index, localFrame }) => {
  const { fps } = useVideoConfig();
  const enter = spring({
    frame: localFrame - (44 + index * 11),
    fps,
    config: { damping: 14, stiffness: 130, mass: 0.7 },
  });

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 20,
        padding: "22px 34px",
        borderRadius: 999,
        border: `2px solid ${accent}66`,
        background: `linear-gradient(120deg, ${accent}22, rgba(255,255,255,0.03))`,
        transform: `translateX(${(1 - enter) * 260}px)`,
        opacity: enter,
        boxShadow: `0 0 46px -14px ${accent}`,
      }}
    >
      <div
        style={{
          width: 16,
          height: 16,
          borderRadius: "50%",
          background: accent,
          boxShadow: `0 0 22px ${accent}`,
        }}
      />
      <div
        style={{
          fontFamily: FONTS.text,
          fontSize: 38,
          fontWeight: 800,
          letterSpacing: "0.16em",
          color: COLORS.white,
        }}
      >
        {text}
      </div>
    </div>
  );
};

export const SceneHighlight: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const scene = SCENES.highlight;
  const local = frame - scene.from;

  const left = spring({
    frame: local - 1,
    fps,
    config: { damping: 20, stiffness: 80, mass: 1.1 },
  });
  const right = spring({
    frame: local - 6,
    fps,
    config: { damping: 20, stiffness: 80, mass: 1.1 },
  });

  // The gold sweep runs once across the right half.
  const sweep = interpolate(local, [30, 78], [-40, 140], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const exit = interpolate(
    local,
    [scene.duration - 14, scene.duration],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.ink }}>
      <Backdrop mood="gold" intensity={0.7} />

      <GlowOrb
        size={900}
        color={COLORS.gold}
        opacity={0.13}
        style={{ position: "absolute", left: 380, top: 620 }}
      />

      <AbsoluteFill style={{ opacity: 1 - exit }}>
        {/* ---- left: NOISE ---------------------------------------------- */}
        <AbsoluteFill
          style={{
            width: "50%",
            padding: "0 54px",
            justifyContent: "center",
            transform: `translateX(${(1 - left) * -520}px)`,
            opacity: 0.55 * left,
            borderRight: `1px solid ${COLORS.faint}`,
          }}
        >
          <div
            style={{
              fontFamily: FONTS.display,
              fontSize: 128,
              lineHeight: 0.9,
              color: "rgba(255,255,255,0.32)",
              letterSpacing: "-0.01em",
            }}
          >
            NOISE
          </div>
          <div
            style={{
              marginTop: 26,
              fontFamily: FONTS.text,
              fontSize: 30,
              fontWeight: 500,
              lineHeight: 1.4,
              color: "rgba(255,255,255,0.34)",
            }}
          >
            Cheap dopamine. Loud opinions. Zero reps.
          </div>
        </AbsoluteFill>

        {/* ---- right: SIGNAL -------------------------------------------- */}
        <AbsoluteFill
          style={{
            left: "50%",
            padding: "0 54px",
            justifyContent: "center",
            transform: `translateX(${(1 - right) * 520}px)`,
            opacity: right,
          }}
        >
          <div style={{ position: "relative" }}>
            <div
              style={{
                fontFamily: FONTS.display,
                fontSize: 148,
                lineHeight: 0.88,
                letterSpacing: "-0.01em",
                backgroundImage: `linear-gradient(96deg, ${COLORS.white} 0%, ${COLORS.gold} 52%, ${COLORS.goldDeep} 100%)`,
                WebkitBackgroundClip: "text",
                backgroundClip: "text",
                color: "transparent",
              }}
            >
              SIGNAL
            </div>
            {/* Specular sweep, masked to the word's bounding box. */}
            <div
              style={{
                position: "absolute",
                inset: 0,
                background: `linear-gradient(100deg, transparent ${sweep - 22}%, rgba(255,255,255,0.85) ${sweep}%, transparent ${sweep + 22}%)`,
                mixBlendMode: "overlay",
                pointerEvents: "none",
              }}
            />
          </div>

          <div
            style={{
              marginTop: 30,
              fontFamily: FONTS.text,
              fontSize: 30,
              fontWeight: 600,
              lineHeight: 1.4,
              color: COLORS.muted,
            }}
          >
            One hour, every day, when nobody claps.
          </div>

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "flex-start",
              gap: 18,
              marginTop: 44,
            }}
          >
            {BADGES.map((badge, index) => (
              <Badge
                key={badge.text}
                text={badge.text}
                accent={badge.accent}
                index={index}
                localFrame={local}
              />
            ))}
          </div>
        </AbsoluteFill>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
