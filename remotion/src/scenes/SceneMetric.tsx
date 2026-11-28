import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { GlassCard } from "../components/GlassCard";
import { GlowOrb } from "../components/GlowOrb";
import { COLORS, FONTS, METRIC, SCENES } from "../theme";

/**
 * Scene 2 — the number (3.5 – 7.5 s, local frames 0–119).
 *
 * A glass card rises into frame, a big numeral counts up from 0, and four
 * bars grow in sequence beneath it. The count uses ``interpolate`` on a
 * spring-shaped curve so it decelerates into its final value instead of
 * stopping dead, and the bars are staggered so attention walks left to right.
 */

const BAR_MAX_HEIGHT = 250;

const Bar: React.FC<{
  label: string;
  value: number;
  index: number;
  localFrame: number;
}> = ({ label, value, index, localFrame }) => {
  const { fps } = useVideoConfig();
  const grow = spring({
    frame: localFrame - (46 + index * 7),
    fps,
    config: { damping: 16, stiffness: 110, mass: 0.8 },
  });
  const height = (value / 100) * BAR_MAX_HEIGHT * grow;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "flex-end",
        gap: 18,
        flex: 1,
      }}
    >
      <div
        style={{
          fontFamily: FONTS.text,
          fontSize: 28,
          fontWeight: 700,
          color: COLORS.muted,
          opacity: grow,
        }}
      >
        {Math.round(value * grow)}
      </div>
      <div
        style={{
          width: "100%",
          height,
          minHeight: 4,
          borderRadius: 14,
          background: `linear-gradient(180deg, ${COLORS.mint} 0%, ${COLORS.violet} 55%, rgba(139,124,246,0.25) 100%)`,
          boxShadow: `0 0 42px -8px ${COLORS.mint}`,
          opacity: 0.35 + grow * 0.65,
        }}
      />
      <div
        style={{
          fontFamily: FONTS.text,
          fontSize: 26,
          fontWeight: 600,
          letterSpacing: "0.06em",
          color: COLORS.white,
          opacity: 0.55 + grow * 0.45,
          whiteSpace: "nowrap",
        }}
      >
        {label}
      </div>
    </div>
  );
};

export const SceneMetric: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const scene = SCENES.metric;
  const local = frame;
  // IMPORTANT: this component lives inside a <Sequence>, so useCurrentFrame()
  // already returns the frame *relative to this scene*. Subtracting
  // SCENES.<x>.from here would offset it twice and leave every spring pinned
  // at its frame-0 value (invisible) — which is exactly what happened on the
  // first render: only the spring-free backgrounds appeared.

  const cardIn = spring({
    frame: local - 2,
    fps,
    config: { damping: 18, stiffness: 90, mass: 1 },
  });

  // Count 0 -> 1 decelerating: a spring's value is used as the easing curve.
  const count = spring({
    frame: local - 14,
    fps,
    config: { damping: 26, stiffness: 55, mass: 1.2 },
  });
  const displayed = METRIC.value * Math.min(1, count);

  const exit = interpolate(
    local,
    [scene.duration - 14, scene.duration],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.ink }}>
      <Backdrop mood="violet" intensity={0.95} />

      <AbsoluteFill
        style={{
          justifyContent: "center",
          alignItems: "center",
          transform: `scale(${interpolate(cardIn, [0, 1], [0.9, 1])}) translateY(${(1 - cardIn) * 90 - exit * 60}px)`,
          opacity: cardIn * (1 - exit),
        }}
      >
        <GlowOrb
          size={980}
          color={COLORS.mint}
          opacity={0.14}
          style={{ position: "absolute" }}
        />

        <GlassCard style={{ width: 900 }} glow={0.8} tint={COLORS.mint}>
          <div
            style={{
              fontFamily: FONTS.text,
              fontSize: 26,
              fontWeight: 700,
              letterSpacing: "0.34em",
              color: COLORS.mint,
              marginBottom: 18,
            }}
          >
            THE COMPOUND CURVE
          </div>

          <div style={{ display: "flex", alignItems: "baseline", gap: 18 }}>
            <div
              style={{
                fontFamily: FONTS.display,
                fontSize: 260,
                lineHeight: 0.86,
                color: COLORS.white,
                // Thirds of a percent are invisible at this size; whole and
                // one-decimal steps keep the count legible while it climbs.
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {displayed.toFixed(0)}
            </div>
            <div
              style={{
                fontFamily: FONTS.display,
                fontSize: 150,
                lineHeight: 1,
                color: COLORS.gold,
              }}
            >
              {METRIC.suffix}
            </div>
          </div>

          <div
            style={{
              fontFamily: FONTS.text,
              fontSize: 46,
              fontWeight: 700,
              color: COLORS.white,
              marginTop: 4,
              opacity: interpolate(local, [18, 32], [0, 1], {
                extrapolateRight: "clamp",
              }),
            }}
          >
            {METRIC.label}
          </div>
          <div
            style={{
              fontFamily: FONTS.text,
              fontSize: 32,
              fontWeight: 500,
              color: COLORS.muted,
              marginTop: 10,
              opacity: interpolate(local, [24, 40], [0, 1], {
                extrapolateRight: "clamp",
              }),
            }}
          >
            {METRIC.caption}
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "flex-end",
              gap: 22,
              height: BAR_MAX_HEIGHT + 100,
              marginTop: 38,
              paddingTop: 10,
              borderTop: `1px solid ${COLORS.faint}`,
            }}
          >
            {METRIC.bars.map((bar, index) => (
              <Bar
                key={bar.label}
                label={bar.label}
                value={bar.value}
                index={index}
                localFrame={local}
              />
            ))}
          </div>
        </GlassCard>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
