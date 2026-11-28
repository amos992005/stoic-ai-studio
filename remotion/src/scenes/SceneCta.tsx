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
import { GlowOrb, PulseRing } from "../components/GlowOrb";
import { COLORS, CTA, FONTS, SCENES } from "../theme";

/**
 * Scene 4 — the outro (11.5 – 15.0 s, local frames 0–104).
 *
 * The CTA lands hard: two words slam in, then the third drops onto them, and
 * behind it all a glow pulses on a 24-frame beat with expanding rings. The
 * scene holds rather than exits — the last frame of the reel is a poster the
 * viewer can read, which is what a CTA is for.
 */

const PULSE_PERIOD = 24;

export const SceneCta: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const scene = SCENES.cta;
  const local = frame;
  // IMPORTANT: this component lives inside a <Sequence>, so useCurrentFrame()
  // already returns the frame *relative to this scene*. Subtracting
  // SCENES.<x>.from here would offset it twice and leave every spring pinned
  // at its frame-0 value (invisible) — which is exactly what happened on the
  // first render: only the spring-free backgrounds appeared.

  const slam = (delay: number) =>
    spring({
      frame: local - delay,
      fps,
      config: { damping: 13, stiffness: 190, mass: 0.55 },
    });

  const kicker = slam(0);
  const first = slam(5);
  const accent = slam(12);
  const footnote = slam(30);

  // Heartbeat: 0 -> 1 -> 0 across each pulse period.
  const beat = Math.sin((local / PULSE_PERIOD) * Math.PI * 2) * 0.5 + 0.5;
  const ringProgress = (local % PULSE_PERIOD) / PULSE_PERIOD;

  // The very last frames fade the whole reel down to black for a clean loop.
  const end = interpolate(local, [scene.duration - 8, scene.duration], [1, 0.92], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.ink, opacity: end }}>
      <Backdrop mood="gold" intensity={0.85 + beat * 0.25} />
      <Grid opacity={0.4} tint={COLORS.gold} />

      <AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
        <GlowOrb
          size={1250}
          color={COLORS.gold}
          opacity={0.18 + beat * 0.12}
          style={{ position: "absolute" }}
        />
        {/* Two rings, offset by half a period, so the pulse never stops. */}
        <PulseRing size={720} progress={ringProgress} color={COLORS.gold} opacity={0.5} />
        <PulseRing
          size={720}
          progress={(ringProgress + 0.5) % 1}
          color={COLORS.violet}
          opacity={0.4}
        />
      </AbsoluteFill>

      <AbsoluteFill
        style={{
          justifyContent: "center",
          alignItems: "center",
          padding: "0 80px",
          textAlign: "center",
        }}
      >
        <div
          style={{
            fontFamily: FONTS.text,
            fontSize: 28,
            fontWeight: 700,
            letterSpacing: "0.4em",
            color: COLORS.gold,
            opacity: kicker,
            transform: `translateY(${(1 - kicker) * 26}px)`,
            marginBottom: 34,
          }}
        >
          {CTA.kicker}
        </div>

        <div
          style={{
            fontFamily: FONTS.display,
            fontSize: 168,
            lineHeight: 0.9,
            color: COLORS.white,
            letterSpacing: "-0.01em",
            transform: `scale(${0.82 + first * 0.18})`,
            opacity: Math.min(1, first * 1.4),
            textShadow: `0 0 ${60 * beat}px rgba(245,196,107,0.35)`,
          }}
        >
          {CTA.line}
        </div>

        <div
          style={{
            fontFamily: FONTS.display,
            fontSize: 132,
            lineHeight: 1,
            letterSpacing: "0.02em",
            backgroundImage: `linear-gradient(180deg, ${COLORS.white} 0%, ${COLORS.gold} 55%, ${COLORS.goldDeep} 100%)`,
            WebkitBackgroundClip: "text",
            backgroundClip: "text",
            color: "transparent",
            transform: `scale(${0.7 + accent * 0.3}) translateY(${(1 - accent) * 40}px)`,
            opacity: Math.min(1, accent * 1.4),
            filter: `drop-shadow(0 0 ${40 + beat * 30}px rgba(245,196,107,0.55))`,
          }}
        >
          {CTA.accent}
        </div>

        <div
          style={{
            marginTop: 56,
            display: "flex",
            alignItems: "center",
            gap: 22,
            opacity: footnote,
            transform: `translateY(${(1 - footnote) * 30}px)`,
          }}
        >
          <div
            style={{
              width: 74,
              height: 2,
              background: `linear-gradient(90deg, transparent, ${COLORS.gold})`,
            }}
          />
          <div
            style={{
              fontFamily: FONTS.text,
              fontSize: 34,
              fontWeight: 800,
              letterSpacing: "0.3em",
              color: COLORS.white,
            }}
          >
            {CTA.footnote}
          </div>
          <div
            style={{
              width: 74,
              height: 2,
              background: `linear-gradient(90deg, ${COLORS.gold}, transparent)`,
            }}
          />
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
