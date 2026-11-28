import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { COLORS } from "../theme";

/**
 * The base every scene sits on: a dark gradient with two slow-moving light
 * blooms and a fine grain.
 *
 * Cheap on purpose — it renders on every frame of every scene, so there is no
 * blur filter and no large image, just gradients and one small repeating SVG
 * noise data-URI. Blur on a 1080x1920 layer is what makes CPU renders slow.
 */

const GRAIN =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" width="180" height="180">
       <filter id="n">
         <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="3" stitchTiles="stitch"/>
         <feColorMatrix type="saturate" values="0"/>
       </filter>
       <rect width="180" height="180" filter="url(#n)" opacity="0.5"/>
     </svg>`,
  );

export const Backdrop: React.FC<{
  /** Hue leaning of the top bloom: "gold" for warm scenes, "violet" for cool. */
  mood?: "gold" | "violet";
  /** Overall brightness of the blooms, 0–1. */
  intensity?: number;
}> = ({ mood = "gold", intensity = 1 }) => {
  const frame = useCurrentFrame();

  const primary = mood === "gold" ? COLORS.gold : COLORS.violet;
  const secondary = mood === "gold" ? COLORS.violet : COLORS.mint;

  // Slow, non-looping drift: two sine waves at different periods so the
  // motion never visibly repeats during a scene.
  const driftX = Math.sin(frame / 90) * 60;
  const driftY = Math.cos(frame / 130) * 80;

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(120% 80% at 50% 0%, ${COLORS.inkSoft} 0%, ${COLORS.ink} 55%, #000 100%)`,
      }}
    >
      <AbsoluteFill
        style={{
          background: `radial-gradient(45% 32% at ${50 + driftX / 12}% ${18 + driftY / 40}%, ${primary} 0%, transparent 70%)`,
          opacity: 0.22 * intensity,
        }}
      />
      <AbsoluteFill
        style={{
          background: `radial-gradient(40% 28% at ${78 - driftX / 16}% ${72 + driftY / 30}%, ${secondary} 0%, transparent 72%)`,
          opacity: 0.18 * intensity,
        }}
      />
      {/* Vignette keeps the eye on the centre column, where the type lives. */}
      <AbsoluteFill
        style={{
          background:
            "radial-gradient(75% 55% at 50% 46%, transparent 40%, rgba(0,0,0,0.72) 100%)",
        }}
      />
      <AbsoluteFill
        style={{
          backgroundImage: `url("${GRAIN}")`,
          backgroundRepeat: "repeat",
          opacity: 0.05,
          mixBlendMode: "overlay",
        }}
      />
      <AbsoluteFill
        style={{
          background: interpolate(intensity, [0, 1], [0.35, 0]),
        }}
      />
    </AbsoluteFill>
  );
};
