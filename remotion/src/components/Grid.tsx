import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { COLORS } from "../theme";

/**
 * A perspective grid that scrolls toward the viewer, with a glow that sweeps
 * along it.
 *
 * Built from CSS 3D transforms on a single square-backed element rather than
 * 40 individually animated lines: one composited layer is far cheaper to
 * rasterise 450 times, and the perspective does the visual work.
 */

export const Grid: React.FC<{
  /** 0–1, how bright the lines are. */
  opacity?: number;
  /** Hue for the sweep highlight. */
  tint?: string;
}> = ({ opacity = 0.5, tint = COLORS.gold }) => {
  const frame = useCurrentFrame();

  // One cell every 90px, moving 1.6px per frame: at 30fps that is a slow
  // glide, fast enough to feel alive without strobing on a phone.
  const offset = (frame * 1.6) % 90;
  const sweep = interpolate(frame % 150, [0, 75, 150], [-30, 130, -30]);

  const line = "rgba(255,255,255,0.16)";

  return (
    <AbsoluteFill style={{ perspective: 900, opacity }}>
      <AbsoluteFill
        style={{
          // Tuned so the grid's bright edge lands in the middle of the frame:
          // lower and it only fills the bottom quarter (which looked empty).
          transform:
            "rotateX(64deg) translateZ(-120px) translateY(-40px) scale(2.1)",
          transformOrigin: "50% 62%",
        }}
      >
        <AbsoluteFill
          style={{
            backgroundImage: `linear-gradient(${line} 1px, transparent 1px), linear-gradient(90deg, ${line} 1px, transparent 1px)`,
            backgroundSize: "90px 90px",
            backgroundPosition: `0px ${offset}px`,
            maskImage:
              "linear-gradient(to top, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.55) 45%, transparent 78%)",
            WebkitMaskImage:
              "linear-gradient(to top, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.55) 45%, transparent 78%)",
          }}
        />
        {/* The travelling highlight, clipped to the same mask. */}
        <AbsoluteFill
          style={{
            background: `linear-gradient(90deg, transparent ${sweep - 18}%, ${tint} ${sweep}%, transparent ${sweep + 18}%)`,
            opacity: 0.35,
            mixBlendMode: "screen",
            maskImage:
              "linear-gradient(to top, rgba(0,0,0,0.9) 0%, transparent 70%)",
            WebkitMaskImage:
              "linear-gradient(to top, rgba(0,0,0,0.9) 0%, transparent 70%)",
          }}
        />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
