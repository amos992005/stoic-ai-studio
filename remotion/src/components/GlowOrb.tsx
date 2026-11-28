import React from "react";
import { COLORS } from "../theme";

/**
 * A soft radial bloom used for emphasis behind type and behind the CTA.
 *
 * Two stacked radial-gradients instead of a blurred div: identical look, no
 * filter, so the CPU renderer stays fast on 15 s of full-frame animation.
 */
export const GlowOrb: React.FC<{
  size: number;
  color?: string;
  opacity?: number;
  style?: React.CSSProperties;
}> = ({ size, color = COLORS.gold, opacity = 0.55, style }) => {
  return (
    <div
      style={{
        width: size,
        height: size,
        background: `radial-gradient(circle, ${color} 0%, ${color}00 62%)`,
        opacity,
        borderRadius: "50%",
        pointerEvents: "none",
        ...style,
      }}
    />
  );
};

/** A thin ring that can be scaled — used as the "pulse" in the outro. */
export const PulseRing: React.FC<{
  size: number;
  color?: string;
  progress: number;
  opacity?: number;
}> = ({ size, color = COLORS.gold, progress, opacity = 1 }) => {
  const scale = 0.85 + progress * 0.55;
  const fade = Math.max(0, 1 - progress) * opacity;
  return (
    <div
      style={{
        position: "absolute",
        width: size,
        height: size,
        borderRadius: "50%",
        border: `3px solid ${color}`,
        transform: `scale(${scale})`,
        opacity: fade,
        pointerEvents: "none",
      }}
    />
  );
};
