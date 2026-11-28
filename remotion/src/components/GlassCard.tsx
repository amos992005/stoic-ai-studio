import React from "react";
import { AbsoluteFill } from "remotion";
import { COLORS } from "../theme";

/**
 * A glassmorphism panel: translucent surface, 1px light rim, and a top-edge
 * sheen that reads as a reflection.
 *
 * ``backdropFilter: blur()`` is used sparingly here because there is exactly
 * one of these on screen at a time — it is the effect the brief asks for and
 * the cost is one layer, not a blurred stack.
 */

export const GlassCard: React.FC<{
  children: React.ReactNode;
  style?: React.CSSProperties;
  /** Strength of the rim light, 0–1. */
  glow?: number;
  tint?: string;
}> = ({ children, style, glow = 1, tint = COLORS.gold }) => {
  return (
    <div
      style={{
        position: "relative",
        borderRadius: 44,
        padding: "56px 52px",
        background:
          "linear-gradient(160deg, rgba(255,255,255,0.13) 0%, rgba(255,255,255,0.05) 42%, rgba(255,255,255,0.03) 100%)",
        backdropFilter: "blur(26px) saturate(140%)",
        WebkitBackdropFilter: "blur(26px) saturate(140%)",
        border: "1px solid rgba(255,255,255,0.16)",
        boxShadow: [
          `0 40px 90px -30px rgba(0,0,0,0.85)`,
          `inset 0 1px 0 rgba(255,255,255,0.28)`,
          `0 0 ${80 * glow}px -20px ${tint}`,
        ].join(", "),
        overflow: "hidden",
        ...style,
      }}
    >
      {/* top-left sheen */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(120% 90% at 12% -10%, rgba(255,255,255,0.20) 0%, transparent 55%)`,
          pointerEvents: "none",
        }}
      />
      <div style={{ position: "relative" }}>{children}</div>
    </div>
  );
};
