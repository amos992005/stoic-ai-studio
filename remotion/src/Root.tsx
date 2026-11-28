import React from "react";
import { Composition } from "remotion";
import { DemoReel } from "./DemoReel";
import { DURATION_IN_FRAMES, FPS, HEIGHT, WIDTH } from "./theme";

/**
 * Composition registry.
 *
 * ``id="DemoReel"`` is what CI names on the command line, so it must not be
 * renamed. 1080x1920 @ 30fps for 450 frames = exactly 15.000 s.
 */
export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="DemoReel"
        component={DemoReel}
        durationInFrames={DURATION_IN_FRAMES}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />
    </>
  );
};
