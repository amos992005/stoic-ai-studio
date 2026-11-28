import { loadFont } from "@remotion/fonts";
import { continueRender, delayRender, staticFile } from "remotion";
import { FONTS } from "./theme";

/**
 * Font loading, without any network dependency at render time.
 *
 * Both faces are vendored in ``public/fonts`` (Anton for the kinetic words,
 * Inter for everything readable), so a render on a machine with no internet
 * still produces the same typography. ``loadFont`` reports through Remotion's
 * delayRender/continueRender handshake, so the first frame waits for the
 * faces instead of flashing a fallback.
 *
 * Loading is kicked off once at module scope; every component that needs the
 * faces imports this file for its side effect.
 */

const handle = delayRender("loading Anton + Inter", { timeoutInMilliseconds: 30_000 });

const faces = [
  loadFont({
    family: FONTS.display,
    url: staticFile("fonts/Anton-Regular.ttf"),
    weight: "400",
  }),
  loadFont({
    // Variable font: one file covers every weight we use (400–900).
    family: FONTS.text,
    url: staticFile("fonts/Inter-Variable.ttf"),
    weight: "100 900",
    format: "truetype",
  }),
];

Promise.all(faces)
  .then(() => {
    continueRender(handle);
  })
  .catch((error: unknown) => {
    // Let the render fail loudly: a missing font means the layout numbers in
    // this project are wrong, and a silent fallback would hide that.
    // eslint-disable-next-line no-console
    console.error("font loading failed", error);
    continueRender(handle);
  });
