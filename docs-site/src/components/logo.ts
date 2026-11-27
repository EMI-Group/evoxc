/**
 * The site's logo, drawn once for every place it appears: the header and the
 * docs layout (Logo.astro) and the social cards (src/pages/og/_card.ts).
 * The evoxc mark — an SSA dataflow glyph in which two input nodes feed one op
 * node, which feeds one output node — reduced to fill-only paths so each place
 * paints it with its own single colour. `width`/`height` are the drawing's
 * tight box; callers size it by height and keep the ratio.
 */
export const logo = {
  width: 90,
  height: 66,
  paths: [
    // two input nodes (rings)
    "M0 11A11 11 0 1 1 22 11A11 11 0 1 1 0 11ZM5 11A6 6 0 1 0 17 11A6 6 0 1 0 5 11Z",
    "M0 55A11 11 0 1 1 22 55A11 11 0 1 1 0 55ZM5 55A6 6 0 1 0 17 55A6 6 0 1 0 5 55Z",
    // the output node (ring)
    "M68 33A11 11 0 1 1 90 33A11 11 0 1 1 68 33ZM73 33A6 6 0 1 0 85 33A6 6 0 1 0 73 33Z",
    // the op node (filled, one per function)
    "M32 33A13 13 0 1 1 58 33A13 13 0 1 1 32 33Z",
    // SSA edges: both inputs into the op, the op into the output
    "M18.74 19.28 L32.59 28.25 L35.58 23.63 L21.73 14.67Z",
    "M21.73 51.33 L35.58 42.37 L32.59 37.75 L18.74 46.72Z",
    "M58 35.75 L68 35.75 L68 30.25 L58 30.25Z",
  ],
};
