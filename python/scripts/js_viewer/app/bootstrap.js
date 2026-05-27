// Module entry point. ``index.html`` injects this as
// ``<script type="module" src="app/bootstrap.js">`` once the lib
// dependencies (jQuery, plotly, d3, underscore) and the global
// ``imgPlotter`` from lib/img_plotting.js are loaded.

import {App} from "./app.js";

// Tiny helpers that the legacy waterfall.js exposed as globals; a couple of
// panels still expect to find them on ``window`` (e.g. baseline.js using
// ``_mean`` over the transposed scroll buffer).
window._dB = (d) => 10 * Math.log10(d);
window._mean = (d) => _.reduce(d, (memo, num) => memo + num, 0) / d.length || 1;

// Expose the app on ``window`` for in-browser debugging.
window.app = new App();
window.app.start();
