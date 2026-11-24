// Module entry point. ``index.html`` injects this as
// ``<script type="module" src="app/bootstrap.js">`` once the lib
// dependencies (jQuery, plotly, d3, underscore) and the global
// ``imgPlotter`` from lib/img_plotting.js are loaded.

import {App} from "./app.js";

// Expose the app on ``window`` for in-browser debugging.
window.app = new App();
window.app.start();
