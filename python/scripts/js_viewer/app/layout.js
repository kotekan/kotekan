// Thin wrapper around GridStack v10 that gives each panel a card with a
// title-bar drag handle and exposes a stable DOM mount-point id inside.
// Panel constructors continue to receive a ``target`` id and call
// ``$("#" + target)`` -- they don't need to know about GridStack at all.

// localStorage key for the user's saved layout (positions + sizes). Bump
// this when the default widget set changes shape (eg the Controls card got
// split into Display / Buffer / Baseline / Airspy / Control) so a stale
// saved layout can't collide with the new defaults.
const STORAGE_KEY = "airspy_viewer_layout_v3";

export class LayoutManager {
    constructor({root}) {
        // GridStack expects to be initialised on a ``.grid-stack`` element.
        // ``index.html`` ships one with id ``layout-root``.
        this._root = $(root);
        this._root.addClass("grid-stack");
        this.grid = GridStack.init({
            cellHeight: 60,
            margin: 8,
            column: 12,
            float: true,
            // Only the title-bar of each card initiates a drag; clicks /
            // jQuery-UI sliders / Plotly drags inside the body don't move
            // the panel around by accident.
            handle: ".panel-header",
            resizable: {handles: "e, se, s, sw, w"},
        }, this._root[0]);

        this._mounts = new Map();    // mount_id -> widget DOM element
        this._pending_save = null;

        this.grid.on("change", () => this._debounced_save());
        this.grid.on("added",  () => this._debounced_save());
        this.grid.on("removed",() => this._debounced_save());
    }

    /**
     * Add a card to the grid. ``mount_id`` becomes the id of an empty div
     * inside the card body; panel constructors mount to that.
     */
    addWidget({mount_id, title, x, y, w, h, min_w, min_h}) {
        const html = `
            <div class="panel-card">
              <div class="panel-header">${title}</div>
              <div class="panel-body"><div id="${mount_id}"></div></div>
            </div>`;
        const item = this.grid.addWidget({
            id: mount_id,
            x, y, w, h,
            minW: min_w || 2,
            minH: min_h || 2,
            content: html,
        });
        this._mounts.set(mount_id, item);
        return item;
    }

    /**
     * Apply a saved layout (positions only -- we keep our own widget set,
     * which is determined by the viewer's enabled modules). Missing
     * entries in the saved layout are left at their defaults; entries in
     * the saved layout that don't match a current widget are ignored.
     */
    restore_from_storage() {
        let saved;
        try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY)); }
        catch (e) { return; }
        if (!saved || !saved.children) return;

        // GridStack v10's load() with addAndRemove=false adjusts existing
        // widgets to match the saved layout but doesn't create/destroy any.
        // It matches by ``id`` so mount_id == id is important.
        try {
            this.grid.load(saved.children, false);
        } catch (e) {
            console.warn("Failed to restore saved layout:", e);
        }
    }

    reset_layout() {
        localStorage.removeItem(STORAGE_KEY);
        location.reload();
    }

    _debounced_save() {
        clearTimeout(this._pending_save);
        this._pending_save = setTimeout(() => {
            try {
                const dump = this.grid.save(false);
                localStorage.setItem(STORAGE_KEY, JSON.stringify({children: dump}));
            } catch (e) {
                console.warn("Failed to save layout:", e);
            }
        }, 200);
    }
}
