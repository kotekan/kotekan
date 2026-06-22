// Tiny event bus wrapping ``EventTarget``. Use ``bus.on(name, fn)`` to
// listen and ``bus.emit(name, detail)`` to publish; ``fn`` receives
// ``detail`` directly (no wrapper event object) so panels can stay terse.
export class Bus {
    constructor() {
        this._t = new EventTarget();
    }

    on(name, handler) {
        const wrapper = (e) => handler(e.detail);
        this._t.addEventListener(name, wrapper);
        return () => this._t.removeEventListener(name, wrapper);
    }

    emit(name, detail) {
        this._t.dispatchEvent(new CustomEvent(name, {detail}));
    }
}
