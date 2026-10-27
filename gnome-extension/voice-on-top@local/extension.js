import GLib from 'gi://GLib';
import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';

const APP_ID = 'dev.local.Voice';

function pin(win) {
    if (!win || (win.get_gtk_application_id() !== APP_ID && win.get_wm_class() !== APP_ID))
        return false;
    win.make_above();
    win.stick();
    return true;
}

export default class VoiceOnTop extends Extension {
    enable() {
        // Wayland clients set their app id after the window is created, so check again shortly after.
        this._handler = global.display.connect('window-created', (_display, win) => {
            if (!pin(win))
                GLib.timeout_add(GLib.PRIORITY_DEFAULT, 300, () => { pin(win); return GLib.SOURCE_REMOVE; });
        });
        for (const actor of global.get_window_actors())
            pin(actor.meta_window);
    }

    disable() {
        global.display.disconnect(this._handler);
        this._handler = 0;
    }
}
