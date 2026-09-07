import { app } from "../../scripts/app.js";

// Change existing widgets in place: no node rebuild, no value-order changes.
function visible(widget, show) {
    if (!widget) return;
    if (!widget._xessOriginal) widget._xessOriginal = {
        type: widget.type, computeSize: widget.computeSize,
    };
    widget.type = show ? widget._xessOriginal.type : "converted-widget";
    widget.computeSize = show ? widget._xessOriginal.computeSize : () => [0, -4];
}

export function updateOptions(node) {
    const w = Object.fromEntries((node.widgets || []).map(x => [x.name, x]));
    const backend = String(w.backend?.value || "");
    const intel = backend.includes("Intel");
    const cpu = backend.includes("CPU DIS");
    const fg = node.comfyClass === "XeSSR4OfflineFrameGeneration";
    const combo = node.comfyClass === "XeSSR4OfflineSuperResolutionFrameGeneration";
    visible(w.depth, cpu);
    if (!cpu && w.depth) w.depth.value = "AI 深度";
    for (const name of ["sharpen", "five_frame", "anti_stripe"]) {
        const show = !intel && (name === "sharpen" || !fg) &&
            !(name === "five_frame" && cpu && combo);
        visible(w[name], show);
        if (!show && w[name]) w[name].value = false;
    }
    visible(w.custom_scale, w.scale?.value === "自定义");
    node.setSize?.([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "XeSS.R4Options",
    beforeRegisterNodeDef(Node, data) {
        if (!data.name?.startsWith("XeSSR4Offline")) return;
        const created = Node.prototype.onNodeCreated;
        Node.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            for (const widget of this.widgets || []) {
                if (!["backend", "scale"].includes(widget.name)) continue;
                const callback = widget.callback;
                widget.callback = (...values) => {
                    callback?.apply(widget, values);
                    updateOptions(this);
                };
            }
            updateOptions(this);
            return result;
        };
        const configured = Node.prototype.onConfigure;
        Node.prototype.onConfigure = function (...args) {
            const result = configured?.apply(this, args);
            updateOptions(this);
            return result;
        };
    },
});
