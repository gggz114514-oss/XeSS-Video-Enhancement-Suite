const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
let extension;
const source = fs.readFileSync(path.join(__dirname, '../web/r4_nodes.js'), 'utf8')
    .replace(/^import .*?;\s*/m, '').replace('export function', 'function');
const context = vm.createContext({app:{registerExtension(e){extension=e;}}});
vm.runInContext(source, context);
function make(type) {
    function Node() {
        this.comfyClass=type; this.size=[340,600];
        this.widgets=['backend','depth','scale','custom_scale','encoder','sharpen','five_frame','anti_stripe'].map(name=>({
            name, value: name==='backend'?'GPU Block（快速）':name==='scale'?'1.5×':name==='encoder'?'FFV1（无损）':true,
            type:'combo', computeSize(){return [300,24];},
        }));
    }
    Node.prototype.computeSize=function(){return [340,300];};
    Node.prototype.setSize=function(v){this.size=v;};
    extension.beforeRegisterNodeDef(Node,{name:type});
    const n=new Node(); n.onNodeCreated(); return n;
}
const n=make('XeSSR4OfflineSuperResolutionFrameGeneration');
const widgets=n.widgets;
const w=Object.fromEntries(widgets.map(x=>[x.name,x]));
assert.equal(w.depth.value,'AI 深度');
assert.equal(w.depth.type,'converted-widget');
w.backend.value='Intel 视频接口'; w.backend.callback();
for(const effect of ['sharpen','five_frame','anti_stripe']) {
    assert.equal(w[effect].type,'converted-widget'); assert.equal(w[effect].value,false);
}
w.backend.value='GPU DIS（实验）'; w.backend.callback();
assert.equal(w.anti_stripe.type,'combo');
assert.equal(w.five_frame.type,'combo');
assert.equal(w.encoder.value,'FFV1（无损）');
assert.equal(n.widgets,widgets); // never rebuilt, serialized order unchanged
w.scale.value='自定义'; w.scale.callback(); assert.equal(w.custom_scale.type,'combo');
w.backend.value='CPU DIS（稳定）'; n.onConfigure();
assert.equal(w.depth.type,'combo'); assert.equal(w.five_frame.type,'converted-widget');
const fg=make('XeSSR4OfflineFrameGeneration');
assert.equal(fg.widgets.find(x=>x.name==='anti_stripe').type,'converted-widget');
assert.equal(fg.widgets.find(x=>x.name==='sharpen').type,'combo');
console.log('R4 widget visibility/state regression passed');
