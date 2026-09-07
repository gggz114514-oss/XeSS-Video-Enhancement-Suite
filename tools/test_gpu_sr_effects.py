"""Bounded native effect validation; same source frames, no full-video raw cache."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import offline_toolbox as box


def stats(a, b):
    delta = a.astype(np.float64)-b.astype(np.float64)
    mse = float(np.mean(delta*delta))
    return {"max_abs": int(np.max(np.abs(delta))), "mean_abs": float(np.mean(np.abs(delta))),
            "psnr": 10*np.log10(255**2/mse) if mse else None, "equal": bool(np.array_equal(a,b))}


def bilinear(a, x, y):
    h,w = a.shape[:2]
    x=np.clip(x,0,w-1);y=np.clip(y,0,h-1)
    x0=x.astype(np.int32);y0=y.astype(np.int32);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    fx=(x-x0).astype(np.float32)[...,None];fy=(y-y0).astype(np.float32)[...,None]
    return ((a[y0,x0]*(1-fx)+a[y0,x1]*fx)*(1-fy)+(a[y1,x0]*(1-fx)+a[y1,x1]*fx)*fy)


def main():
    ap=argparse.ArgumentParser()
    for name in ("runtime","baseline-runtime","source","work"):ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--routes", nargs="+", choices=("gpu-block", "gpu-dis", "amd-of"), default=["gpu-block", "gpu-dis"])
    ap.add_argument("--frames", type=int, choices=range(7, 17), default=7)
    args=ap.parse_args();args.work.mkdir(parents=True,exist_ok=False)
    paths=box.runtime_paths(args.runtime)
    # At least seven for slot wrap; AMD uses twelve to include post-warmup flow.
    n=args.frames; vw,vh=864,480; ow,oh=1296,720
    src=args.work/"input.h264"
    subprocess.run([str(paths["ffmpeg"]),"-v","error","-i",str(args.source),"-frames:v",str(n),
                    "-an","-c:v","libx264","-crf","18","-bf","0","-pix_fmt","yuv420p","-f","h264",str(src)],check=True)
    results={}
    for route in args.routes:
        outputs={}
        for kind in ("old","off","guard","five","both"):
            root=args.baseline_runtime if kind=="old" else args.runtime
            pth=box.runtime_paths(root);case=args.work/(route+"-"+kind);case.mkdir();diag=case/"diag";diag.mkdir()
            cmd=[str(pth["bin"]/box.WORKERS[route]),"--input",str(src),"--codec","h264","--max-frames",str(n),
                 "--fps","24","--shader-dir",str(pth["shaders"]),"--qsv-out",str(case/"output.h264"),
                 "--report",str(case/"native.json"),"--gpu-mode","sr","--output-width",str(ow),"--output-height",str(oh),
                 "--motion-backend",route,"--motion-scale",".5" if route=="gpu-block" else "1",
                 "--motion-repair","refine" if route=="gpu-block" else "off","--slots","4","--xess-quality","ultra-quality",
                 "--gpu-post","off","--diagnostic-core-dir",str(diag),
                 "--depth-model",str(pth["model"]),"--gpu-depth-input","ffmpeg-rgb"]
            if route=="gpu-dis":cmd += ["--dis-shader-dir",str(pth["dis_shaders"])]
            if kind!="old":cmd += ["--gpu-five-frame","on" if kind in ("five","both") else "off",
                                   "--gpu-anti-stripe","on" if kind in ("guard","both") else "off"]
            env=dict(os.environ,TEMP=str(case),TMP=str(case),PYTHONIOENCODING="utf-8")
            with (case/"native.log").open("wb") as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=pth["bin"],timeout=90,check=True)
            report=json.loads((case/"native.json").read_text())
            outputs[kind]=[]
            for i in range(n):
                outputs[kind].append(np.fromfile(diag/f"output{i}-post.bin",dtype=np.uint8).reshape(oh,ow,4)[...,:3].copy())
            effects=report.get("sr_effects",{})
            if kind in ("guard","five","both"):assert effects["frames"]==n
            if kind in ("five","both"):assert effects["history_samples"]==sum(min(i,4) for i in range(n)), effects
        def source(i):
            # Native diagnostic naming is part of this explicit test contract.
            return np.fromfile(args.work/(route+"-off")/"diag"/f"frame{i}-color.bin",dtype=np.uint8).reshape(vh,vw,4)[...,:3].copy()
        references=[];fusion_refs=[];both_refs=[]
        cols=[source(i).astype(np.float32) for i in range(n)]
        flows=[np.fromfile(args.work/(route+"-five")/"diag"/f"frame{i}-motion.bin",dtype=np.float16).astype(np.float32).reshape(vh,vw,2) for i in range(n)]
        gx,gy=np.meshgrid(np.arange(ow,dtype=np.float32),np.arange(oh,dtype=np.float32))
        bx=np.clip((gx+.5)*vw/ow-.5,0,vw-1);by=np.clip((gy+.5)*vh/oh-.5,0,vh-1)
        weights=[.34,.17,.10,.06]
        for i,raw in enumerate(outputs["off"]):
            guide=cv2.resize(cols[i].astype(np.uint8),(ow,oh),interpolation=cv2.INTER_CUBIC).astype(np.float32)
            gray=cv2.cvtColor(guide,cv2.COLOR_RGB2GRAY)
            blend=np.clip((np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3))/8-.5)/4,0,1)
            blend=np.clip(cv2.GaussianBlur(blend,(0,0),2.5)*.9,0,.9)[...,None]
            references.append(np.clip(raw*(1-blend)+guide*blend,0,255).astype(np.uint8))
            x=bx.copy();y=by.copy();center=bilinear(cols[i],bx,by);total=np.ones((oh,ow,1),np.float32);acc=np.zeros_like(center)
            alive=np.ones((oh,ow),bool)
            for step in range(min(i,4)):
                flow=bilinear(flows[i-step],x,y);x+=flow[...,0];y+=flow[...,1]
                alive&=(x>=0)&(y>=0)&(x<=vw-1)&(y<=vh-1)
                prior=bilinear(cols[i-step-1],x,y)
                err=np.abs(np.sum((prior-center)*np.array([.299,.587,.114],np.float32),axis=2))
                weight=(.35*weights[step]*np.exp(-err/11)*np.clip((26-err)/18,0,1)
                        *np.clip((96-np.sqrt((x-bx)**2+(y-by)**2))/64,0,1)*alive)[...,None]
                acc+=np.clip(prior-center,-24,24)*weight;total+=weight
            fused=raw.astype(np.float32)+acc/total
            fusion_refs.append(np.floor(np.clip(fused,0,255)+.00001).astype(np.uint8))
            both_refs.append(np.clip(fused*(1-blend)+guide*blend,0,255).astype(np.uint8))
        summary={"default_off":[stats(a,b) for a,b in zip(outputs["old"],outputs["off"])],
                 "guard_reference":[stats(a,b) for a,b in zip(references,outputs["guard"])],
                 "fusion_reference":[stats(a,b) for a,b in zip(fusion_refs,outputs["five"])],
                 "both_reference":[stats(a,b) for a,b in zip(both_refs,outputs["both"])],
                 "guard_changes_image":any(not np.array_equal(a,b) for a,b in zip(outputs["off"],outputs["guard"])),
                 "fusion_changes_image":any(not np.array_equal(a,b) for a,b in zip(outputs["off"],outputs["five"]))}
        results[route]=summary
        (args.work/"result.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
        assert all(x["equal"] for x in summary["default_off"]), summary["default_off"]
        assert summary["guard_changes_image"] and summary["fusion_changes_image"]
        for kind in ("guard_reference","fusion_reference","both_reference"):
            assert all(x["max_abs"]<=3 and x["mean_abs"]<.15 for x in summary[kind]), (route,kind,summary[kind])
        print(json.dumps({"route":route,"default_bitexact":True,"reference_maxima":{k:max(x["max_abs"] for x in summary[k]) for k in ("guard_reference","fusion_reference","both_reference")}}),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
