# -*- coding: utf-8 -*-
"""Evaluate exported two-head models on the 30 FPS day-22/day-24 test set."""
import csv, json, runpy, tempfile, zipfile
from pathlib import Path
CONFIG={"artifact_root":"/kaggle/input/datasets/anamariabarrosnonato/ss2ere/sibgrapi","dataset_dir":"auto","output_dir":"/kaggle/working/sibgrapi_route_accuracy","trainer_script":"/kaggle/input/datasets/anamariabarrosnonato/scripthelper/train_all_two_head_models_kaggle.py","test_days":[22,24],"fps":30.0,"batch_size":64,"num_workers":2}
ROUTE_CLASSES=["forward","right","left","stop"]
SCRIPT_DIR=Path(globals().get("__file__", Path.cwd())).resolve().parent if "__file__" in globals() else Path.cwd()

def route_index(obstacle,deviation,obstacle_classes,deviation_classes):
    if deviation==deviation_classes.index("left"): return 2
    if deviation==deviation_classes.index("right"): return 1
    if obstacle==obstacle_classes.index("obstacle"): return 3
    return 0

def metrics(truth,predicted):
    matrix=[[0]*4 for _ in range(4)]
    for actual,estimate in zip(truth,predicted): matrix[actual][estimate]+=1
    total=sum(map(sum,matrix)); accuracy=sum(matrix[i][i] for i in range(4))/max(1,total); f1=[]; evaluated=[]
    for i in range(4):
        support=sum(matrix[i])
        if support == 0: continue
        tp=matrix[i][i]; fp=sum(matrix[r][i] for r in range(4))-tp; fn=support-tp
        precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn); f1.append(2*precision*recall/max(1e-12,precision+recall)); evaluated.append(ROUTE_CLASSES[i])
    return {"frames":total,"route_accuracy":round(accuracy,6),"macro_f1":round(sum(f1)/max(1,len(f1)),6),"macro_f1_classes":evaluated,"route_classes":ROUTE_CLASSES,"confusion_matrix":matrix,"true_route_counts":dict(zip(ROUTE_CLASSES,map(sum,matrix)))}

def checkpoint_files(root):
    candidates=list(root.rglob("*.pt"))+list(root.rglob("*.zip"))
    candidates += [path for path in root.rglob("model") if path.is_file()]
    candidates += [path for path in root.rglob("model") if path.is_dir() and (path / "data.pkl").is_file()]
    files=[]; seen=set()
    for path in sorted(candidates):
        if path in seen: continue
        seen.add(path)
        run_dir=next((parent for parent in path.parents if parent.name.lower().startswith("run")), path.parent)
        files.append((run_dir.name,path))
    if not files: raise FileNotFoundError(f"No model checkpoints found below {root}")
    return files

def load_checkpoint(torch, path, device):
    if path.is_dir():
        with tempfile.NamedTemporaryFile(suffix=".zip") as stream:
            with zipfile.ZipFile(stream.name, "w", compression=zipfile.ZIP_STORED) as archive:
                for child in path.rglob("*"):
                    if child.is_file(): archive.write(child, "model/" + child.relative_to(path).as_posix())
            return torch.load(stream.name, map_location=device, weights_only=False)
    return torch.load(path, map_location=device, weights_only=False)

def main():
    root=Path(CONFIG["artifact_root"]); trainer=Path(CONFIG["trainer_script"]) if CONFIG["trainer_script"] else SCRIPT_DIR/"train_all_two_head_models_kaggle.py"
    if not trainer.is_file():
        fallback=Path("/kaggle/input/datasets/anamariabarrosnonato/scripthelper/train_all_two_head_models_kaggle.py")
        if fallback.is_file(): trainer=fallback
    base=runpy.run_path(str(trainer),run_name="_route_base_trainer"); torch=base["torch"]
    base["CONFIG"].update({"batch_size":CONFIG["batch_size"],"num_workers":CONFIG["num_workers"],"pretrained":False,"test_days":CONFIG["test_days"],"require_july24_data":False,"require_auto_trim_data":False})
    dataset_root=base["find_dataset"](CONFIG["dataset_dir"])
    cache_dir=Path(CONFIG["output_dir"])/"dataset_cache"; rows_cache=cache_dir/"rows.json"
    if rows_cache.is_file():
        rows=json.loads(rows_cache.read_text(encoding="utf-8"))
        print(f"Loaded generated dataset rows from {rows_cache}")
    else:
        rows=base["load_allframe_rows"](dataset_root,cache_dir)
        cache_dir.mkdir(parents=True,exist_ok=True)
        rows_cache.write_text(json.dumps(rows),encoding="utf-8")
        print(f"Saved generated dataset rows to {rows_cache}")
    _,_,test_rows=base["grouped_split"](rows,base["CONFIG"]["val_fraction"],base["CONFIG"]["seed"]); test_rows=[r for r in test_rows if base["collection_day"](r["clip"]) in CONFIG["test_days"]]
    fps_rows=[r for r in test_rows if "fps" in r]
    if fps_rows: test_rows=[r for r in fps_rows if float(r["fps"])==CONFIG["fps"]]
    if not test_rows: raise RuntimeError("The day-22/day-24 test set has no rows at the requested FPS.")
    transform=base["transforms"].Compose([base["transforms"].Resize((224,224),base["transforms"].InterpolationMode.BICUBIC),base["transforms"].ToTensor(),base["transforms"].Normalize(base["MEAN"],base["STD"])])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); report={"dataset":str(dataset_root),"artifact_root":str(root),"test_days":CONFIG["test_days"],"fps":CONFIG["fps"],"test_frames":len(test_rows),"test_clips":sorted({r["clip"] for r in test_rows}),"route_definition":{"forward":"free_path + none","right":"right, with or without obstacle","left":"left, with or without obstacle","stop":"obstacle + none"},"models":{}}; csv_rows=[]
    for run_name,checkpoint_path in checkpoint_files(root):
        checkpoint=load_checkpoint(torch,checkpoint_path,device); architecture=checkpoint["architecture"]
        if architecture not in {"mobilenetv1","mobilenetv4","resnet50","efficientnet_b0"}: continue
        base["CONFIG"]["image_size"]=checkpoint.get("image_size",224); model=base["make_model"](architecture,device); model.load_state_dict(checkpoint["state_dict"]); loader=base["loader"](test_rows,transform,device,False,False,base["CONFIG"]["seed"]); truth=[]; predicted=[]; model.eval()
        with torch.inference_mode():
            for images,obstacle,deviation,_,_ in loader:
                ol,dl=model(images.to(device))
                for ao,ad,po,pd in zip(obstacle.tolist(),deviation.tolist(),ol.argmax(1).cpu().tolist(),dl.argmax(1).cpu().tolist()): truth.append(route_index(ao,ad,checkpoint["obstacle_classes"],checkpoint["deviation_classes"])); predicted.append(route_index(po,pd,checkpoint["obstacle_classes"],checkpoint["deviation_classes"]))
        result={"run":run_name,"checkpoint":str(checkpoint_path),"architecture":architecture,**metrics(truth,predicted)}; report["models"][f"{run_name}/{architecture}"]=result; csv_rows.append(result); print(f"{run_name}/{architecture}: accuracy={result['route_accuracy']:.4%}, macro-f1={result['macro_f1']:.4%}, frames={result['frames']}")
    output=Path(CONFIG["output_dir"]); output.mkdir(parents=True,exist_ok=True); (output/"route_accuracy_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (output/"route_accuracy_report.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=["run","architecture","frames","route_accuracy","macro_f1"]); writer.writeheader(); writer.writerows({k:r[k] for k in writer.fieldnames} for r in csv_rows)
    print(f"Wrote {output/'route_accuracy_report.json'}"); print(f"Wrote {output/'route_accuracy_report.csv'}")

if __name__=="__main__": main()
