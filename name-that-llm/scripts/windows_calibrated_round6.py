"""One nested calibration experiment. Readiness by default; no retry or legacy run."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

import joblib
import numpy as np
import psutil
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import windows_modernbert_dev as dev
import windows_stylegram_round5 as previous
from probe_modernbert import digest, write_json
from prompt_gaussian_head import fit_head
from stylegram_round5 import make_model, blend_scores
from temperature_round6 import fit_temperature, calibrated_probabilities, gaussian_scores, nested_folds, valid_scores, nll
from windows_prompt_gaussian_round3 import paired_comparison

MODULE = Path(__file__).resolve().parents[1]
CONFIG = MODULE / "config/windows_calibrated_round6_20261025.json"
CONFIG_SHA256 = "7025610475fb946bd6a7cb696661a384c99c8a028c71c0ad2917857a08c1052b"
BASE = MODULE / "config/windows_modernbert_20261023.json"
STYLE_CONFIG = MODULE / "config/windows_stylegram_round5_20261025.json"
GAUSSIAN_CONFIG = MODULE / "config/windows_prompt_gaussian_round3_20261024.json"
OUTPUT = MODULE / "outputs/windows_calibrated_20261025_r6"
CLAIM = MODULE / "outputs/continuation_claims/windows-calibrated-fusion-dev-20261025-r6.json"
SOURCES = tuple(dict.fromkeys((*previous.SOURCES, Path(__file__), Path(__file__).with_name("temperature_round6.py"),
                               CONFIG, STYLE_CONFIG, GAUSSIAN_CONFIG)))


def configuration():
    if digest(CONFIG) != CONFIG_SHA256:
        raise ValueError("Round6 configuration identity differs")
    config = json.loads(CONFIG.read_bytes())
    if config["experiment_id"] != "windows-calibrated-fusion-dev-20261025-r6":
        raise ValueError("Unexpected experiment")
    if digest(STYLE_CONFIG) != config["style_config_sha256"] or digest(GAUSSIAN_CONFIG) != config["gaussian_config_sha256"]:
        raise ValueError("Base recipes changed")
    return config, json.loads(BASE.read_bytes())


def load_inputs(config, base):
    data, baseline, folds, fold_id = previous.load_inputs(config, base)
    feature_path = MODULE / config["fold_reference_output"] / "features.npz"
    if digest(feature_path) != config["features_sha256"]:
        raise ValueError("Frozen feature hash changed")
    features = dev.load_features(feature_path, data, fold_id, config["fold_reference_contract_sha256"], base["feature"]["dimensions"])
    folder = MODULE / config["style_output"]
    claim = json.loads((MODULE / "outputs/continuation_claims/windows-stylegram-fusion-dev-20261025-r5.json").read_bytes())
    prior_contract = json.loads((folder / "contract.json").read_bytes())
    if (claim["state"] != "completed" or claim["contract_sha256"] != config["style_contract_sha256"]
            or dev.object_hash(prior_contract) != config["style_contract_sha256"]
            or digest(folder / "summary.json") != claim["summary_sha256"]
            or digest(folder / "supervisor.json") != claim["supervisor_sha256"]
            or digest(folder / "style_oof.npz") != config["style_oof_sha256"]):
        raise ValueError("Completed Round5 artifact identity differs")
    with np.load(folder / "style_oof.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"probabilities","row_index","ids","groups","truth","labels","fold_id","contract_sha256"}:
            raise ValueError("Style OOF schema differs")
        for key in ("row_index","ids","groups","truth","labels"):
            np.testing.assert_array_equal(saved[key], data[key])
        np.testing.assert_array_equal(saved["fold_id"], fold_id)
        if str(saved["contract_sha256"]) != config["style_contract_sha256"]:
            raise ValueError("Style OOF contract differs")
        style = dev.probabilities(saved["probabilities"],len(fold_id),18)
    model_hashes = {}
    for directory, prefix, suffix in ((folder,"style","joblib"),(MODULE/config["baseline_output"],"head","npz")):
        summary = json.loads((directory / "summary.json").read_bytes())
        for i, record in enumerate(summary["folds"]):
            path = directory / f"{prefix}_fold{i}.{suffix}"
            if record["fold"] != i or digest(path) != record["model_sha256"]:
                raise ValueError("Original outer-fold model changed")
            model_hashes[str(path.relative_to(MODULE))] = record["model_sha256"]
    if len(model_hashes) != 10:
        raise ValueError("Missing outer models")
    return data, features, baseline, style, folds, fold_id, model_hashes


def contract(config, base, data, folds, fold_id, model_hashes):
    inner = [nested_folds(data,train,valid,config["calibration"]["inner_folds"])[1].tolist() for train,valid in folds]
    return {"config":config,"base_config":base,"sources":{str(p.relative_to(MODULE)):digest(p) for p in SOURCES},
            "outer_models":model_hashes,"row_identity_sha256":dev.object_hash({k:data[k].tolist() for k in ("row_index","ids","groups","truth","labels")}),
            "text_sha256":dev.object_hash(data["texts"].tolist()),"fold_sha256":dev.object_hash(fold_id.tolist()),
            "inner_fold_sha256":dev.object_hash(inner),"python":sys.version,"platform":sys.platform,
            "package_versions":{name:importlib.metadata.version(name) for name in config["runtime"]}}


def readiness(config, base):
    blockers = []
    if CLAIM.exists() or OUTPUT.exists():
        blockers.append("Fixed Round6 claim/output exists; no retry/resume")
    if sys.platform != "win32":
        blockers.append("Native Windows required")
    for name, expected in config["runtime"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            blockers.append(f"Runtime {name} differs: {actual}")
    if blockers:
        return {"state":"blocked","blockers":blockers}, None
    try:
        data, _, _, _, folds, fold_id, model_hashes = load_inputs(config,base)
        value = contract(config,base,data,folds,fold_id,model_hashes)
    except (OSError,ValueError,AssertionError) as error:
        return {"state":"blocked","blockers":[f"{type(error).__name__}: {error}"]},None
    return {"state":"ready","rows":len(fold_id),"contract_sha256":dev.object_hash(value)},value


def consume(claim, kind, config):
    if claim["attempted_fits"] + claim["attempted_temperature_fits"] >= config["budget"]["max_total_fit_attempts"]:
        raise RuntimeError("Total fit budget exhausted")
    limit = "max_classifier_fit_attempts" if kind == "attempted_fits" else "max_temperature_fit_attempts"
    dev.consume(claim,kind,config["budget"][limit],CLAIM)


def inner_scores(data, features, outer_train, outer_valid, config, claim, folder):
    folds, identity = nested_folds(data,outer_train,outer_valid,config["calibration"]["inner_folds"])
    style = np.full((len(outer_train),18),np.nan)
    gaussian = np.full_like(style,np.nan)
    style_settings = json.loads(STYLE_CONFIG.read_bytes())["method"]
    gaussian_settings = json.loads(GAUSSIAN_CONFIG.read_bytes())["method"]
    records = []
    for number,(train,valid,local_valid) in enumerate(folds):
        started = time.perf_counter()
        consume(claim,"attempted_fits",config)
        model = make_model(style_settings)
        with warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            model.fit(data["texts"][train].tolist(),data["truth"][train])
        claim["completed_fits"] += 1
        write_json(CLAIM,claim,exclusive=False)
        if not np.array_equal(model.classes_,np.arange(18)):
            raise ValueError("SVC class order differs")
        style[local_valid] = valid_scores(model.decision_function(data["texts"][valid].tolist()))
        iterations = int(model[-1].n_iter_)
        del model
        consume(claim,"attempted_fits",config)
        head, info = fit_head(features[train],data["truth"][train],data["groups"][train],gaussian_settings)
        claim["completed_fits"] += 1
        write_json(CLAIM,claim,exclusive=False)
        if info["dimensions"] != gaussian_settings["dimensions"]:
            raise ValueError("Inner Gaussian dimensionality changed")
        gaussian[local_valid] = gaussian_scores(features[valid],head)
        del head
        records.append({"inner_fold":number,"train_rows":len(train),"valid_rows":len(valid),
                        "train_groups":len(set(data["groups"][train])),"valid_groups":len(set(data["groups"][valid])),
                        "train_row_identity_sha256":dev.object_hash(data["row_index"][train].tolist()),
                        "valid_row_identity_sha256":dev.object_hash(data["row_index"][valid].tolist()),
                        "svc_iterations":iterations,"gaussian":info,"seconds":time.perf_counter()-started})
        print(f"{folder.name} inner={number} two_fits_completed",flush=True)
    valid_scores(style); valid_scores(gaussian)
    with (folder / "inner_scores.npz").open("xb") as handle:
        np.savez_compressed(handle,style_scores=style,gaussian_scores=gaussian,
                            **{k:data[k][outer_train] for k in ("row_index","ids","groups","truth")},
                            inner_fold_id=identity,outer_train_positions=outer_train,outer_valid_positions=outer_valid,
                            labels=data["labels"],contract_sha256=claim["contract_sha256"])
    return style,gaussian,records


def execute(config, base):
    claim = json.loads(CLAIM.read_bytes())
    dev.require_live_supervisor(claim)
    previous.start_worker_guard(claim,config["budget"])
    data,features,baseline,style_control,folds,fold_id,model_hashes = load_inputs(config,base)
    if dev.object_hash(contract(config,base,data,folds,fold_id,model_hashes)) != claim["contract_sha256"]:
        raise ValueError("Execution contract changed")
    oof = np.full_like(baseline,np.nan)
    calibrated_style, calibrated_gaussian = np.full_like(oof,np.nan),np.full_like(oof,np.nan)
    records = []
    with threadpool_limits(limits=config["budget"]["threads"]):
        for number,(train,valid) in enumerate(folds):
            folder = OUTPUT / f"fold{number}"
            folder.mkdir(exist_ok=False)
            print(f"outer={number} started",flush=True)
            a,b,inner_records = inner_scores(data,features,train,valid,config,claim,folder)
            calibration = {}
            for family,scores in (("style",a),("gaussian",b)):
                consume(claim,"attempted_temperature_fits",config)
                calibration[family] = fit_temperature(scores,data["truth"][train],config["calibration"])
                claim["completed_temperature_fits"] += 1
                write_json(CLAIM,claim,exclusive=False)
            write_json(folder / "calibration.json",calibration)
            model = joblib.load(MODULE/config["style_output"]/f"style_fold{number}.joblib")
            if not np.array_equal(model.classes_,np.arange(18)):
                raise ValueError("Outer SVC classes differ")
            style_raw = valid_scores(model.decision_function(data["texts"][valid].tolist()))
            with np.load(MODULE/config["baseline_output"]/f"head_fold{number}.npz",allow_pickle=False) as saved:
                head = {key:saved[key].copy() for key in saved.files}
            gaussian_raw = gaussian_scores(features[valid],head)
            np.testing.assert_allclose(softmax(style_raw,axis=1),style_control[valid],atol=1e-12,rtol=0)
            np.testing.assert_allclose(softmax(gaussian_raw,axis=1),baseline[valid],atol=1e-12,rtol=0)
            pa,pb = calibrated_probabilities(style_raw,calibration["style"]),calibrated_probabilities(gaussian_raw,calibration["gaussian"])
            p = blend_scores(pa,pb,config["style_weight"])
            reverse = blend_scores(calibrated_probabilities(style_raw[::-1],calibration["style"]),calibrated_probabilities(gaussian_raw[::-1],calibration["gaussian"]),config["style_weight"])[::-1]
            np.testing.assert_array_equal(p,reverse)
            calibrated_style[valid],calibrated_gaussian[valid],oof[valid] = pa,pb,p
            with (folder / "outer_scores.npz").open("xb") as handle:
                np.savez_compressed(handle,style_scores=style_raw,gaussian_scores=gaussian_raw,
                                    **{key:data[key][valid] for key in ("row_index","ids","groups","truth")},
                                    labels=data["labels"],fold_id=fold_id[valid],contract_sha256=claim["contract_sha256"])
            records.append({"fold":number,"inner_records":inner_records,"calibration":calibration,
                            "calibration_sha256":digest(folder/"calibration.json"),"inner_scores_sha256":digest(folder/"inner_scores.npz"),
                            "outer_scores_sha256":digest(folder/"outer_scores.npz"),"inference_invariance_max_abs_error":0.0,
                            "outer_nll":{"style_before":nll(style_raw,data["truth"][valid]),"style_after":nll(style_raw,data["truth"][valid],calibration["style"]["inverse_temperature"]),
                                         "gaussian_before":nll(gaussian_raw,data["truth"][valid]),"gaussian_after":nll(gaussian_raw,data["truth"][valid],calibration["gaussian"]["inverse_temperature"])}})
            print(f"outer={number} completed temperatures={calibration['style']['temperature']:.6f},{calibration['gaussian']['temperature']:.6f}",flush=True)
            del model,head
    for filename,array in (("oof.npz",oof),("calibrated_style_oof.npz",calibrated_style),("calibrated_gaussian_oof.npz",calibrated_gaussian)):
        dev.probabilities(array,len(fold_id),18)
        dev.save_array(OUTPUT/filename,"probabilities",array,data,fold_id,claim["contract_sha256"])
    metrics = dev.metrics(oof,data,fold_id,config["evaluation"])
    comparison = paired_comparison(oof,baseline,data,fold_id,config["evaluation"])
    control = paired_comparison(oof,style_control,data,fold_id,config["evaluation"])
    summary = {"scope":config["scope"],"contract_sha256":claim["contract_sha256"],"metrics":metrics,
               "paired_round3_comparison":comparison,"paired_style_control_comparison":control,
               "candidate_gate":previous.metric_gate(metrics,comparison,control,config),"folds":records,
               "oof_sha256":digest(OUTPUT/"oof.npz"),"attempted_fits":claim["attempted_fits"],"completed_fits":claim["completed_fits"],
               "attempted_temperature_fits":claim["attempted_temperature_fits"],"completed_temperature_fits":claim["completed_temperature_fits"],
               "new_encoder_forward_calls":0,"conditional_final_fits":0,"old_holdout_evaluations":0,"new_kaggle_submissions":0,
               "limitations":"Repeated adaptive original-development evaluation. Inner grouped calibration prevents outer-fold leakage but does not create an independent holdout."}
    dev.verify_preserved_claims(config)
    write_json(OUTPUT/"summary.json",summary)
    claim.update(state="worker_completed",summary_sha256=digest(OUTPUT/"summary.json"))
    write_json(CLAIM,claim,exclusive=False)


def supervise(config,value):
    claim = {"state":"started","experiment_id":config["experiment_id"],"supervisor_pid":os.getpid(),
             "supervisor_create_time":psutil.Process().create_time(),"started_unix_seconds":time.time(),"contract_sha256":dev.object_hash(value),
             "attempted_forward_calls":0,"completed_forward_calls":0,"attempted_fits":0,"completed_fits":0,
             "attempted_temperature_fits":0,"completed_temperature_fits":0,"old_holdout_evaluations":0,"full_data_final_fits":0}
    write_json(CLAIM,claim)
    process,phase = None,"pre_spawn"
    started,peak = time.perf_counter(),0
    try:
        OUTPUT.mkdir(exist_ok=False,parents=True)
        write_json(OUTPUT/"contract.json",value)
        for source in SOURCES:
            target = OUTPUT/"source_snapshot"/source.relative_to(MODULE)
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("xb") as handle:
                handle.write(source.read_bytes())
        env = {**os.environ,**{key:str(config["budget"]["threads"]) for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}}
        with (OUTPUT/"stdout.log").open("xb") as stdout,(OUTPUT/"stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--execute"],env=env,stdout=stdout,stderr=stderr)
            phase = "worker"
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                if time.perf_counter()-started > config["budget"]["development_timeout_seconds"]:
                    raise TimeoutError("Round6 wall-clock budget exceeded")
                try:
                    rss = psutil.Process().memory_info().rss + monitored.memory_info().rss + sum(p.memory_info().rss for p in monitored.children(recursive=True))
                    peak = max(peak,rss)
                    if rss > config["budget"]["max_host_rss_gib"]*2**30:
                        raise MemoryError("Round6 RSS budget exceeded")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.2)
        phase = "reconcile"
        claim = json.loads(CLAIM.read_bytes())
        if (process.returncode != 0 or claim["state"] != "worker_completed" or claim["attempted_fits"] != 30 or claim["completed_fits"] != 30
                or claim["attempted_temperature_fits"] != 10 or claim["completed_temperature_fits"] != 10
                or claim["attempted_forward_calls"] != 0 or claim["completed_forward_calls"] != 0
                or digest(OUTPUT/"summary.json") != claim["summary_sha256"]):
            raise RuntimeError("Round6 worker failed or incomplete")
        dev.verify_preserved_claims(config)
        if any(digest(MODULE/path) != expected for path,expected in {**value["sources"],**value["outer_models"]}.items()):
            raise ValueError("Execution source/model identity changed")
        summary = json.loads((OUTPUT/"summary.json").read_bytes())
        gate = previous.metric_gate(summary["metrics"],summary["paired_round3_comparison"],summary["paired_style_control_comparison"],config)
        receipt = {"state":"completed","exit_code":0,"wall_seconds":time.perf_counter()-started,"sampled_peak_process_tree_rss_bytes":peak,
                   "summary_sha256":claim["summary_sha256"],"candidate_development_gate_passed":gate["metric_gate_passed"],"deployment_validation_complete":False}
        write_json(OUTPUT/"supervisor.json",receipt)
        claim.update(state="completed",supervisor_sha256=digest(OUTPUT/"supervisor.json"))
        write_json(CLAIM,claim,exclusive=False)
        return receipt
    except BaseException as error:
        termination = None
        try:
            if process is not None:
                termination = previous.terminate_tree(process)
        except BaseException as cleanup_error:
            termination = {"exception":type(cleanup_error).__name__}
        finally:
            claim = json.loads(CLAIM.read_bytes())
            claim.update(state="failed",exception_type=type(error).__name__,exception_message=str(error),phase=phase,
                         termination=termination,wall_seconds=time.perf_counter()-started,sampled_peak_process_tree_rss_bytes=peak)
            write_json(CLAIM,claim,exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run",action="store_true")
    mode.add_argument("--execute",action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args()
    config,base = configuration()
    if args.execute:
        execute(config,base)
        return 0
    state,value = readiness(config,base)
    if not args.run or state["state"] != "ready":
        print(json.dumps(state))
        return 0 if state["state"] == "ready" else 2
    print(json.dumps(supervise(config,value)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
