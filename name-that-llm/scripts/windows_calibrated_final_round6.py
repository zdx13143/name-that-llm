"""Conditional one-shot final 120-prompt fit; never evaluates protected holdout."""
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
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import windows_calibrated_round6 as devrun
from verify_round6_results import check
from probe_modernbert import digest,write_json
from stylegram_round5 import make_model
from prompt_gaussian_head import fit_head
from temperature_round6 import nested_folds,valid_scores,gaussian_scores,fit_temperature,calibrated_probabilities

MODULE = devrun.MODULE
CLAIM = MODULE/"outputs/continuation_claims/windows-calibrated-fusion-final-20261025-r6.json"
OUTPUT = MODULE/"outputs/windows_calibrated_final_20261025_r6"
SOURCES = tuple(dict.fromkeys((*devrun.SOURCES,Path(__file__),Path(__file__).with_name("verify_round6_results.py"))))


def admission():
    if CLAIM.exists() or OUTPUT.exists():
        raise ValueError("Final claim/output already exists; no retry/resume")
    config,base = devrun.configuration()
    verified = check()
    if verified["state"] != "verified" or verified["development_gate_passed"] is not True:
        raise ValueError("Development gate/reconciliation failed")
    data,features,_,_,_,fold_id,_ = devrun.load_inputs(config,base)
    if len(data["truth"]) != 2159 or len(set(data["groups"])) != 120:
        raise ValueError("Final development boundary changed")
    for key,expected in config["runtime"].items():
        if importlib.metadata.version(key) != expected:
            raise ValueError("Final runtime differs")
    value = {"development_verification":verified,"development_claim_sha256":digest(devrun.CLAIM),
             "config":config,"row_identity_sha256":devrun.dev.object_hash({k:data[k].tolist() for k in ("row_index","ids","groups","truth","labels")}),
             "sources":{str(p.relative_to(MODULE)):digest(p) for p in SOURCES}}
    return config,base,data,features,value


def consume(claim,key,budget):
    if claim["attempted_fits"]+claim["attempted_temperature_fits"] >= budget["max_total_fit_attempts"]:
        raise RuntimeError("Final total fit budget exhausted")
    limit = budget["max_classifier_fit_attempts"] if key == "attempted_fits" else budget["max_temperature_fit_attempts"]
    devrun.dev.consume(claim,key,limit,CLAIM)


def execute():
    claim = json.loads(CLAIM.read_bytes())
    devrun.dev.require_live_supervisor(claim)
    config,base = devrun.configuration()
    budget = config["conditional_deployment_budget"]
    devrun.previous.start_worker_guard(claim,budget)
    verified = check()
    contract = json.loads((OUTPUT/"contract.json").read_bytes())
    if (not verified["development_gate_passed"] or verified != contract["development_verification"]
            or digest(devrun.CLAIM) != contract["development_claim_sha256"]
            or any(digest(MODULE/p)!=sha for p,sha in contract["sources"].items())
            or devrun.dev.object_hash(contract) != claim["contract_sha256"]):
        raise ValueError("Final admission changed")
    data,features,_,_,_,_,_ = devrun.load_inputs(config,base)
    style_settings = json.loads(devrun.STYLE_CONFIG.read_bytes())["method"]
    gaussian_settings = json.loads(devrun.GAUSSIAN_CONFIG.read_bytes())["method"]
    folds,inner_id = nested_folds(data,np.arange(2159),np.array([],dtype=int),3)
    a,b = np.full((2159,18),np.nan),np.full((2159,18),np.nan)
    records = []
    def fit_style(indices):
        consume(claim,"attempted_fits",budget)
        model = make_model(style_settings)
        with warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            model.fit(data["texts"][indices].tolist(),data["truth"][indices])
        claim["completed_fits"] += 1
        write_json(CLAIM,claim,exclusive=False)
        if not np.array_equal(model.classes_,np.arange(18)):
            raise ValueError("Final SVC class order differs")
        return model
    def fit_gaussian(indices):
        consume(claim,"attempted_fits",budget)
        head,info = fit_head(features[indices],data["truth"][indices],data["groups"][indices],gaussian_settings)
        claim["completed_fits"] += 1
        write_json(CLAIM,claim,exclusive=False)
        if info["dimensions"] != 128:
            raise ValueError("Final Gaussian dimensions differ")
        return head,info
    with threadpool_limits(limits=budget["threads"]):
        for number,(tr,va,local) in enumerate(folds):
            model = fit_style(tr)
            a[local] = valid_scores(model.decision_function(data["texts"][va].tolist()))
            head,info = fit_gaussian(tr)
            b[local] = gaussian_scores(features[va],head)
            records.append({"fold":number,"train_rows":len(tr),"valid_rows":len(va),"gaussian":info})
            print(f"final_inner={number} completed",flush=True)
            del model,head
        with (OUTPUT/"calibration_oof.npz").open("xb") as handle:
            np.savez_compressed(handle,style_scores=a,gaussian_scores=b,inner_fold_id=inner_id,
                                **{k:data[k] for k in ("row_index","ids","groups","truth","labels")})
        calibration = {}
        for family,scores in (("style",a),("gaussian",b)):
            consume(claim,"attempted_temperature_fits",budget)
            calibration[family] = fit_temperature(scores,data["truth"],config["calibration"])
            claim["completed_temperature_fits"] += 1
            write_json(CLAIM,claim,exclusive=False)
        write_json(OUTPUT/"calibration.json",calibration)
        indices = np.arange(2159)
        model = fit_style(indices)
        head,info = fit_gaussian(indices)
        with (OUTPUT/"style.joblib").open("xb") as handle:
            joblib.dump(model,handle,compress=3)
        with (OUTPUT/"gaussian.npz").open("xb") as handle:
            np.savez_compressed(handle,**head)
        restored = joblib.load(OUTPUT/"style.joblib")
        original = model.decision_function(data["texts"][:3].tolist())
        np.testing.assert_array_equal(original,restored.decision_function(data["texts"][:3].tolist()))
        with np.load(OUTPUT/"gaussian.npz",allow_pickle=False) as handle:
            restored_head = {k:handle[k].copy() for k in handle.files}
        np.testing.assert_array_equal(gaussian_scores(features[:3],head),gaussian_scores(features[:3],restored_head))
    report = {"state":"models_exported","training_rows":2159,"training_groups":120,"classifier_fits":8,"temperature_fits":2,
              "new_encoder_forward_calls":0,"old_holdout_evaluations":0,"full_2699_row_fits":0,"inner_folds":records,
              "calibration":calibration,"final_gaussian":info,"model_hashes":{file:digest(OUTPUT/file) for file in ("style.joblib","gaussian.npz","calibration.json")},
              "export_max_abs_error":0.0,"linux_offline_validation_complete":False,"kaggle_submissions":0}
    devrun.dev.verify_preserved_claims(config)
    write_json(OUTPUT/"summary.json",report)
    claim.update(state="worker_completed",summary_sha256=digest(OUTPUT/"summary.json"))
    write_json(CLAIM,claim,exclusive=False)


def supervise(config,value):
    budget = config["conditional_deployment_budget"]
    claim = {"state":"started","supervisor_pid":os.getpid(),"supervisor_create_time":psutil.Process().create_time(),
             "experiment_id":"windows-calibrated-fusion-final-20261025-r6","contract_sha256":devrun.dev.object_hash(value),
             "started_unix_seconds":time.time(),"attempted_fits":0,"completed_fits":0,"attempted_temperature_fits":0,"completed_temperature_fits":0,
             "attempted_forward_calls":0,"completed_forward_calls":0,"old_holdout_evaluations":0,"full_data_final_fits":0}
    write_json(CLAIM,claim)
    process = None
    started,peak = time.perf_counter(),0
    try:
        OUTPUT.mkdir(exist_ok=False,parents=True)
        write_json(OUTPUT/"contract.json",value)
        for source in SOURCES:
            target = OUTPUT/"source_snapshot"/source.relative_to(MODULE)
            target.parent.mkdir(exist_ok=True,parents=True)
            with target.open("xb") as handle: handle.write(source.read_bytes())
        env = {**os.environ,**{key:"2" for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}}
        with (OUTPUT/"stdout.log").open("xb") as stdout,(OUTPUT/"stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--execute"],env=env,stdout=stdout,stderr=stderr)
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                if time.perf_counter()-started > budget["development_timeout_seconds"]:
                    raise TimeoutError("Final wall-clock budget exceeded")
                try:
                    rss = psutil.Process().memory_info().rss+monitored.memory_info().rss+sum(p.memory_info().rss for p in monitored.children(recursive=True))
                    peak = max(peak,rss)
                    if rss>budget["max_host_rss_gib"]*2**30: raise MemoryError("Final RSS budget exceeded")
                except psutil.NoSuchProcess: pass
                time.sleep(.2)
        claim = json.loads(CLAIM.read_bytes())
        if (process.returncode!=0 or claim["state"]!="worker_completed" or claim["attempted_fits"]!=8 or claim["completed_fits"]!=8
                or claim["attempted_temperature_fits"]!=2 or claim["completed_temperature_fits"]!=2
                or digest(OUTPUT/"summary.json")!=claim["summary_sha256"]):
            raise RuntimeError("Final worker failed or incomplete")
        if any(digest(MODULE/p)!=sha for p,sha in value["sources"].items()): raise ValueError("Final sources changed")
        if digest(devrun.CLAIM)!=value["development_claim_sha256"]: raise ValueError("Development claim changed")
        devrun.dev.verify_preserved_claims(config)
        receipt = {"state":"completed","exit_code":0,"wall_seconds":time.perf_counter()-started,"sampled_peak_process_tree_rss_bytes":peak,
                   "summary_sha256":claim["summary_sha256"],"linux_offline_validation_complete":False}
        write_json(OUTPUT/"supervisor.json",receipt)
        claim.update(state="completed",supervisor_sha256=digest(OUTPUT/"supervisor.json"))
        write_json(CLAIM,claim,exclusive=False)
        return receipt
    except BaseException as error:
        termination = devrun.previous.terminate_tree(process) if process else None
        claim = json.loads(CLAIM.read_bytes())
        claim.update(state="failed",error_type=type(error).__name__,error=str(error),termination=termination)
        write_json(CLAIM,claim,exclusive=False)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--run",action="store_true")
    mode.add_argument("--execute",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.execute:
        execute(); return 0
    try:
        config,_,_,_,value=admission()
    except (OSError,ValueError,AssertionError,importlib.metadata.PackageNotFoundError) as error:
        print(json.dumps({"state":"blocked","reason":str(error)})); return 2
    if args.run: print(json.dumps(supervise(config,value)))
    else: print(json.dumps({"state":"ready","contract_sha256":devrun.dev.object_hash(value)}))
    return 0


if __name__=="__main__": raise SystemExit(main())
