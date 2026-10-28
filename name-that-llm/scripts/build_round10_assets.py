"""Allowlisted private R10 frozen artifact and inference source export."""
import json
from pathlib import Path
import shutil
from inference_round7 import digest, write_json
from verify_round10_results import check

MODULE=Path(__file__).resolve().parents[1]
FINAL=MODULE/"outputs/windows_topic_word_final_20261028_r10"
ASSETS=MODULE/"outputs/kaggle_r10_assets/ntl-r10-assets"


def main():
    verified=check()
    cp=MODULE/"outputs/continuation_claims/windows-topic-word-final-20261028-r10.json"
    claim=json.loads(cp.read_bytes())
    summary=json.loads((FINAL/"summary.json").read_bytes())
    receipt=json.loads((FINAL/"supervisor.json").read_bytes())
    contract=json.loads((FINAL/"contract.json").read_bytes())
    if (claim["state"]!=receipt["state"] or claim["state"]!="completed" or receipt["exit_code"]!=0
            or digest(FINAL/"summary.json")!=claim["summary_sha256"] or digest(FINAL/"supervisor.json")!=claim["supervisor_sha256"]
            or claim["attempted_fits"]!=1 or claim["completed_fits"]!=1 or claim["attempted_vectorizer_fits"]!=0
            or summary["margin_correction"]!=verified["margin_correction"]):
        raise ValueError("Final evidence incomplete")
    for name,key in [("topic_svc.joblib","model_sha256"),("synthetic_scores.npz","synthetic_scores_sha256")]:
        if digest(FINAL/name)!=summary[key]:
            raise ValueError("Export changed")
    source_names=("stylegram_round5.py","weighted_style_round7.py","topic_word_round9.py","margin_center_round10.py")
    for name in source_names:
        if digest(MODULE/"scripts"/name)!=contract["sources"]["scripts/"+name]:
            raise ValueError("Training source changed")
    ASSETS.mkdir(parents=True,exist_ok=False)
    for name in ("topic_svc.joblib","synthetic_scores.npz"):
        shutil.copyfile(FINAL/name,ASSETS/name)
    for name in (*source_names,"inference_round7.py","inference_round10.py"):
        shutil.copyfile(MODULE/"scripts"/name,ASSETS/name)
    shutil.copyfile(MODULE/"config/labels.json",ASSETS/"labels.json")
    shutil.copyfile(MODULE/"config/round7_external_sources_20261026.json",ASSETS/"external_source_manifest.json")
    (ASSETS/"README.txt").write_text(
        "Private frozen R10 topic-word four-view SVC. Official/external weights 4/1.\n"
        "Trained on original 2159 development rows and fixed 38724 public external rows; protected 30 prompts excluded.\n"
        "Whole test batch text-only grouping; partial word centering and fixed margin shrinkage. No supervised test-time fitting.\n"
        "Development BA 0.47740429505135384; R9 0.4542794895736073; R10 gate passed, R9 historic gate remains failed.\n"
        "No raw competition passages, OOF, holdout, encoder or credentials. Artificial texts test compatibility only.\n"
        "External source/version attribution is recorded in external_source_manifest.json. Hailey/Michael sources excluded.\n"
        "Grouping reference https://www.kaggle.com/code/zdx13143/ntl-e651?scriptVersionId=362577633 ; not the full ensemble.\n",encoding="utf-8")
    files={p.name:{"bytes":p.stat().st_size,"sha256":digest(p)} for p in sorted(ASSETS.iterdir())}
    manifest={"candidate":claim["experiment_id"],"official_weight":4,"word_alpha":.5,
              "grouping":contract["config"]["method"]["grouping"],"margin_correction":summary["margin_correction"],
              "development_gate_passed":verified["gate_passed"],"final_contract_sha256":claim["contract_sha256"],
              "final_claim_sha256":digest(cp),"final_summary_sha256":digest(FINAL/"summary.json"),"files":files}
    write_json(ASSETS/"manifest.json",manifest,True)
    print(json.dumps({"state":"assets_built","manifest_sha256":digest(ASSETS/"manifest.json"),"files":len(files)}))


if __name__=="__main__":main()
