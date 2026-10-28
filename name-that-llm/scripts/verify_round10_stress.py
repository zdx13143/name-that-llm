"""One bounded 4096-row artificial grouping check; no classifier or real data."""
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
from inference_round10 import digest, verify_assets, write_json, block_network, verify_runtime


def main():
    if sys.platform != "linux":
        raise RuntimeError("Use Linux for deployment resource verification")
    module=Path(__file__).resolve().parents[1]
    assets=module/"outputs/kaggle_r10_assets/ntl-r10-assets"
    package=module/"outputs/kaggle_r10_package"
    cp=package/"grouping_stress_claim.json"
    target=package/"grouping_stress_verification.json"
    if cp.exists() or target.exists():
        raise FileExistsError("Preserve prior stress attempt")
    manifest_sha=digest(assets/"manifest.json")
    manifest=verify_assets(assets,manifest_sha)
    verify_runtime();block_network()
    # Address-space and time limits are additional guards for this synthetic check.
    resource.setrlimit(resource.RLIMIT_AS,(12*2**30,12*2**30))
    claim={"state":"started","rows":4096,"attempted_groupings":1,"completed_groupings":0,
           "manifest_sha256":manifest_sha,"classifier_fits":0,"model_predictions":0,"old_holdout_evaluations":0}
    write_json(cp,claim,True)
    finished=threading.Event()
    def watchdog():
        if not finished.wait(900):
            os._exit(124)
    threading.Thread(target=watchdog,daemon=True).start()
    started=time.monotonic()
    try:
        import numpy as np
        from threadpoolctl import threadpool_limits
        sys.path.insert(0,str(assets))
        from topic_word_round9 import infer_groups
        texts=[f"artificialtopic{i//18:04d} artificialtopic{i//18:04d} common observation" for i in range(4096)]
        ids=[f"artificial-stress-{i:05d}" for i in range(4096)]
        with threadpool_limits(limits=2):
            groups=infer_groups(texts,ids,manifest["grouping"])
        sizes=np.unique(groups,return_counts=True)[1]
        if len(sizes)!=228 or sorted(sizes.tolist())!=[10]+[18]*227:
            raise ValueError("Synthetic groups were not recovered")
        proof={**claim,"state":"verified","completed_groupings":1,"wall_seconds":time.monotonic()-started,
               "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"clusters":len(sizes),
               "source_sha256":digest(assets/"topic_word_round9.py"),"scope":"artificial texts only; runtime check, not hidden accuracy"}
        if proof["peak_rss_bytes"]>12*2**30:
            raise MemoryError("RSS exceeded")
        write_json(target,proof,True)
        claim.update(state="completed",completed_groupings=1,verification_sha256=digest(target))
        write_json(cp,claim)
        print(json.dumps(proof),flush=True)
    except BaseException as error:
        claim.update(state="failed",error_type=type(error).__name__,error=str(error))
        write_json(cp,claim)
        raise
    finally:
        finished.set()


if __name__=="__main__":main()
