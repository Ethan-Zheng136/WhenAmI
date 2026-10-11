"""Verify observed command intervals against original OXE records, without any GPU."""
import argparse
from pathlib import Path

import numpy as np
import torch

from stage3_oxe_eval.prepare import write_json, sha
from evaluation.past_protocol import normalize_actions
from whenami_revision_control.data import Inputs, executed_intervals, INDEX, DEV


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    data=Inputs()
    rows=[]
    for domain in data.raw.domains:
        entry,times=data.raw.select(20261007,1,0,0,domain=domain)
        observations,labels,provenance=data.raw.materialize(entry,times,data.stats)
        case={"observations":observations,"labels":labels,"provenance":provenance}
        actions,mask=executed_intervals(case)
        _,episode=data.raw._episode(entry)
        for j in (1,2):
            expected=normalize_actions(episode["raw_actions"][int(times[j-1]):int(times[j])].numpy(),
                                       data.stats[domain]["action"])
            assert np.array_equal(actions[j,mask[j]].numpy(),expected)
        before=actions.clone()
        labels["actions"][2]=torch.nan
        after,_=executed_intervals(case)
        assert torch.equal(before,after)
        rows.append({"dataset":domain,"episode_key":provenance["episode_key"],
                     "source_times":times.tolist(),"executed_lengths":mask.sum(-1).tolist(),
                     "actual_raw_command_intervals_equal":True})
        print("TRAIN_DOMAIN_ACTION_AUDIT",domain,flush=True)
    for i in range(len(data.clips)):
        case=data.dev(i);actions,mask=executed_intervals(case)
        assert torch.isfinite(actions[mask]).all()
        assert torch.equal(mask[1:].sum(-1),case["observations"]["times"].diff())
    write_json(a.output,{"complete":True,"train_domains":len(rows),"train_examples":rows,
        "dev_episodes":len(data.clips),"dev_domains":len({r["dataset"] for r in data.clips}),
        "train_index_sha256":sha(INDEX),"dev_manifest_sha256":sha(DEV/"MANIFEST.json"),
        "last_target_actions_never_used_as_model_input":True,
        "protocol":"revision_last_anchor_executed_history_v1",
        "scope":"Observed images/commands aligned; not model training or closed-loop evaluation"})


if __name__=="__main__":main()
