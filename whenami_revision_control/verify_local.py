"""Create source-bound admission proof only from actual completed local evidence."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from stage3.common import BASE
from stage3_oxe_eval.prepare import sha,write_json
from whenami_revision_control.loading import source_record
from whenami_revision_control.train import fingerprints


def main():
    p=argparse.ArgumentParser();p.add_argument("--evidence",type=Path,required=True);a=p.parse_args()
    root=a.evidence
    suites=list(ET.parse(root/"pytest.xml").getroot().iter("testsuite"))
    if not suites or any(int(s.get(k,0)) for s in suites for k in ("failures","errors","skipped")):
        raise ValueError("unit/distributed/launcher tests incomplete")
    count=sum(int(s.get("tests",0)) for s in suites)
    if count<26:raise ValueError("missing revision tests")
    names={c.get("name") for s in suites for c in s.findall("testcase")}
    for n in ("test_cpu4rank_B_warmstart_and_full_optimizer_rng_restart",
              "test_full_scan_old_versions_do_not_read_later_frame_cognition",
              "test_last_target_actions_not_used_as_executed_inputs"):
        if n not in names:raise ValueError("missing critical test "+n)
    smoke_path=root/"real_smoke_v2/RESULT.json"
    smoke=json.loads(smoke_path.read_text())
    for key in ("complete","initial_B_condition_cognition_velocity_bitwise_equal",
                "teacher_student_nonaliased","teacher_frozen",
                "later_evidence_and_future_target_action_invariance"):
        if smoke.get(key) is not True:raise ValueError("real smoke missing "+key)
    sources=source_record()
    if smoke["sources"]!=sources or smoke["student_load"]["loaded_entries"]!=1550 or smoke["teacher_load"]["loaded_entries"]!=1548:
        raise ValueError("actual source loading not verified")
    if min(smoke["controlled_open_bridge_old_node_action_gradient"],
           smoke["controlled_open_bridge_velocity_intervention_max_abs"])<=0:
        raise ValueError("old-node revision not connected to action")
    for name,digest in smoke["source"].items():
        if sha(BASE/"whenami_revision_control"/name)!=digest:
            raise ValueError("source changed after real smoke: "+name)
    audit=json.loads((root/"DATA_AUDIT.json").read_text())
    if not audit["complete"] or audit["train_domains"]!=27 or audit["dev_domains"]!=27 or audit["dev_episodes"]!=410:
        raise ValueError("full-domain command/observation audit incomplete")
    protected=json.loads((root/"PROTECTED_SOURCE_BEFORE.json").read_text())
    for name,digest in protected.items():
        if sha(BASE/name)!=digest:raise ValueError("protected existing source changed: "+name)
    proof={"checked_utc":datetime.now(timezone.utc).isoformat(),"local_passed":True,
        "source":fingerprints(),"launcher_sha256":sha(BASE/"run_revision_control_320.sh"),
        "sources":sources,"tests_passed":count,"real_smoke_sha256":sha(smoke_path),
        "data_audit_sha256":sha(root/"DATA_AUDIT.json"),"pytest_xml_sha256":sha(root/"pytest.xml"),
        "initial_B_action_path_bitwise_preserved":True,"teacher_frozen_and_private":True,
        "full_cognition_per_memory_node_verified":True,
        "node_revision_action_gradient_and_intervention_verified":True,
        "future_observation_and_target_command_isolation_verified":True,
        "cpu4rank_nested_fsdp_optimizer_rng_restart_verified":True,
        "train_domains_audited":27,"dev_episodes_audited":410,
        "peak_real_smoke_allocated_gib":smoke["peak_allocated_gib"],
        "real_smoke_update":smoke["actual_update"],
        "protected_existing_A_B_C_sources_unchanged":True,
        "actual320rank_full_Adam_admission_complete":False,
        "learned_effectiveness_established":False,"evidence_root":str(root),
        "scope":"Three observed anchors, last-action evaluation. Controlled bridge intervention is not a learned gain."}
    write_json(BASE/"whenami_revision_control/LOCAL_VERIFICATION.json",proof)
    print("REVISION_LOCAL_VERIFIED",count,sources,flush=True)


if __name__=="__main__":main()
