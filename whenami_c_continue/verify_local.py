"""Source-bound proof from tests, all parent optimizer audits, and real models."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from stage3.common import BASE
from stage3_oxe_eval.prepare import sha,write_json
from whenami_c_continue.contracts import PARENTS,source_record,SCHEDULE
from whenami_c_continue.train import fingerprints


def main():
    p=argparse.ArgumentParser();p.add_argument("--evidence",type=Path,required=True);a=p.parse_args()
    root=a.evidence
    suites=list(ET.parse(root/"pytest.xml").getroot().iter("testsuite"))
    if not suites or any(int(s.get(k,0)) for s in suites for k in ("failures","errors","skipped")):
        raise ValueError("continuation tests incomplete")
    tests=sum(int(s.get("tests",0)) for s in suites)
    names={c.get("name") for s in suites for c in s.findall("testcase")}
    if (tests<16 or "test_cpu4rank_parent_and_continuation_restore_next_update_bitwise" not in names
            or "test_original_teacher_and_student_microstep_unchanged" not in names):
        raise ValueError("missing parent-to-continuation optimizer/RNG test")
    parents={case:source_record(case) for case in PARENTS}
    audit=json.loads((root/"PARENT_AUDIT.json").read_text())
    smoke_path=root/"real_smoke/RESULT.json"
    smoke=json.loads(smoke_path.read_text())
    if not audit["complete"] or not smoke["complete"] or not smoke["teacher_frozen"] or not smoke["student_teacher_private"]:
        raise ValueError("parent audit/real model verification incomplete")
    for case in PARENTS:
        row=audit["parents"][case]
        if (not row["complete"] or row["parent"]!=parents[case] or row["model_entries"]!=1549
                or len(row["optimizer_shards"])!=16 or row["rng_shards_verified"]!=320):
            raise ValueError("original parent Adam/RNG not audited: "+case)
        row=smoke["cases"][case]
        if not row["complete"] or row["parent"]!=parents[case] or row["loaded_C_entries"]!=1549:
            raise ValueError("actual C20k model not checked: "+case)
        if row["read_scale"]!=parents[case]["read_scale"] or row["condition_tokens"]!=32:
            raise ValueError("actual C read-in geometry changed")
    for name,digest in smoke["source"].items():
        if sha(BASE/"whenami_c_continue"/name)!=digest:
            raise ValueError("source modified after real smoke: "+name)
    protected=json.loads((root/"PROTECTED_SOURCE_BEFORE.json").read_text())
    for name,digest in protected.items():
        if sha(BASE/name)!=digest:
            raise ValueError("existing training source/launcher changed: "+name)
    proof={"checked_utc":datetime.now(timezone.utc).isoformat(),"local_passed":True,
        "source":fingerprints(),"launcher_sha256":sha(BASE/"run_c_continue_320.sh"),
        "parents":parents,"tests_passed":tests,"schedule":SCHEDULE,
        "all32_parent_Adam_shards_parsed":True,"all640_parent_RNG_records_parsed":True,
        "two_actual_C20k_parent_forward_backward_passed":True,
        "CPU4rank_parent_and_continuation_next_update_bitwise":True,
        "parent_recipe_immutability_checked":True,"protected_existing_jobs_unchanged":True,
        "peak_smoke_allocated_gib":smoke["peak_allocated_gib"],
        "real_smoke_sha256":sha(smoke_path),"parent_audit_sha256":sha(root/"PARENT_AUDIT.json"),
        "pytest_xml_sha256":sha(root/"pytest.xml"),"evidence_root":str(root),
        "actual320_full_C_parent_restore_and_replay_passed":False,
        "scope":"Local engineering admission only. Full320 original model/Adam/RNG restore and saved DEV replay remain mandatory at cluster launch."}
    write_json(BASE/"whenami_c_continue/LOCAL_VERIFICATION.json",proof)
    print("C_CONTINUATION_LOCAL_VERIFIED",tests,list(parents),flush=True)


if __name__=="__main__":main()
