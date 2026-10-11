import copy

import pytest
import torch

from whenami_timeline_joint.test_joint import fixture as joint_fixture, CFG
from whenami_timeline_b.model import Student
from whenami_timeline_transfer.contracts import Prefix
from whenami_revision_control.contracts import RevisionInput, ObservedWindow
from whenami_revision_control.model import RevisionPolicy, NodeMemory, draw, objective, VARIANTS

torch.set_num_threads(1)


def fixture(variant="revised_nodes"):
    a, x = joint_fixture()
    baseline = Student(a.vlm, a.local_encoder, a.reader, "timeline", CFG)
    baseline.timeline.load_state_dict(a.timeline.state_dict())
    torch.manual_seed(777)
    model = RevisionPolicy(copy.deepcopy(a.vlm), copy.deepcopy(a.local_encoder),
                           copy.deepcopy(a.reader), variant, CFG, hidden=16, heads=4)
    result = model.load_state_dict(baseline.state_dict(), strict=False)
    assert all(k.startswith(("memory.", "bridge.")) for k in result.missing_keys)
    b, k = x["times"].shape
    executed = torch.randn(b, k, 12, 7)
    mask = torch.zeros(b, k, 12, dtype=torch.bool)
    for j in range(1, k):
        for n in range(b):
            mask[n, j, :int(x["times"][n,j]-x["times"][n,j-1])] = True
    window = ObservedWindow(x["patches"], x["vae"], x["times"], executed, mask)
    return model, baseline, x, window


def inputs(x, window, count=3):
    b = len(window.times)
    start = max(0, count-2)
    ids = x["input_ids"].reshape(b, 3, -1)[:,start:count].flatten(0, 1)
    attention = x["attention_mask"].reshape(b, 3, -1)[:,start:count].flatten(0, 1)
    time = x["times"][:,start:count]
    prefix = Prefix({k:v[:,start:count] for k,v in x["pixels"].items()}, ids, attention,
                    x["patches"][:,start:count], x["vae"][:,start:count], time-time[:,:1])
    frames=[]
    for j in range(count):
        frames.append(Prefix({k:v[:,j:j+1] for k,v in x["pixels"].items()},
                             x["input_ids"].reshape(b,3,-1)[:,j],
                             x["attention_mask"].reshape(b,3,-1)[:,j],
                             x["patches"][:,j:j+1],x["vae"][:,j:j+1],
                             torch.zeros(b,1,dtype=torch.long)))
    return RevisionInput(prefix, window.prefix(count), tuple(frames)).validate()


@pytest.mark.parametrize("variant", VARIANTS)
def test_zero_bridge_exactly_preserves_B(variant):
    model, base, x, w = fixture(variant)
    model.eval(); base.eval()
    inp = inputs(x, w)
    noise, tau, _ = draw(torch.ones(2, 16, 7), 42)
    expected = base(prefix=inp.base, noisy_actions=noise, tau=tau)
    actual = model(inputs=inp, noisy_actions=noise, tau=tau)
    assert torch.equal(actual["delta"], torch.zeros_like(actual["delta"]))
    for k in ("conditions", "reader_z", "velocity"):
        torch.testing.assert_close(actual[k], expected[k], rtol=0, atol=0)


def test_matched_parameter_sets_and_initial_values():
    a, _, _, _ = fixture("direct_history")
    b, _, _, _ = fixture("revised_nodes")
    assert a.state_dict().keys() == b.state_dict().keys()
    for k, v in a.state_dict().items():
        torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_future_evidence_cannot_change_earlier_prefix(variant):
    m, _, x, w = fixture(variant); m.eval()
    y, q = copy.deepcopy(x), copy.deepcopy(w)
    y["pixels"]["rgb"][:, 2] = torch.nan
    y["input_ids"].reshape(2, 3, -1)[:, 2] = 37
    y["patches"][:, 2] = torch.nan
    q.patches[:, 2] = torch.nan
    q.vae[:, 2] = torch.nan
    q.executed[:, 2] = 1e6
    a = m(inputs=inputs(x,w,2), condition_only=True)
    b = m(inputs=inputs(y,q,2), condition_only=True)
    for k in ("conditions", "reader_z", "node_conditions"):
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)


def test_streamed_node_updates_equal_window_scan():
    m, _, x, w = fixture(); m.eval()
    h = torch.randn(2,3,CFG.queries,CFG.feature)
    local = torch.randn(2,3,CFG.queries,CFG.condition)
    scan = m.memory(h, local, w)
    state = None
    for j in range(3):
        state, result = m.memory.step(h[:,j],local[:,j],w.times[:,j],
                                     w.executed[:,j],w.executed_mask[:,j],state)
        torch.testing.assert_close(result, scan["versions"][j], rtol=0, atol=0)


def test_duplicate_and_missing_executed_interval_rejected():
    _, _, x, w = fixture()
    changed = copy.deepcopy(w)
    changed.times[:, 2] = changed.times[:, 1]
    with pytest.raises(ValueError): changed.validate()
    changed = copy.deepcopy(w)
    changed.executed_mask[:, 1, 0] = False
    with pytest.raises(ValueError): changed.validate()


def test_padded_commands_never_enter_memory():
    m, _, x, w = fixture(); m.eval()
    changed = copy.deepcopy(w)
    changed.executed[~changed.executed_mask] = torch.nan
    a = m(inputs=inputs(x,w), condition_only=True)
    b = m(inputs=inputs(x,changed), condition_only=True)
    for k in ("conditions", "node_conditions"):
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_action_gradient_reaches_supervised_old_nodes(variant):
    m, _, x, w = fixture(variant)
    with torch.no_grad(): m.bridge.output.weight.normal_(std=.05)
    inp=inputs(x,w)
    noise,tau,target=draw(torch.ones(2,16,7),42)
    out=m(inputs=inp,noisy_actions=noise,tau=tau)
    out["node_conditions"].retain_grad()
    loss=(out["velocity"]-target).square().mean()
    loss.backward()
    assert out["node_conditions"].grad[:, :-1].abs().sum()>0
    assert sum(float(p.grad.abs().sum()) for p in m.memory.parameters() if p.grad is not None)>0
    assert m.memory.actions.weight_ih.grad.abs().sum()>0
    assert m.memory.cognition[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in m.forecast_head.parameters())


def test_revision_intervention_changes_action_and_targets_detached():
    m, _, x, w = fixture()
    with torch.no_grad():m.bridge.output.weight.normal_(std=.1)
    m.eval();noise,tau,target=draw(torch.ones(2,16,7),8)
    a=m(inputs=inputs(x,w),noisy_actions=noise,tau=tau)
    b=m(inputs=inputs(x,w),noisy_actions=noise,tau=tau,unrevised_action=True)
    assert not torch.allclose(a["velocity"],b["velocity"])
    before=torch.randn_like(a["node_versions"][-2],requires_grad=True)
    after=torch.randn_like(a["node_conditions"][:,:-1],requires_grad=True)
    loss,_=objective(a,target,torch.ones(2,16,dtype=torch.bool),before,after)
    loss.backward()
    assert before.grad is None and after.grad is None


def test_direct_history_is_not_recurrent_revision():
    a,_,x,w=fixture("direct_history");b,_,_,_=fixture("revised_nodes")
    a.eval();b.eval()
    oa=a(inputs=inputs(x,w),condition_only=True)
    ob=b(inputs=inputs(x,w),condition_only=True)
    assert not torch.allclose(oa["node_conditions"],ob["node_conditions"])


def test_last_target_actions_not_used_as_executed_inputs():
    from whenami_revision_control.data import executed_intervals
    case={"observations":{"times":torch.tensor([0,4,11])},
          "labels":{"actions":torch.randn(3,16,7),"action_mask":torch.ones(3,16,dtype=torch.bool)}}
    a=executed_intervals(case)
    changed=copy.deepcopy(case);changed["labels"]["actions"][-1]=torch.nan
    b=executed_intervals(changed)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    torch.testing.assert_close(a[0][1,:4],case["labels"]["actions"][0,:4])
    torch.testing.assert_close(a[0][2,:7],case["labels"]["actions"][1,:7])


def test_cold_start_has_no_fake_history_or_delta():
    m,base,x,w=fixture();m.eval();base.eval()
    inp=inputs(x,w,1)
    out=m(inputs=inp,condition_only=True)
    expected=base(prefix=inp.base,condition_only=True)
    torch.testing.assert_close(out["conditions"],expected["conditions"],rtol=0,atol=0)


def test_full_scan_old_versions_do_not_read_later_frame_cognition():
    m,_,x,w=fixture();m.eval()
    y=copy.deepcopy(x)
    y["pixels"]["rgb"][:,2]+=1000
    y["input_ids"].reshape(2,3,-1)[:,2]=37
    a=m(inputs=inputs(x,w),condition_only=True)
    b=m(inputs=inputs(y,w),condition_only=True)
    for i in (0,1):
        torch.testing.assert_close(a["node_versions"][i],b["node_versions"][i],rtol=0,atol=0)
    assert not torch.allclose(a["node_versions"][2],b["node_versions"][2])
