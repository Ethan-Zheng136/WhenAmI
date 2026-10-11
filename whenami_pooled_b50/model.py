"""Original causal B, with pooled_query aggregation in BOTH student and teacher."""

from whenami_timeline_b.model import Student, draw, objective
from whenami_timeline_joint.model import load_joint

MODE = "pooled_query"


def from_base(base):
    if base.timeline.mode != MODE:
        raise ValueError("pooled B must inherit a pooled_query A model")
    student = Student(base.vlm, base.local_encoder, base.reader, MODE,
                      base.timeline.config)
    student.timeline.load_state_dict(base.timeline.state_dict(), strict=True)
    if student.timeline.mode != MODE:
        raise ValueError("student unexpectedly reverted to timeline attention")
    return student


def build_student(seed=42):
    base, visual, transform, tokenizer, prompt = load_joint(MODE, seed)
    return from_base(base), visual, transform, tokenizer, prompt
