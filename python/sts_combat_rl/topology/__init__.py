"""Value net topologies, one frozen file per kind (topology/README.md)."""

import inspect

from .deep_sets_v1 import DeepSetsV1
from .deep_sets_v2 import DeepSetsV2
from .deep_sets_v3 import DeepSetsV3

KINDS = {cls.KIND: cls for cls in (DeepSetsV1, DeepSetsV2, DeepSetsV3)}
# The encoding version each kind reads (combat/encoding.hpp).
ENCODING_VERSIONS = {"deep_sets_v1": 3, "deep_sets_v2": 3, "deep_sets_v3": 4}


def build(architecture):
    """The value net for an architecture dict: "kind" plus every constructor argument of that kind, all required.
    model.config is the same dict."""
    architecture = dict(architecture)
    kind = architecture.pop("kind")
    if kind not in KINDS:
        raise ValueError(f"unknown value net kind {kind!r}")
    expected = set(inspect.signature(KINDS[kind].__init__).parameters) - {"self"}
    if set(architecture) != expected:
        raise ValueError(f"{kind} architecture: missing {sorted(expected - set(architecture))}, "
                         f"unknown {sorted(set(architecture) - expected)}")
    return KINDS[kind](**architecture)
