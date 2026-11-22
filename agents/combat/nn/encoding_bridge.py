"""Python bridge to the native combat-state token encoders."""
import functools
import importlib.util
import os
from pathlib import Path
import sysconfig


@functools.cache
def _native():
    build = Path(__file__).resolve().parents[3] / 'build' / os.environ.get('STSRL_BUILD_DIR', 'main')
    path = build / 'agents/combat/nn' / ('_encoding_bridge' + sysconfig.get_config_var('EXT_SUFFIX'))
    spec = importlib.util.spec_from_file_location('_encoding_bridge', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode(states, action_orders, encoding):
    """Return float32 token buffers in the supplied uint32 action orders (same-build bytes only)."""
    return _native().encode(states, action_orders, encoding)


def state_codec_version():
    return _native().state_codec_version
