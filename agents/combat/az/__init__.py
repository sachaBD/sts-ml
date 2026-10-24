"""AlphaZero-style combat agent on the JAX simulator (jaxsim). See README.md.

On macOS (CPU only, used for smoke runs and tests) XLA's LLVM backend optimiser takes minutes on jaxsim's batched
step (over 6 minutes for vmap(step) alone); optimisation level 0 compiles it in seconds. Set before JAX starts;
an explicit XLA_FLAGS wins. Other platforms (the GPU box) are untouched.
"""
import os
import sys

if sys.platform == "darwin" and "xla_backend_optimization_level" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_backend_optimization_level=0").strip()
