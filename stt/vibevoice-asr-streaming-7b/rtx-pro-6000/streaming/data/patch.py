"""Apply the one microsoft/VibeVoice plugin patch the streaming server needs.

Runs against the plugin installed from
git+https://github.com/microsoft/VibeVoice.git@1541f590c70 (pinned in
config.yaml) and rewrites the installed `vllm_plugin/model.py` in place at
boot, before the streaming ASR server imports it.

Patch applied (REQUIRED on vLLM 0.14.1):
  KV-cache delegation forwarders on VibeVoiceForCausalLM. vLLM's KV-cache
  discovery looks for `.get_kv_cache_spec()` and `.model` on the top-level
  registered class. VibeVoiceForCausalLM defines neither -- it is a wrapper
  holding the real Qwen2 decoder as `self.language_model`. Without the
  forwarders vLLM sees zero attention layers and startup crashes with
  `IndexError: list index out of range` on `available_gpu_memory[0]`.
  The streaming server builds its engine through AsyncLLM.from_engine_args,
  which runs the same discovery, so the fix is needed here exactly as it is
  for `vllm serve`.

Deliberately NOT carried over from stt/vibevoice-asr/latency/data/patch.py:
  - `get_data_parser` on the Info class. A forward-compat shim for vLLM
    0.21+, inert on the pinned 0.14.1 (which calls `_get_data_parser` on the
    Processor, and the plugin defines that). It would also return a plain
    MultiModalDataParser where the plugin uses `_NoneTolerantAudioParser`,
    so if it ever did fire it would be subtly wrong.
  - the `mm_data` -> `mm_data_items` rename try/except. A forward-compat shim
    for vLLM 0.15+, and unlike the sibling this package cannot treat it as
    dead code: the sibling passes `--skip-mm-profiling` to `vllm serve`,
    while the streaming server's AsyncEngineArgs does not set
    skip_mm_profiling, so dummy profiling really runs and the rewritten body
    would be live code with no upstream-behaviour guarantee. The image tag is
    pinned immutably, so neither shim can ever be reached.

Idempotent: re-running is a no-op once the marker is present.
Anchor-checked: the insertion `assert`s its target exists, so a plugin update
that breaks the anchor causes a loud startup error instead of a silent
miscompile. The plugin is commit-pinned, so that assert should only ever fire
after a deliberate bump.

Once Microsoft fixes the plugin upstream, delete this file and remove the
patch.py call from config.yaml's start_command.
"""

import os
import sys

import vllm_plugin

TARGET = os.path.join(os.path.dirname(vllm_plugin.__file__), "model.py")
MARKER = "[Truss local patch]"

src = open(TARGET).read()

if MARKER in src:
    print(f"[patch.py] {TARGET} already patched, skipping")
    sys.exit(0)


# ── KV-cache delegation forwarders (REQUIRED on vLLM 0.14.1) ─────────────────
ANCHOR = (
    "    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:\n"
    "        return self.language_model.compute_logits(hidden_states)"
)
INSERT = """

    # [Truss local patch] vLLM 0.14.1 KV-cache discovery looks for
    # `.get_kv_cache_spec()` and `.model` on the top-level registered class.
    # VibeVoiceForCausalLM is a wrapper around self.language_model -- forward
    # these so vLLM sees the real attention layers. Without this, startup
    # crashes with IndexError on `available_gpu_memory[0]`.
    def get_kv_cache_spec(self, *args, **kwargs):
        return self.language_model.get_kv_cache_spec(*args, **kwargs)

    @property
    def model(self):
        return self.language_model.model"""

assert ANCHOR in src, (
    "patch anchor (compute_logits signature) not found in plugin — upstream "
    "may have changed; re-check the pinned VibeVoice commit"
)
src = src.replace(ANCHOR, ANCHOR + INSERT)

open(TARGET, "w").write(src)
print(f"[patch.py] applied 1 patch to {TARGET}")
