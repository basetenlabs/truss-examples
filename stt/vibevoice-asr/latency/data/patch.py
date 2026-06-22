"""Apply microsoft/VibeVoice plugin patches at container boot.

The Microsoft VibeVoice vLLM plugin (installed via `pip install
git+https://github.com/microsoft/VibeVoice.git`) has three known issues
against current vLLM releases. This script patches the installed
`vllm_plugin/model.py` file in-place at boot, before `vllm serve` starts.

Patches applied:
  1. KV-cache delegation forwarders on VibeVoiceForCausalLM (REQUIRED for vLLM 0.14.1)
     Without this, vLLM 0.14.1 sees zero attention layers and startup crashes
     with `IndexError: list index out of range` on `available_gpu_memory[0]`.

  2. get_data_parser method on VibeVoiceProcessingInfo (forward-compat for vLLM 0.21+)
     v0.14.1 calls `_get_data_parser` on the Processor; v0.21+ calls `get_data_parser`
     on the Info class. No-op on v0.14.1.

  3. mm_data_items field rename try/except (forward-compat for vLLM 0.15+)
     v0.14.1 ProcessorInputs uses `mm_data`; v0.15+ uses `mm_data_items`.
     No-op on v0.14.1.

Idempotent: re-running is a no-op once the marker is present.
Anchor-checked: each insertion `assert`s its target exists, so an upstream
plugin update that breaks the anchors causes a loud startup error instead
of silent miscompile.

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


# ── Patch 1: KV-cache delegation forwarders (REQUIRED on vLLM 0.14.1) ────────
ANCHOR_1 = (
    "    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:\n"
    "        return self.language_model.compute_logits(hidden_states)"
)
INSERT_1 = """

    # [Truss local patch] vLLM 0.14.1 KV-cache discovery looks for
    # `.get_kv_cache_spec()` and `.model` on the top-level registered class.
    # VibeVoiceForCausalLM is a wrapper around self.language_model — forward
    # these so vLLM sees the real attention layers. Without this, startup
    # crashes with IndexError on `available_gpu_memory[0]`.
    def get_kv_cache_spec(self, *args, **kwargs):
        return self.language_model.get_kv_cache_spec(*args, **kwargs)

    @property
    def model(self):
        return self.language_model.model"""

assert ANCHOR_1 in src, (
    "Patch 1 anchor (compute_logits signature) not found in plugin — upstream may have changed"
)
src = src.replace(ANCHOR_1, ANCHOR_1 + INSERT_1)


# ── Patch 2: get_data_parser on Info class (forward-compat for vLLM 0.21+) ──
ANCHOR_2 = (
    "class VibeVoiceProcessingInfo(BaseProcessingInfo):\n"
    '    """Processing info for VibeVoice multimodal model."""'
)
INSERT_2 = """

    # [Truss local patch] vLLM 0.21+ calls info.get_data_parser() on this
    # class (the plugin only defines _get_data_parser on the Processor).
    # No-op on v0.14.1 which uses the older API.
    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=24000)"""

assert ANCHOR_2 in src, (
    "Patch 2 anchor (VibeVoiceProcessingInfo class) not found in plugin"
)
src = src.replace(ANCHOR_2, ANCHOR_2 + INSERT_2)


# ── Patch 3: mm_data_items rename try/except (forward-compat for vLLM 0.15+) ─
ANCHOR_3 = (
    '        """Build ProcessorInputs for dummy profiling."""\n'
    "        return ProcessorInputs(\n"
    "            prompt=self.get_dummy_text(mm_counts),\n"
    "            mm_data=self.get_dummy_mm_data(seq_len, mm_counts, mm_options),\n"
    "        )"
)
REPLACE_3 = '''        """[Truss local patch] vLLM 0.15+ renamed mm_data → mm_data_items.
        Try the new signature first, fall back to legacy for v0.14.1."""
        mm_data_dict = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        try:
            mm_data_items = MultiModalDataParser().parse_mm_data(mm_data_dict)
            return ProcessorInputs(
                prompt=self.get_dummy_text(mm_counts),
                mm_data_items=mm_data_items,
            )
        except TypeError:
            return ProcessorInputs(
                prompt=self.get_dummy_text(mm_counts),
                mm_data=mm_data_dict,
            )'''

assert ANCHOR_3 in src, (
    "Patch 3 anchor (get_dummy_processor_inputs body) not found in plugin"
)
src = src.replace(ANCHOR_3, REPLACE_3)


# Write back
open(TARGET, "w").write(src)
print(f"[patch.py] applied 3 patches to {TARGET}")
