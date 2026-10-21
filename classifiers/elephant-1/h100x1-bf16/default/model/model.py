import subprocess

MODEL_NAME = "elephant-1"
REPO_NAME = "baselabs/elephant-1"
MODEL_PATH = "/models/elephant"
ONEJEV_PATH = "/opt/OneJev"


def apply_serve_patch():
    # serve.patch ships with the pinned weights (not vendored here); skip it if a restart already applied it.
    patch = f"{MODEL_PATH}/serve.patch"
    git = ["git", "-C", ONEJEV_PATH, "apply"]
    if subprocess.run([*git, "--reverse", "--check", patch], capture_output=True).returncode != 0:
        subprocess.run([*git, patch], check=True)


class Model:
    def load(self):
        apply_serve_patch()
        from qev.engine import DecisionEngine
        from qev.schema import SystemOneRequest

        self._request = SystemOneRequest
        # Upstream engine.py defaults: bf16, fp32 LM head, readout/attention from the checkpoint, 32,768-token
        # branch and 65,536-token request limits.
        self._engine = DecisionEngine(MODEL_PATH, device="cuda:0")
        self.predict({"model": MODEL_NAME, "state": "The parcel arrived damaged.",
                      "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}}})

    def predict(self, payload):
        request = self._request(state=payload["state"], questions=payload["questions"])
        try:
            response, _ = self._engine.decide(request)
        except RuntimeError as exc:
            # Upstream engine.py: the batched prefix fork overflows 32-bit indexing on the largest multi-question
            # requests; sequential branches are OneJev's exact fallback for the same computation.
            if "32BitIndexMath" not in str(exc):
                raise
            mode, self._engine.info.fork_mode = self._engine.info.fork_mode, "sequential"
            try:
                response, _ = self._engine.decide(request)
            finally:
                self._engine.info.fork_mode = mode
        return {**response.model_dump(), "model": payload["model"]}
