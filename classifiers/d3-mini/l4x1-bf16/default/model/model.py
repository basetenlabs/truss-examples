class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        from transformers import AutoModel

        # D3 verifies manifest files through the mount, including symlink targets.
        self._model = AutoModel.from_pretrained(
            "/models/d3",
            trust_remote_code=True,
            local_files_only=True,
            device="cuda:0",
            batch_size=1,
            verify="full",
        )
        runtime = self._model.runtime
        for kind in ("image", "video"):
            reason = getattr(runtime, f"{kind}_unavailable")
            if reason is not None:
                raise RuntimeError(f"D3 {kind} inputs failed to initialize: {reason}")
        self._model.system_one(
            state="The parcel arrived damaged.",
            questions={
                "damaged": {
                    "type": "noul",
                    "instructions": "Is the parcel damaged?",
                }
            },
        )

    def predict(self, model_input):
        runtime = self._model.runtime
        # The HTTP API accepts uploaded data URLs only. Use upstream's strict
        # decoders to check media types, encoded size, dimensions and duration.
        images = runtime.load_images(model_input.get("images") or [], strict=True)
        videos = runtime.load_videos(model_input.get("videos") or [], strict=True)
        return self._model.system_one(
            state=model_input.get("state"),
            questions=model_input["questions"],
            images=images or None,
            videos=videos or None,
        )
