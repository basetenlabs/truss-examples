import importlib
import sys


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        # Load the complete mounted package: Transformers' dynamic-module cache
        # can miss transitive relative imports when loading a local directory.
        sys.path.insert(0, "/models")
        model_class = importlib.import_module("kai.modeling_decision1").Decision1Model
        self._model = model_class.from_pretrained(
            "/models/kai", local_files_only=True, device="cuda:0"
        )
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
        return self._model.system_one(
            state=model_input["state"], questions=model_input["questions"]
        )
