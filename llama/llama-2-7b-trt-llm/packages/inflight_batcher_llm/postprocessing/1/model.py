# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import json

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer
from collections import OrderedDict


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Parse model configs
        model_config = json.loads(args["model_config"])
        tokenizer_dir = os.environ["triton_tokenizer_repository"]
        tokenizer_type = model_config["parameters"]["tokenizer_type"]["string_value"]

        if tokenizer_type == "t5":
            self.tokenizer = T5Tokenizer(vocab_file=tokenizer_dir, padding_side="left")
        elif tokenizer_type == "auto":
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir, padding_side="left"
            )
        elif tokenizer_type == "llama":
            self.tokenizer = LlamaTokenizer.from_pretrained(
                tokenizer_dir, legacy=False, padding_side="left"
            )
        else:
            raise AttributeError(f"Unexpected tokenizer type: {tokenizer_type}")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Parse model output configs
        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")
        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        self.state_dict = OrderedDict()
        self.cache_size = 100

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            # Get request ID
            request_id = request.request_id()

            # Get input tensors
            tokens_batch = (
                pb_utils.get_input_tensor_by_name(request, "TOKENS_BATCH")
                .as_numpy()
                .flatten()
            )

            # Get prior state for request ID
            if request_id in self.state_dict:
                previous_tokens = self.state_dict[request_id]["tokens"]
                accumulated_tokens = np.concatenate([previous_tokens, tokens_batch])
                self.state_dict[request_id]["tokens"] = accumulated_tokens

                # Move request ID to end of queue to prevent it from being evicted
                self.state_dict.move_to_end(request_id)
            else:
                # Evict least recently used item if cache is full
                if len(self.state_dict) > self.cache_size:
                    self.state_dict.popitem(last=False)

                self.state_dict[request_id] = {"tokens": tokens_batch, "prev_str": ""}

            # Postprocess output data
            new_string = self._postprocessing(self.state_dict[request_id]["tokens"])
            old_string = self.state_dict[request_id]["prev_str"]

            # Compute delta between previous and new string
            delta = self._compute_delta(old_string, new_string)
            self.state_dict[request_id]["prev_str"] = new_string

            # Create output tensor
            output_tensor = pb_utils.Tensor(
                "OUTPUT", np.array([delta]).astype(self.output_dtype)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        return responses

    def _compute_delta(self, prev_str, new_str):
        delta = "".join(
            [
                char
                for index, char in enumerate(new_str)
                if index >= len(prev_str) or char != prev_str[index]
            ]
        )
        return delta

    def finalize(self):
        print("Cleaning up...")

    def _postprocessing(self, tokens):
        decoded_tokens = self.tokenizer.decode(tokens)
        return decoded_tokens
