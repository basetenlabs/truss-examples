## Autodesk WaLa (single‑view image → 3D) Truss

- Status: supports only `ADSKAILab/WaLa-SV-1B` (single‑view) at the moment
- Output: OBJ (default) or SDF, base64‑encoded in the response
- License: non‑commercial per the model card

### Prerequisites
- Baseten account and API key.
- Hugging Face access:
  - Accept the model license on `ADSKAILab/WaLa-SV-1B`.
  - Add a Baseten secret named `hf_access_token` with your token (read access).

### Deploy
From this folder (`autodesk-wala`):

```bash
# Option A: Truss CLI
truss push --trusted --publish
```

Notes:
- You can use the client code in `autodesk-wala/test.py` to call your endpoint and save the OBJ locally.
- The Truss downloads WaLa source code at runtime (pinned commit) and imports `src/*` to run inference.
- The HF token is read from the `hf_access_token` secret and exported for `hf_hub_download`.

### Invoke (Python)
A simple client is provided in `test.py`. It reads `examples/single_view/table.png`, sends it to your deployed endpoint, saves `output.obj`, and displays it if `trimesh` + `plotly` are installed.

```bash
export BASETEN_API_KEY=...  # your Baseten API key
python autodesk-wala/test.py
```

### Invoke (cURL)
Replace `model_id` and `API_KEY` with your values.

```bash
IMG_B64=$(base64 -i autodesk-wala/examples/single_view/table.png)

curl -s -X POST "https://model-<model_id>.api.baseten.co/production/predict" \
  -H "Authorization: Api-Key <API_KEY>" \
  -H "Content-Type: application/json" \
  -d "{\
    \"image_b64\": \"${IMG_B64}\",\
    \"model_name\": \"ADSKAILab/WaLa-SV-1B\",\
    \"output_format\": \"obj\",\
    \"scale\": 1.8,\
    \"diffusion_rescale_timestep\": 5,\
    \"seed\": 42\
  }" | tee response.json

# Decode OBJ
jq -r .obj_b64 response.json | base64 --decode > output.obj
```

### Request schema
- Required:
  - `image_b64`: base64‑encoded RGB image (single view)
- Optional:
  - `model_name`: HF repo id (defaults to `ADSKAILab/WaLa-SV-1B`)
  - `output_format`: `obj` (default) or `sdf`
  - `scale`: float, default `3.0` (authors often use `1.8`)
  - `diffusion_rescale_timestep`: int, default `100` (authors often use `5`)
  - `seed`: int, default `42`
  - `target_num_faces`: int, optional mesh simplification target (server best‑effort; skipped if Open3D/libGL unavailable). For Colab‑like behavior, omit this.

### Response
- Success:
  - `obj_b64` or `sdf_b64`: base64 of the generated asset
  - `data`: same base64 payload (for convenience)
  - `format`: `obj` or `sdf`
  - `output_path`: server‑side path where the file was written
  - `time`: seconds
- Error:
  - `{ "status": "error", "message": "..." }`
