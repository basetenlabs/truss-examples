import base64, json, requests
import os

with open("examples/single_view/table.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "image_b64": img_b64,
    "model_name": "ADSKAILab/WaLa-SV-1B",
    "output_format": "obj",
    "scale": 1.8,
    "diffusion_rescale_timestep": 5,
    "seed": 42,
}

# Hosted
url = "https://model-lqznelkq.api.baseten.co/development/predict"
api_key = os.getenv("BASETEN_API_KEY")
headers = {"Authorization": f"Api-Key {api_key}"} if api_key else {}

resp = requests.post(url, headers=headers, json=payload)
resp.raise_for_status()
r = resp.json()

# Use obj_b64 if present, else fall back to data
b64 = r.get("obj_b64") or r.get("data")
if not b64:
    raise RuntimeError(f"No base64 data found in response: {json.dumps(r, indent=2)}")

outfile = "output.obj"
with open(outfile, "wb") as f:
    f.write(base64.b64decode(b64))
print("Saved:", r.get("output_path", "<no path>"), "and", outfile)

# Helper to display the generated mesh (matches Colab style)
def display_mesh(mesh_file: str):
    try:
        import trimesh
        import plotly.graph_objs as go
    except Exception as e:
        print(f"Display skipped (missing deps): {e}. Tip: pip install trimesh plotly")
        return
    mesh = trimesh.load(mesh_file)
    vertices = mesh.vertices
    faces = mesh.faces
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#B4B7C2",
                lighting=dict(ambient=0.5, diffuse=1, specular=0.5, roughness=0.5),
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            bgcolor="white",
        )
    )
    fig.show()

# Display the saved OBJ
display_mesh(outfile)