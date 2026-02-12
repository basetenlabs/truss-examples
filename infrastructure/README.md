# Infrastructure Examples

Examples demonstrating advanced Truss features, custom serving infrastructure, and integrations. These cover topics like custom servers, gRPC, chaining models, caching, and specialized tooling.

| Directory | Description |
|-----------|-------------|
| [custom-server](custom-server/) | Custom inference servers using SGLang, LMDeploy, and other engines with Dockerfile-based Truss configs |
| [grpc](grpc/) | Serve a model over gRPC instead of HTTP |
| [custom-engine-builder-control](custom-engine-builder-control/) | Custom engine builder with fine-grained control over the build process |
| [chains-examples](chains-examples/) | Truss Chains examples for multi-model pipelines |
| [multiprocessing](multiprocessing/) | Use Python multiprocessing within a Truss model |
| [model-cache](model-cache/) | Cache model weights across deployments for faster cold starts |
| [metrics](metrics/) | Export custom metrics from a Truss model |
| [jsonformatter](jsonformatter/) | Custom JSON formatting for model outputs |
| [ngram-speculator](ngram-speculator/) | N-gram speculative decoding for faster LLM inference |
| [llama-cpp-server](llama-cpp-server/) | Serve models using llama.cpp as the backend |
| [binocular](binocular/) | Binocular LLM-generated text detection |
| [layoutlm-document-qa](layoutlm-document-qa/) | LayoutLM document question answering |
| [autodesk-wala](autodesk-wala/) | Autodesk WALA model integration |
| [paddlepaddle](paddlepaddle/) | PaddlePaddle framework model serving |

## Deploying

Each example can be deployed to Baseten with:

```bash
truss push
```
