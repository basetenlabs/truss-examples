# Truss templates for different model backends
`generate.yaml` contains configurations of different models with different backends / engines. `generate.py` generates described models by copying content of the template and overriding config with provided values.

`generate.py` accepts following arguments:
- `--only_check` if passed files aren't getting generated, fails if currently existing files are different from suppose to be generated ones 
- `--root` path to root of `truss-examples`, models are being generated under this path
- `--templates` path to templates, generator reads `based_on` models from it 
- `--config` path to generation config
