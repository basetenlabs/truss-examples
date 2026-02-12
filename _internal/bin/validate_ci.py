import truss
import yaml

with open("ci.yaml", "r") as file:
    paths = yaml.safe_load(file)

for path in paths["tests"]:
    _ = truss.load(path)
