import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel
from truss.patch.hash import directory_content_hash


class Replacement(BaseModel):
    from_str: str
    to_str: str


class Generate(BaseModel):
    based_on: str
    config: Dict[str, Any]
    ignore: List[str]
    template: Dict[str, str]


def process(dst: Path, templates: Path, generate: Generate, only_check: bool):
    logging.info(f"processing {str(dst)}")
    with tempfile.TemporaryDirectory() as temp_dir:
        generated = Path(temp_dir) / "truss"

        # copy template
        shutil.copytree(templates / generate.based_on, generated)
        # copy ignore files
        for ignored in generate.ignore:
            shutil.copy2(dst / ignored, generated / ignored)
        # apply config changes
        template_config = yaml.safe_load((generated / "config.yaml").read_text())
        merged_config = merge_configs(template_config, generate.config)
        (generated / "config.yaml").write_text(merged_config)
        # apply template variables
        for filepath in generated.rglob("*"):
            if filepath.suffix == ".jinja":
                apply_template(filepath, generate.template)

        if only_check:
            # check if directories are the same
            if directory_content_hash(generated) == directory_content_hash(dst):
                logging.info("Generated content is the same as existing")
            else:
                raise Exception("Generated content is different from existing")
        else:
            # copy generated directory
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(generated, dst)


def merge_configs(template: Dict[str, Any], patch: Dict[str, Any]):
    def merge(d1, d2):
        for key in d2:
            if key in d1 and isinstance(d1[key], dict) and isinstance(d2[key], dict):
                merge(d1[key], d2[key])
            else:
                d1[key] = d2[key]

    merge(template, patch)

    # we need this hack to preserve `model_input` as one line json in merged config
    model_input = json.dumps(template["model_metadata"]["example_model_input"])
    template["model_metadata"]["example_model_input"] = "<model_input>"
    merged = yaml.dump(template, default_flow_style=False, width=120)
    return merged.replace("<model_input>", model_input)


def apply_template(file: Path, variables: Dict[str, str]):
    env = Environment(
        loader=FileSystemLoader(file.parent),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    template = env.get_template(file.name)
    rendered_template = template.render(variables)
    file.with_suffix("").write_text(rendered_template)
    file.unlink()


def run(args):
    with open(args.config, "r") as file:
        data = yaml.safe_load(file)
        generates = {key: Generate(**value) for key, value in data.items()}

    for dst, generate in generates.items():
        process(Path(args.root) / dst, Path(args.templates), generate, args.only_check)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate trusses based on templates")
    parser.add_argument(
        "--only_check",
        default=False,
        action="store_true",
        help="Only check that nothing will be changed after generation",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="..",
        help="Root directory to put generated trusses to",
    )
    parser.add_argument(
        "--templates", type=str, default=".", help="Templates directory"
    )
    parser.add_argument(
        "--config", type=str, default="generate.yaml", help="Generate config"
    )

    run(parser.parse_args())


if __name__ == "__main__":
    main()
