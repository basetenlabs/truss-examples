"""Regenerate the gRPC Python stubs from ``greeter.proto``.

Run with ``uv run grpc-greeter-codegen``. Writes
``greeter_pb2.py`` and ``greeter_pb2_grpc.py`` into this package, then rewrites
the generated grpc module's flat ``import greeter_pb2`` to a package-relative
import so the committed stubs import cleanly as part of ``grpc_greeter``.
"""

import pathlib
import subprocess
import sys

_PACKAGE_DIR = pathlib.Path(__file__).parent
_PROTO = _PACKAGE_DIR.parent.parent / "greeter.proto"


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={_PROTO.parent}",
            f"--python_out={_PACKAGE_DIR}",
            f"--grpc_python_out={_PACKAGE_DIR}",
            str(_PROTO),
        ],
        check=True,
    )
    # grpc_tools emits `import greeter_pb2`, which only resolves if the package
    # directory itself is on sys.path. Rewrite it to a package-relative import.
    grpc_module = _PACKAGE_DIR / "greeter_pb2_grpc.py"
    grpc_module.write_text(
        grpc_module.read_text().replace(
            "import greeter_pb2 as greeter__pb2",
            "from . import greeter_pb2 as greeter__pb2",
        )
    )
    print(f"Generated stubs in {_PACKAGE_DIR}")


if __name__ == "__main__":
    main()
