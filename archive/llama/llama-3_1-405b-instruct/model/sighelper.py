import fileinput
import sys

MODULE_FILE_PATH = (
    "/usr/local/lib/python3.11/dist-packages/vllm/executor/multiproc_gpu_executor.py"
)


def patch():
    # This is for SIGINT
    search_text = "signal.signal(signal.SIGINT, shutdown)"

    with fileinput.FileInput(MODULE_FILE_PATH, inplace=True) as file:
        for line in file:
            if search_text in line:
                line = "    # " + line.lstrip()
            sys.stdout.write(line)

    # This is for SIGTERM
    search_text = "signal.signal(signal.SIGTERM, shutdown)"

    with fileinput.FileInput(MODULE_FILE_PATH, inplace=True) as file:
        for line in file:
            if search_text in line:
                line = "    # " + line.lstrip()
            sys.stdout.write(line)
