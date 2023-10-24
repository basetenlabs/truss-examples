"""
truss predict -d '{"prompt": "A heavily constructed solarpunk bridge over a canyon at sunset"}' | python show.py
"""

import base64
import json
import os
import sys

resp = sys.stdin.read()
image = json.loads(resp)["data"]
img = base64.b64decode(image)

file_name = f'{image[-10:].replace("/", "")}.jpeg'
img_file = open(file_name, "wb")
img_file.write(img)
img_file.close()
os.system(f"open {file_name}")
