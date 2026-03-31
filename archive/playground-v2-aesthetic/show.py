"""
truss predict -d '{"prompt": "A heavily constructed solarpunk bridge over a canyon at sunset", "steps": 50}' | python show.py
"""

import base64
import json
import os
import sys

resp = sys.stdin.read()
image = json.loads(resp)["output"]
img = base64.b64decode(image)

img_file = open("playground.png", "wb")
img_file.write(img)
img_file.close()
os.system("open playground.png")
