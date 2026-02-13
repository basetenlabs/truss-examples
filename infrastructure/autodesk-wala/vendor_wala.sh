#!/bin/bash
# Script to vendor WaLa source code into packages directory

echo "Downloading WaLa source code..."
cd autodesk-wala
curl -L https://github.com/AutodeskAILab/WaLa/archive/10abccb018c851337258236bc9f63ad5e3e348d3.zip -o wala.zip
unzip -q wala.zip
cp -r WaLa-10abccb018c851337258236bc9f63ad5e3e348d3/src packages/
rm -rf WaLa-10abccb018c851337258236bc9f63ad5e3e348d3 wala.zip
echo "Done! WaLa source vendored to packages/src/"
