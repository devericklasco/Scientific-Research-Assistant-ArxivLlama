#!/bin/bash
pip install --upgrade pip
pip uninstall -y numpy faiss-cpu
pip install numpy==1.26.4
pip install --no-cache-dir faiss-cpu==1.7.4