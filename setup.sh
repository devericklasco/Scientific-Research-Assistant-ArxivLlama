#!/bin/bash
pip install --upgrade pip
pip uninstall -y faiss-cpu  # Remove any broken installs
pip install --no-cache-dir faiss-cpu==1.7.4