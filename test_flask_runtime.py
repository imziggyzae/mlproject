#!/usr/bin/env python3
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.app import app

print('TEMPLATE_FOLDER', app.template_folder)
client = app.test_client()

for path in ['/', '/predictdata']:
    response = client.get(path)
    print(path, response.status_code, response.mimetype)
    print(response.data.decode('utf-8')[:200])
    print('---')
