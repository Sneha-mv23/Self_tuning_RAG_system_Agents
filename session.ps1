.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "backend"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:OLLAMA_KEEP_ALIVE = "2h"
 

  
git add .
git status
git commit -m "feat: search-strategy simulation on the free retrieval space"
git push origin main

