.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "backend"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:OLLAMA_KEEP_ALIVE = "2h"

 RUN 1  prompt=strict  cs/ov/k=256/32/4  58 questions
  pass                 36  [none]  
  retrieval_miss        2  [retrieval]  q010, q095
  partial_retrieval     3  [retrieval]  q012, q025, q047
  ranking_failure       4  [retrieval]  q034, q045, q067, q080
  over_refusal         11  [generation]  q003, q009, q018, q024, q030, q070, q077, q078, q081, q088, q091
  mixed_refusal         2  [generation]  q036, q083
  failures: 22 of 58  |  retrieval 9, generation 13, error 0

RUN 2  prompt=basic  cs/ov/k=256/32/4  58 questions
  pass                 42  [none]  
  retrieval_miss        2  [retrieval]  q010, q095
  partial_retrieval     3  [retrieval]  q012, q025, q047
  ranking_failure       3  [retrieval]  q034, q067, q080
  incorrect_answer      4  [generation]  q005, q006, q018, q024
  failed_to_decline     4  [generation]  q048, q051, q073, q074
  failures: 16 of 58  |  retrieval 8, generation 8, error 0
 