"""
API keys template.
===================
Copy (or rename) this file to  "api_keys.py"  in the SAME folder
(so the path is  code/api_keys.py ) and paste your own keys below.
`api_keys.py` is listed in .gitignore and is never committed.

- APIKEY_GEMINI     : REQUIRED. Used by every VLM grader (Morphology,
                      Controller, Controller_MJX) and the proto VLM scripts.
                      Create one at https://aistudio.google.com/api-keys
- APIKEY_OPENROUTER : OPTIONAL. Only used by the experimental
                      code/proto/VLM/Gemma.py script (OpenRouter-hosted
                      models). Leave the placeholder if you do not run it.
                      Create one at https://openrouter.ai/keys
"""

APIKEY_GEMINI     = "put-your-gemini-api-key"
APIKEY_OPENROUTER = "put-your-openrouter-api-key"  # optional (proto/VLM/Gemma.py only)
