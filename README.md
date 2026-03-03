# growl-demographic-inference

Pipeline that groups chat logs by `user_id` and uses Claude (Anthropic API) to infer:
- age segment
- gender
- IAB content category/subcategory

It saves results to a CSV file (resume mode supported).

## Requirements
- Python 3.10+ recommended

## Install
```bash
pip install -r requirements.txt
```

## API key setup (required to run inference)
This repository does NOT include any API keys.
Create a file named .env in the project root directory and add the following line:
ANTHROPIC_API_KEY=YOUR_KEY_HERE

Do NOT commit the .env file.
You can copy .env.example to .env and replace YOUR_KEY_HERE with your own key.

## Run
python demographic_inference.py

## Notes
Input/output CSV files are ignored by git via .gitignore to avoid leaking sensitive data.
