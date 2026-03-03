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