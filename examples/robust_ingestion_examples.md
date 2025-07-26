# Robust Ingestion API - Options Parsing Fix

## Problem Solved
The ingestion API previously failed with "Failed to parse options as JSON: string" when users passed plain strings or improperly formatted JSON. The API now robustly handles multiple input formats.

## Supported Input Formats

### 1. JSON Format (Recommended)
```bash
# Valid JSON with metadata wrapper
curl -X POST "http://localhost:8000/ingest/upload" \
  -H "soeid: user123" \
  -F "file=@document.pdf" \
  -F "options={\"metadata\": {\"source\": \"finance\", \"type\": \"report\"}}"

# Valid JSON without metadata wrapper
curl -X POST "http://localhost:8000/ingest/upload" \
  -H "soeid: user123" \
  -F "file=@document.pdf" \
  -F "options={\"source\": \"finance\", \"type\": \"report\"}"

# JSON primitive values
curl -X POST "http://localhost:8000/ingest/upload" \
  -H "soeid: user123" \
  -F "file=@document.pdf" \
  -F "options=\"financial_report\""
```

### 2. Key-Value Pairs
```bash
# Comma-separated pairs
curl -X POST "http://localhost:8000/ingest/upload" \
  -H "soeid: user123" \
  -F "file=@document.pdf" \
  -F "options=source=finance,type=report,year=2024"

# Semicolon-separated pairs
curl -X POST "http://localhost:8000/ingest/upload" \
  -H "soeid: user123" \
  -F "file=@document.pdf" \
  -F "options=source=finance;type=report;year=2024"

# Single key-value pair
curl -X POST "http://localhost:8000/ingest/upload" \
  -H "soeid: user123" \
  -F "file=@document.pdf" \
  -F "options=source=finance"
```

### 3. Plain String (Fallback)
```bash
# Plain string description - this was causing the original error
curl -X POST "http://localhost:8000/ingest/upload" \
  -H "soeid: user123" \
  -F "file=@document.pdf" \
  -F "options=financial_report_q3_2024"

# Any string works now
curl -X POST "http://localhost:8000/ingest/upload" \
  -H "soeid: user123" \
  -F "file=@document.pdf" \
  -F "options=string"  # This was the original error case
```

### 4. No Options (Always Worked)
```bash
# No options parameter
curl -X POST "http://localhost:8000/ingest/upload" \
  -H "soeid: user123" \
  -F "file=@document.pdf"
```

## Python Client Examples

### Using requests library
```python
import requests
import json

def upload_with_json_options(file_path, metadata):
    """Upload with JSON options (recommended)."""
    url = "http://localhost:8000/ingest/upload"
    headers = {"soeid": "user123"}
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'options': json.dumps({"metadata": metadata})}
        response = requests.post(url, headers=headers, files=files, data=data)
        return response.json()

def upload_with_keyvalue_options(file_path, **kwargs):
    """Upload with key-value options."""
    url = "http://localhost:8000/ingest/upload"
    headers = {"soeid": "user123"}
    
    # Convert kwargs to key=value,key=value format
    options = ','.join(f"{k}={v}" for k, v in kwargs.items())
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'options': options}
        response = requests.post(url, headers=headers, files=files, data=data)
        return response.json()

def upload_with_string_options(file_path, description):
    """Upload with plain string options."""
    url = "http://localhost:8000/ingest/upload"
    headers = {"soeid": "user123"}
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'options': description}  # Plain string
        response = requests.post(url, headers=headers, files=files, data=data)
        return response.json()

# Example usage
result1 = upload_with_json_options("report.pdf", {"source": "finance", "type": "report"})
result2 = upload_with_keyvalue_options("report.pdf", source="finance", type="report")
result3 = upload_with_string_options("report.pdf", "financial_report_q3")
```

## How the Robust Parsing Works

The API now uses a multi-strategy approach:

1. **JSON Parsing**: Tries to parse as valid JSON first
2. **Key-Value Parsing**: Handles `key=value,key=value` format
3. **Single Pair Parsing**: Handles single `key=value` format  
4. **String Fallback**: Treats any string as a description

## Metadata Results

Different input formats produce different metadata structures:

```python
# Input: '{"metadata": {"source": "finance"}}'
# Result: {"source": "finance"}

# Input: 'source=finance,type=report'
# Result: {"source": "finance", "type": "report"}

# Input: 'financial_report'
# Result: {"description": "financial_report"}

# Input: 'source=finance'
# Result: {"source": "finance"}
```

## Benefits

✅ **No More Errors**: The "Failed to parse options as JSON: string" error is eliminated
✅ **Backward Compatible**: All existing API calls continue to work
✅ **Flexible Input**: Supports multiple input formats for user convenience
✅ **Graceful Fallback**: Always produces valid metadata, never fails
✅ **Better Logging**: Clear logs show which parsing strategy was used

## Testing Your Fix

You can now test with the exact command that was failing:

```bash
# This command previously failed but now works
curl -X POST "http://localhost:8000/ingest/upload" \
  -H "soeid: user123" \
  -F "file=@your_document.pdf" \
  -F "options=string"
```

The API will now gracefully handle this and create metadata: `{"description": "string"}`
