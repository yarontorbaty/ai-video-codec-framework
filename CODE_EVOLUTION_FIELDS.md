# Code Evolution Fields - Dashboard Integration Guide

## Overview
The experiment results now include a `code_evolution` field that tracks LLM code generation attempts, adoptions, and GitHub commits.

## DynamoDB Schema

### Main Table: `ai-video-codec-experiments`

**New Field:** `code_evolution` (JSON string)

The `code_evolution` field is a JSON string that needs to be parsed. It contains the following structure:

```json
{
  "code_changed": boolean,
  "version": number,
  "status": string,
  "improvement": string,
  "summary": string,
  "deployment_status": string,
  "github_committed": boolean,
  "github_commit_hash": string | null
}
```

## Field Definitions

### `code_changed` (boolean)
- **Type:** Boolean
- **Description:** Whether the LLM successfully generated code that was adopted
- **Values:**
  - `true` - Code was generated, tested, and adopted
  - `false` - Code was not adopted (rejected, failed, or not generated)

### `version` (number)
- **Type:** Integer
- **Description:** The codec version number after this experiment
- **Values:**
  - `0` - No evolution yet
  - `1, 2, 3...` - Incremental version numbers
- **Note:** Only increments when code is adopted

### `status` (string)
- **Type:** String
- **Description:** The status of the code evolution attempt
- **Possible Values:**
  - `"adopted"` - Code was successfully adopted
  - `"rejected"` - Code was generated but not better than current
  - `"test_failed"` - Code failed validation or testing
  - `"skipped"` - No code generation attempted
  - `"no_code"` - LLM didn't generate code
  - `"failed"` - Code generation failed

### `improvement` (string)
- **Type:** String
- **Description:** Human-readable description of the improvement
- **Example Values:**
  - `"N/A"` - No improvement (code rejected/failed)
  - `"15.3% bitrate reduction"` - For adopted code
  - `"First implementation"` - For version 1

### `summary` (string)
- **Type:** String
- **Description:** Concise summary of what happened with code evolution
- **Example Values:**
  - `"LLM evolved codec to v2 - 15.3% bitrate reduction, 1.85 Mbps, 4.2x compression"`
  - `"Code generated but rejected: Bitrate 2.10 Mbps not better than current 1.85 Mbps"`
  - `"No code generation attempted"`
  - `"Code testing failed or produced invalid results"`

### `deployment_status` (string)
- **Type:** String
- **Description:** Whether the code was deployed to the system
- **Possible Values:**
  - `"deployed"` - Code is now active
  - `"not_deployed"` - Code was not deployed

### `github_committed` (boolean)
- **Type:** Boolean
- **Description:** Whether the code was successfully committed to GitHub
- **Values:**
  - `true` - Code pushed to GitHub
  - `false` - Not committed (failed or not attempted)

### `github_commit_hash` (string | null)
- **Type:** String or null
- **Description:** The Git commit hash if successfully committed
- **Values:**
  - `"a13aa5fe..."` - Full commit SHA
  - `null` - No commit

## Dashboard Display Recommendations

### Status Badge
Display based on `code_changed` and `status`:
```javascript
if (code_evolution.code_changed) {
  return '🎉 Code Evolved'; // Green badge
} else if (code_evolution.status === 'rejected') {
  return '⏭️ Code Rejected'; // Orange badge
} else if (code_evolution.status === 'test_failed') {
  return '❌ Test Failed'; // Red badge
} else if (code_evolution.status === 'skipped') {
  return '⏸️ Skipped'; // Gray badge
}
```

### Version Display
Show version only if code_changed is true:
```javascript
if (code_evolution.code_changed) {
  return `v${code_evolution.version}`;
}
```

### Summary Display
Show the summary with appropriate styling:
```javascript
<div class="code-evolution-summary">
  {code_evolution.summary}
</div>
```

### GitHub Link
If code was committed, show a link:
```javascript
if (code_evolution.github_committed && code_evolution.github_commit_hash) {
  return (
    <a href={`https://github.com/yarontorbaty/ai-video-codec-framework/commit/${code_evolution.github_commit_hash}`} 
       target="_blank">
      View Commit {code_evolution.github_commit_hash.substring(0, 7)}
    </a>
  );
}
```

## Example Lambda Code (for SSR)

```python
# In lambda/index_ssr.py - when processing experiment data

experiment_item = {
    # ... existing fields ...
}

# Parse code_evolution if present
if 'code_evolution' in item:
    try:
        code_evolution = json.loads(item['code_evolution'])
        experiment_item['code_evolution'] = code_evolution
        
        # Add evolution badge
        if code_evolution.get('code_changed'):
            experiment_item['evolution_badge'] = f"✅ v{code_evolution['version']}"
            experiment_item['evolution_class'] = 'evolved'
        elif code_evolution.get('status') == 'rejected':
            experiment_item['evolution_badge'] = "⏭️ Rejected"
            experiment_item['evolution_class'] = 'rejected'
        else:
            experiment_item['evolution_badge'] = f"⏸️ {code_evolution.get('status', 'No change').title()}"
            experiment_item['evolution_class'] = 'no-change'
    except json.JSONDecodeError:
        pass  # Field not present or invalid
```

## Example HTML/CSS

```html
<tr class="experiment-row">
  <td>{experiment_id}</td>
  <td>{timestamp}</td>
  <td>{bitrate}</td>
  <td>
    <span class="evolution-badge {evolution_class}">
      {evolution_badge}
    </span>
  </td>
  <td class="evolution-summary">{code_evolution.summary}</td>
  <td>
    {github_committed && (
      <a href="https://github.com/yarontorbaty/ai-video-codec-framework/commit/{commit_hash}" 
         target="_blank" 
         class="github-link">
        <i class="fa fa-github"></i> {commit_hash_short}
      </a>
    )}
  </td>
</tr>
```

```css
.evolution-badge {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.85em;
  font-weight: 600;
}

.evolution-badge.evolved {
  background-color: #10b981;
  color: white;
}

.evolution-badge.rejected {
  background-color: #f59e0b;
  color: white;
}

.evolution-badge.no-change {
  background-color: #6b7280;
  color: white;
}

.evolution-summary {
  font-size: 0.9em;
  color: #4b5563;
  max-width: 400px;
}

.github-link {
  color: #3b82f6;
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.github-link:hover {
  text-decoration: underline;
}
```

## Testing

To test if the fields are working, run a query:

```bash
aws dynamodb scan \
  --table-name ai-video-codec-experiments \
  --limit 5 \
  --region us-east-1 \
  --query 'Items[*].[experiment_id.S, code_evolution.S]' \
  --output json
```

Or check in Python:
```python
import boto3
import json

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('ai-video-codec-experiments')

response = table.scan(Limit=1)
if response['Items']:
    item = response['Items'][0]
    if 'code_evolution' in item:
        evolution = json.loads(item['code_evolution'])
        print(json.dumps(evolution, indent=2))
```

## Notes

1. **Backward Compatibility:** Older experiments won't have this field. Always check if the field exists before parsing.

2. **JSON Parsing:** The `code_evolution` field is stored as a JSON string, so it needs to be parsed with `JSON.parse()` in JavaScript or `json.loads()` in Python.

3. **Real-time Updates:** This field is populated during experiment execution, so it will only appear in new experiments (after deployment).

4. **GitHub Integration:** The `github_committed` and `github_commit_hash` fields will only be populated if:
   - Code was adopted (`code_changed` = true)
   - GitHub credentials are configured
   - The push to GitHub was successful

5. **Null Values:** Some fields may be `null` if not applicable (e.g., `github_commit_hash` when not committed).

