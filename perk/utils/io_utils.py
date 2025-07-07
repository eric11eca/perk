import os
import json

def read_json(path, mode="r", **kwargs):
    return json.loads(read_file(path, mode=mode, **kwargs))

def write_json(data, path):
    return write_file(json.dumps(data, indent=2), path)

def to_jsonl(data):
    return json.dumps(data).replace("\n", "")

def read_file(path, mode="r", **kwargs):
    """Reads a file and returns its content."""
    with open(path, mode=mode, **kwargs) as f:
        return f.read()

def write_file(data, path, mode="w", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        f.write(data)

def read_jsonl(path, mode="r", **kwargs):
    ls = []
    with open(path, mode, **kwargs) as f:
        for line in f:
            try:
                data = json.loads(line)
                ls.append(data)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e.msg} at line {e.lineno} column {e.colno} (char {e.pos})")
                context_radius = 20
                start = max(e.pos - context_radius, 0)
                end = e.pos + context_radius
                print("Error context:", repr(line[start:end]))
                continue
    return ls

def write_jsonl(data, path, mode="w"):
    assert isinstance(data, list)
    lines = [to_jsonl(elem) for elem in data]
    write_file("\n".join(lines) + "\n", path, mode=mode)

def write_generations(generations, output_dir, log_name):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, log_name)
    write_jsonl(generations, save_path)

def write_metrics(metrics, output_dir, log_name):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, log_name)
    write_json(metrics, save_path)
