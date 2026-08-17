import os
import re
import shutil
from collections import defaultdict

delete = 0      # will delete the files in models
assess = 1      # will assess all models in the data folders and print memory usage

models = []
# models.append('fDys40-sc')
# models.append('fDys40_eta20-sc')
models.append('fDys30-sc')

xdmf_path = os.path.join(os.path.dirname(__file__), 'XDMF-H5')
ttk_root = os.path.join(os.path.dirname(__file__), '1-Tessellation/TTK_outputs')
opti_path = os.path.join(os.path.dirname(__file__), '2-Persistence-analysis/OPTIMIZED/')


def format_size_mb(size_bytes):
    return f"{size_bytes / (1024 * 1024):.3f} MB"
def format_size_gb(size_bytes):
    return f"{size_bytes / (1024 * 1024 * 1024):.3f} GB"
def matches_model_file(name, model):
    # match files like model_123.h5 or model_123.xdmf (optional extra suffixes)
    return bool(re.match(rf'^{re.escape(model)}(?:_\d+(?:_.*)?)?\.(?:h5|xdmf)$', name))
def matches_ttk_model(entry_name, model):
    return bool(re.search(rf'(^|_)' + re.escape(model) + r'_(\d+)(?:\.[^/]+)?$', entry_name))


def get_ttk_dirs(base_path):
    if not os.path.isdir(base_path):
        return []
    return [
        os.path.join(base_path, entry_name)
        for entry_name in sorted(os.listdir(base_path))
        if os.path.isdir(os.path.join(base_path, entry_name)) and re.match(r'^p\d+$', entry_name)
    ]


def get_dir_size_bytes(path):
    """Return total size in bytes for files under `path` (walks recursively)."""
    total = 0
    if not os.path.exists(path):
        return 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, filename))
            except OSError:
                pass
    return total


# Option to delete models / frames
if delete:
    total_freed_bytes = 0
    for model in models:
        print(f"Deleting model: {model}")
        model_freed_bytes = 0

        # Track deleted size per frame: frame_freed[frame_id] = cumulative_bytes
        frame_freed = defaultdict(int)

        # Delete from XDMF-H5 (aggregate per frame) -- handle .h5 and .xdmf
        for file_name in os.listdir(xdmf_path):
            if (file_name.endswith('.h5') or file_name.endswith('.xdmf')) and matches_model_file(file_name, model):
                path_to_delete = os.path.join(xdmf_path, file_name)
                size_before = os.path.getsize(path_to_delete)
                frame_match = re.match(rf'^{re.escape(model)}_(\d+)(?:_.*)?\.(?:h5|xdmf)$', file_name)
                frame = frame_match.group(1) if frame_match else '?'
                os.remove(path_to_delete)
                frame_freed[frame] += size_before
                model_freed_bytes += size_before
                total_freed_bytes += size_before

        # Delete from TTK_outputs (aggregate per frame for both files and directories)
        for ttk_dir in get_ttk_dirs(ttk_root):
            for entry_name in os.listdir(ttk_dir):
                entry_path = os.path.join(ttk_dir, entry_name)
                m = re.search(rf'(^|_)' + re.escape(model) + r'_(\d+)(?:\.[^/]+)?$', entry_name)
                if not m:
                    continue
                frame = m.group(2)

                if os.path.isdir(entry_path):
                    size_before = 0
                    for dirpath, _, filenames in os.walk(entry_path):
                        for filename in filenames:
                            size_before += os.path.getsize(os.path.join(dirpath, filename))
                    shutil.rmtree(entry_path)
                    frame_freed[frame] += size_before
                    model_freed_bytes += size_before
                    total_freed_bytes += size_before
                else:
                    size_before = os.path.getsize(entry_path)
                    os.remove(entry_path)
                    frame_freed[frame] += size_before
                    model_freed_bytes += size_before
                    total_freed_bytes += size_before

        # Delete from OPTIMIZED (aggregate per frame)
        for file_name in os.listdir(opti_path):
            if file_name.endswith('.h5') and matches_model_file(file_name, model):
                path_to_delete = os.path.join(opti_path, file_name)
                size_before = os.path.getsize(path_to_delete)
                frame_match = re.match(rf'^{re.escape(model)}_(\d+)(?:_.*)?\.h5$', file_name)
                frame = frame_match.group(1) if frame_match else '?'
                os.remove(path_to_delete)
                frame_freed[frame] += size_before
                model_freed_bytes += size_before
                total_freed_bytes += size_before

        # Print a single entry for each deleted frame (sorted numerically if possible)
        for frame in sorted(frame_freed.keys(), key=lambda x: int(x) if x.isdigit() else x):
            bytes_deleted = frame_freed[frame]
            print(f"  Deleted frame {frame} ({format_size_mb(bytes_deleted)})")

        print(f"> Freed for model {model}: {format_size_mb(model_freed_bytes)}")

    print(f"Total freed: {format_size_gb(total_freed_bytes)}")
    print()


# Give overview of which models and which frames are in the data folders
for model in models:
    print(f"Model: {model}")
    xdmf_files = [f for f in os.listdir(xdmf_path) if (f.endswith('.h5') or f.endswith('.xdmf')) and matches_model_file(f, model)]
    xdmf_frames = []
    for file_name in xdmf_files:
        match = re.match(rf'^{re.escape(model)}_(\d+)(?:_.*)?\.(?:h5|xdmf)$', file_name)
        if match:
            xdmf_frames.append(int(match.group(1)))
    xdmf_frames = sorted(set(xdmf_frames))
    print(f"  frames found in XDMF-H5:     {xdmf_frames}")

    ttk_frames = []
    for ttk_dir in get_ttk_dirs(ttk_root):
        for entry_name in os.listdir(ttk_dir):
            match = re.search(rf'(^|_)' + re.escape(model) + r'_(\d+)(?:\.[^/]+)?$', entry_name)
            if match:
                ttk_frames.append(int(match.group(2)))
    ttk_frames = sorted(set(ttk_frames))
    print(f"  frames found in TTK_outputs: {ttk_frames}")

    opti_frames = []
    opti_files = [f for f in os.listdir(opti_path) if f.endswith('.h5') and matches_model_file(f, model)]
    for file_name in opti_files:
        match = re.match(rf'^{re.escape(model)}_(\d+)(?:_.*)?\.h5$', file_name)
        if match:
            opti_frames.append(int(match.group(1)))
    opti_frames = sorted(set(opti_frames))
    print(f"  frames found in OPTIMIZED:   {opti_frames}")

if assess:
    print()
    print("Other models found per directory:")

    directory_patterns = [
        ("XDMF-H5", xdmf_path, [rf'^(.+)_\d+\.(?:h5|xdmf)$']),
        ("TTK_outputs", ttk_root, [rf'^[^_]+_(.+)_\d+\.csv$', rf'^[^_]+_(.+)_\d+$']),
        ("OPTIMIZED", opti_path, [rf'^(.+)_\d+_.*\.h5$']),
    ]

    for directory_name, directory_path, patterns in directory_patterns:
        directory_models = set()
        if directory_name == "TTK_outputs":
            for ttk_dir in get_ttk_dirs(directory_path):
                for entry_name in os.listdir(ttk_dir):
                    for pattern in patterns:
                        match = re.match(pattern, entry_name)
                        if match:
                            directory_models.add(match.group(1))
                            break
        else:
            for entry_name in os.listdir(directory_path):
                for pattern in patterns:
                    match = re.match(pattern, entry_name)
                    if match:
                        directory_models.add(match.group(1))
                        break

        extra_models = sorted(directory_models.difference(models))
        summary = ", ".join(extra_models) if extra_models else "none"
        print(f"  {directory_name} ({len(extra_models)} models): {summary}")

    # Compute and print total sizes for the three data folders
    print()
    print("Calculating data size summary:")

    xdmf_total_bytes = get_dir_size_bytes(xdmf_path)
    opti_total_bytes = get_dir_size_bytes(opti_path)
    ttk_total_bytes = 0
    for ttk_dir in get_ttk_dirs(ttk_root):
        ttk_total_bytes += get_dir_size_bytes(ttk_dir)
    combined_bytes = xdmf_total_bytes + ttk_total_bytes + opti_total_bytes

    print(f"  XDMF-H5:     {format_size_gb(xdmf_total_bytes)} ({format_size_mb(xdmf_total_bytes)})")
    print(f"  TTK_outputs: {format_size_gb(ttk_total_bytes)} ({format_size_mb(ttk_total_bytes)})")
    print(f"  OPTIMIZED:   {format_size_gb(opti_total_bytes)} ({format_size_mb(opti_total_bytes)})")
    print(f"  Combined:    {format_size_gb(combined_bytes)} ({format_size_mb(combined_bytes)})")
