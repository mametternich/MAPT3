import os
import re
import shutil

delete = True
assess = False

models = []
models.append('fDys20_eta20-sc')
# models.append('fDys30_eta20-sc')
# models.append('fDys50_eta20-sc')

xdmf_path = os.path.join(os.path.dirname(__file__), 'XDMF-H5')
ttk_root = os.path.join(os.path.dirname(__file__), '1-Tessellation/TTK_outputs')
opti_path = os.path.join(os.path.dirname(__file__), '2-Persistence-analysis/OPTIMIZED/')


def format_size_mb(size_bytes):
    return f"{size_bytes / (1024 * 1024):.2f} MB"
def format_size_gb(size_bytes):
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
def matches_model_file(name, model):
    return bool(re.match(rf'^{re.escape(model)}(?:_\d+(?:_.*)?)?\.h5$', name))
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


# Option to delete models / frames
if delete:
    total_freed_bytes = 0
    for model in models:
        print(f"Deleting model: {model}")
        model_freed_bytes = 0

        # Delete from XDMF-H5
        for file_name in os.listdir(xdmf_path):
            if file_name.endswith('.h5') and matches_model_file(file_name, model):
                path_to_delete = os.path.join(xdmf_path, file_name)
                size_before = os.path.getsize(path_to_delete)
                os.remove(path_to_delete)
                model_freed_bytes += size_before
                total_freed_bytes += size_before
                print(f"  Deleted from XDMF-H5: {file_name} ({format_size_mb(size_before)})")

        # Delete from TTK_outputs
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
                    model_freed_bytes += size_before
                    total_freed_bytes += size_before
                    print(f"  Deleted TTK output: {model} frame {frame} ({format_size_mb(size_before)})")
                else:
                    size_before = os.path.getsize(entry_path)
                    os.remove(entry_path)
                    model_freed_bytes += size_before
                    total_freed_bytes += size_before
                    print(f"  Deleted TTK output: {model} frame {frame} ({format_size_mb(size_before)})")

        # Delete from OPTIMIZED
        for file_name in os.listdir(opti_path):
            if file_name.endswith('.h5') and matches_model_file(file_name, model):
                path_to_delete = os.path.join(opti_path, file_name)
                size_before = os.path.getsize(path_to_delete)
                os.remove(path_to_delete)
                model_freed_bytes += size_before
                total_freed_bytes += size_before
                print(f"  Deleted from OPTIMIZED: {file_name} ({format_size_mb(size_before)})")

        print(f"> Freed for model {model}: {format_size_mb(model_freed_bytes)}")

    print(f"Total freed: {format_size_gb(total_freed_bytes)}")


# Give overview of which models and which frames are in the data folders
for model in models:
    print(f"Model: {model}")
    xdmf_files = [f for f in os.listdir(xdmf_path) if f.endswith('.h5') and matches_model_file(f, model)]
    xdmf_frames = []
    for file_name in xdmf_files:
        match = re.match(rf'^{re.escape(model)}_(\d+)(?:_.*)?\.h5$', file_name)
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
