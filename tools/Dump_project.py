"""
dump_project.py

Dumps specific parts of the CPX_Vision project into separate text files:
  1. core_dump.txt   -> contents of the 'core' folder
  2. gui_dump.txt     -> contents of the 'gui' folder
  3. theme_dump.txt   -> contents of theme.py
  4. main_dump.txt    -> contents of main.py

Usage:
    python dump_project.py
    python dump_project.py --root "C:\\Appl\\Python projects\\CPX_Vision\\cpx_recognizer"
"""

import os
import argparse

TEXT_EXTENSIONS = {
    ".py", ".json", ".txt", ".yaml", ".yml", ".md", ".cfg", ".ini",
    ".toml", ".csv", ".xml", ".html", ".css", ".js"
}

SKIP_DIRS = {
    "__pycache__", ".git", "venv", ".venv", "node_modules",
    "data", "models", "manual_capture_temp", "relabel_staging",
    "runs", "images", "output", "turntable", "yolo_dataset"
}

DEFAULT_ROOT_DIR = r"C:\Appl\Python projects\CPX_Vision\cpx_recognizer"


def dump_folder(root_dir, target_name, out_path):
    """Dump all matching files inside root_dir/target_name into out_path."""
    target_dir = os.path.join(root_dir, target_name)
    if not os.path.isdir(target_dir):
        print(f"WARNING: folder not found, skipping: {target_dir}")
        return

    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"DUMP OF FOLDER: {target_dir}\n")
        out.write("=" * 80 + "\n\n")

        for dirpath, dirnames, filenames in os.walk(target_dir):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in TEXT_EXTENSIONS:
                    continue
                fpath = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(fpath, root_dir)

                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except Exception as e:
                    out.write(f"----- {rel_path} (error reading: {e}) -----\n\n")
                    continue

                out.write(f"----- {rel_path} -----\n")
                out.write(content)
                out.write("\n\n")

    print(f"Done. Dump written to: {out_path} ({os.path.getsize(out_path)} bytes)")


def dump_file(root_dir, target_name, out_path):
    """Dump a single file (e.g. main.py, theme.py) into out_path."""
    fpath = os.path.join(root_dir, target_name)
    if not os.path.isfile(fpath):
        print(f"WARNING: file not found, skipping: {fpath}")
        return

    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"DUMP OF FILE: {fpath}\n")
        out.write("=" * 80 + "\n\n")
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            out.write(f"(error reading: {e})\n")
        else:
            out.write(content)

    print(f"Done. Dump written to: {out_path} ({os.path.getsize(out_path)} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=DEFAULT_ROOT_DIR,
        help=f"Folder to scan (default: {DEFAULT_ROOT_DIR})"
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    if not os.path.isdir(root_dir):
        print(f"ERROR: folder not found: {root_dir}")
        return

    cwd = os.getcwd()

    dump_folder(root_dir, "core", os.path.join(cwd, "core_dump.txt"))
    dump_folder(root_dir, "gui", os.path.join(cwd, "gui_dump.txt"))
    dump_file(root_dir, "theme.py", os.path.join(cwd, "theme_dump.txt"))
    dump_file(root_dir, "main.py", os.path.join(cwd, "main_dump.txt"))


if __name__ == "__main__":
    main()
