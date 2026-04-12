import json
import os
import re
import subprocess
import sys
import shutil
from typing import Any, Dict, List, Set, Optional

import pandas as pd
import streamlit as st
from rich import print as rprint
from streamlit.runtime.uploaded_file_manager import UploadedFile

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
CASE_FILE_EXTENSIONS = (".xlsx", ".xls", ".json", ".raw", ".m", ".mat", ".csv", ".dyr", ".dat", ".txt", ".seq", ".rcd")
CASE_PREVIEW_EXTENSIONS = (".xlsx", ".xls", ".json")
CASE_PREVIEW_SECTIONS = ("Bus", "Slack", "PV", "PQ", "Line", "Shunt", "Area")
CASE_PREVIEW_COLUMNS = ("idx", "name", "bus", "bus1", "bus2", "p0", "q0", "v0", "a0", "tap", "phi")


def resolve_python_executable() -> str:
    """
    Resolve the Python executable used for code execution/install.
    Priority:
    1) CONDA_ENV_PATH (supports env prefix, bin dir, python path, or conda path)
    2) CONDA_PREFIX/bin/python
    3) Current Streamlit interpreter (sys.executable)
    4) python3/python from PATH
    """
    candidates: List[str] = []
    conda_env_path = os.environ.get("CONDA_ENV_PATH")

    if conda_env_path:
        raw_path = os.path.abspath(os.path.expanduser(conda_env_path))
        if os.path.isdir(raw_path):
            if os.path.basename(raw_path) == "bin":
                candidates.append(os.path.join(raw_path, "python"))
            candidates.append(os.path.join(raw_path, "bin", "python"))
        else:
            basename = os.path.basename(raw_path)
            if basename.startswith("python"):
                candidates.append(raw_path)
            if basename in {"conda", "mamba", "micromamba"}:
                candidates.append(os.path.join(os.path.dirname(raw_path), "python"))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(os.path.join(os.path.abspath(os.path.expanduser(conda_prefix)), "bin", "python"))

    if sys.executable:
        candidates.append(sys.executable)

    for cmd in ("python3", "python"):
        from_path = shutil.which(cmd)
        if from_path:
            candidates.append(from_path)

    seen = set()
    unique_candidates = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique_candidates.append(path)

    for path in unique_candidates:
        if path and os.path.exists(path) and os.access(path, os.X_OK):
            return path

    # Final fallback so error message is explicit if execution fails.
    return "python3"


def detect_file_operations(code: str) -> List[str]:
    """Detect files that need to be read in the code"""
    file_patterns = [
        r'open\s*\(\s*["\']([^"\']+)["\']',  # open("filename")
        r'with\s+open\s*\(\s*["\']([^"\']+)["\']',  # with open("filename")
    ]
    
    detected_files = []
    for pattern in file_patterns:
        matches = re.findall(pattern, code)
        detected_files.extend(matches)
    
    return list(set(detected_files))  # Remove duplicates


def modify_code_file_paths(code: str, file_mappings: Dict[str, str]) -> str:
    """Modify code to use the correct file paths"""
    modified_code = code
    for original_file, new_path in file_mappings.items():
        modified_code = re.sub(
            rf'open\s*\(\s*["\']({re.escape(original_file)})["\']',
            f'open("{new_path}"',
            modified_code,
        )
        modified_code = re.sub(
            rf'with\s+open\s*\(\s*["\']({re.escape(original_file)})["\']',
            f'with open("{new_path}"',
            modified_code,
        )
    return modified_code


def save_uploaded_file(uploaded_file: UploadedFile, target_path: str):
    """Save uploaded file to target path"""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "wb") as f:
        if uploaded_file.type.startswith('text/'):
            f.write(uploaded_file.read())
        else:
            f.write(uploaded_file.getvalue())


def _normalize_case_preview_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _build_case_preview_records(records: List[Dict[str, Any]], max_rows: int) -> List[Dict[str, Any]]:
    preview: List[Dict[str, Any]] = []
    for record in records[:max_rows]:
        preview.append({key: _normalize_case_preview_value(value) for key, value in record.items()})
    return preview


def _summarize_case_preview_table(section: str, records: List[Dict[str, Any]]) -> List[str]:
    if not records:
        return []

    first_record = records[0]
    columns = list(first_record.keys())
    preview_columns = [column for column in CASE_PREVIEW_COLUMNS if column in columns]
    if not preview_columns:
        preview_columns = columns[:5]

    trimmed_records = [{key: record.get(key) for key in preview_columns} for record in records]
    preview = _build_case_preview_records(trimmed_records, max_rows=3)
    column_preview = ", ".join(columns[:8])
    if len(columns) > 8:
        column_preview += ", ..."

    return [
        f"- Sheet {section} columns: {column_preview}",
        f"- Sheet {section} sample rows: {json.dumps(preview, ensure_ascii=True)}",
    ]


def summarize_andes_case_file(file_path: str) -> str:
    """Build a compact prompt-ready preview for an ANDES Excel/JSON case file."""
    lower_path = file_path.lower()
    lines: List[str] = []

    try:
        if lower_path.endswith((".xlsx", ".xls")):
            with pd.ExcelFile(file_path) as excel_file:
                sections = [name for name in CASE_PREVIEW_SECTIONS if name in excel_file.sheet_names]
                if not sections:
                    sections = excel_file.sheet_names[:3]

                for section in sections[:5]:
                    df = pd.read_excel(file_path, sheet_name=section)
                    records = df.head(3).to_dict(orient="records")
                    lines.extend(_summarize_case_preview_table(section, records))
        elif lower_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            if isinstance(payload, dict):
                sections = [
                    name for name in CASE_PREVIEW_SECTIONS
                    if isinstance(payload.get(name), list) and payload.get(name)
                ]
                for section in sections[:5]:
                    raw_records = payload.get(section, [])
                    if not raw_records or not isinstance(raw_records[0], dict):
                        continue
                    lines.extend(_summarize_case_preview_table(section, raw_records))
    except Exception:
        return ""

    if not lines:
        return ""

    basename = os.path.basename(file_path)
    return "\n".join(
        [
            f"Selected ANDES case preview for {basename}:",
            "- Use the actual case rows below to resolve device idx values before modifying existing devices.",
            *lines,
        ]
    )


def build_uploaded_case_prompt_context(
    runtime_data_dir: str,
    user_input: str = "",
    active_case: Optional[Dict[str, str]] = None,
) -> str:
    """Return a concise prompt context block for the most relevant uploaded ANDES case."""
    runtime_files = get_files_in_directory(runtime_data_dir)
    previewable_files = [
        name for name in runtime_files
        if name.lower().endswith(CASE_PREVIEW_EXTENSIONS)
    ]
    if not previewable_files:
        return ""

    selected_file = ""
    normalized_user_input = (user_input or "").lower()
    for name in previewable_files:
        basename = os.path.basename(name)
        if basename.lower() in normalized_user_input:
            selected_file = basename
            break

    if not selected_file and active_case and active_case.get("source") == "uploaded":
        active_value = os.path.basename(active_case.get("value", ""))
        if active_value in previewable_files:
            selected_file = active_value

    if not selected_file and len(previewable_files) == 1:
        selected_file = previewable_files[0]

    if not selected_file:
        return ""

    preview_path = os.path.join(runtime_data_dir, selected_file)
    if not os.path.exists(preview_path):
        return ""

    return summarize_andes_case_file(preview_path)


def infer_andes_case_context(code: str, runtime_data_dir: str) -> Optional[Dict[str, str]]:
    """
    Infer the active ANDES case from generated code.
    Returns {"source": "builtin|uploaded|local", "value": "<case path or filename>"}.
    """
    if "andes.load" not in code:
        return None

    runtime_files = set(get_files_in_directory(runtime_data_dir))

    get_case_args = re.findall(r'andes\.get_case\(\s*["\']([^"\']+)["\']\s*\)', code)
    if get_case_args:
        return {"source": "builtin", "value": get_case_args[-1]}

    var_map: Dict[str, str] = {}
    join_assignments = re.findall(
        r'([A-Za-z_]\w*)\s*=\s*os\.path\.join\([^)]*["\']([^"\']+)["\']\)',
        code,
    )
    for var_name, file_name in join_assignments:
        if file_name.lower().endswith(CASE_FILE_EXTENSIONS):
            var_map[var_name] = file_name

    literal_assignments = re.findall(
        r'([A-Za-z_]\w*)\s*=\s*[rRuUbBfF]*["\']([^"\']+)["\']',
        code,
    )
    for var_name, value in literal_assignments:
        if value.lower().endswith(CASE_FILE_EXTENSIONS):
            var_map[var_name] = value

    load_var_refs = re.findall(r'andes\.load\(\s*([A-Za-z_]\w*)\s*(?:,|\))', code)
    for var_name in reversed(load_var_refs):
        inferred = var_map.get(var_name)
        if not inferred:
            continue
        basename = os.path.basename(inferred)
        source = "uploaded" if basename in runtime_files else "local"
        return {"source": source, "value": basename if source == "uploaded" else inferred}

    load_literal_refs = re.findall(
        r'andes\.load\(\s*[rRuUbBfF]*["\']([^"\']+)["\']\s*(?:,|\))',
        code,
    )
    if load_literal_refs:
        inferred = load_literal_refs[-1]
        basename = os.path.basename(inferred)
        source = "uploaded" if basename in runtime_files else "local"
        return {"source": source, "value": basename if source == "uploaded" else inferred}

    return None


def execute_python_code(code: str) -> str:
    """Execute Python code safely and return output"""
    try:
        python_bin_path = resolve_python_executable()
        rprint(f"Using Python executable: {python_bin_path}")
        session_data_dir = f"./code_executions/{st.session_state.session_id}/data"
        output_dir = os.path.join(session_data_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        images_before_run = list_image_files(session_data_dir)

        detected_files = detect_file_operations(code)
        file_mappings = {}
        missing_files = []
        for file_name in detected_files:
            file_name = file_name.split("/")[-1]
            target_path = f"./code_executions/{st.session_state.session_id}/data/{os.path.basename(file_name)}"
            file_mappings[file_name] = target_path
            if not os.path.exists(target_path):
                missing_files.append((file_name, target_path))

        if 'required_dependencies' in code:
            deps_pattern = r'#\s*required_dependencies:\s*([a-zA-Z0-9_,\s]+)'
            deps_match = re.search(deps_pattern, code)
            if deps_match:
                rprint(f"Extracted dependencies: {deps_match.group(1)}")
                for dep in deps_match.group(1).split('\n')[0].split(','):
                    try:
                        dep_name = dep.strip()
                        if not dep_name:
                            continue
                        subprocess.run(
                            [python_bin_path, "-m", "pip", "install", dep_name],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                    except Exception as e:
                        rprint(f"Failed to install dependency {dep_name}: {str(e)}")

        plot_capture_preamble = """
# Auto-capture matplotlib plt.show() for headless execution and store plots in output/.
try:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(_plot_output_dir, exist_ok=True)
    _existing_plots = [
        name for name in os.listdir(_plot_output_dir)
        if name.startswith("plot_") and name.endswith(".png")
    ]
    _plot_counter = len(_existing_plots)

    def _streamlit_safe_show(*args, **kwargs):
        global _plot_counter
        saved_paths = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            _plot_counter += 1
            plot_path = os.path.join(_plot_output_dir, f"plot_{_plot_counter}.png")
            fig.savefig(plot_path, bbox_inches="tight")
            saved_paths.append(plot_path)
        if saved_paths:
            print("Saved plot(s):")
            for path in saved_paths:
                print(f"- {path}")
        plt.close("all")

    plt.show = _streamlit_safe_show
except Exception:
    pass
"""

        code_to_execute = f"{plot_capture_preamble}\n{code}"
        code_hash = hash(code_to_execute)
        code_path = os.path.join("code_executions", f"{st.session_state.session_id}", "data", f"exec_code_{code_hash}.py")
        with open(code_path, "w") as f:
            f.write(code_to_execute)
        
        result = subprocess.run(
            [python_bin_path, f"exec_code_{code_hash}.py"],
            cwd=session_data_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        images_after_run = list_image_files(session_data_dir)
        copied_images = copy_new_images_to_output(
            images_before_run,
            images_after_run,
            base_directory=session_data_dir,
            output_directory=output_dir,
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n" + "=" * 50 + "\nSTDERR:\n" + result.stderr
            else:
                output = result.stderr
        if copied_images:
            image_summary = "\n".join(f"- {name}" for name in copied_images)
            if output:
                output += "\n"
            output += f"Copied image file(s) to output:\n{image_summary}"

        if result.returncode != 0:
            return f"Error (exit code {result.returncode}):\n{output}" if output else f"Process failed with exit code {result.returncode}"

        inferred_case = infer_andes_case_context(code, session_data_dir)
        if inferred_case:
            st.session_state.active_andes_case = inferred_case

        st.session_state.refresh_files = True
        return output if output else "Code executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (30 seconds)"
    except Exception as e:
        return f"Error: {str(e)}"


def resolve_generated_image_path(raw_path: str, session_data_dir: str) -> Optional[str]:
    """Resolve a generated image reference from execution output into an existing file path."""
    normalized = (raw_path or "").strip().strip('"').strip("'")
    if not normalized:
        return None

    candidate_paths = []
    if os.path.isabs(normalized):
        candidate_paths.append(normalized)
    else:
        candidate_paths.append(os.path.join(session_data_dir, normalized))
        candidate_paths.append(os.path.join(session_data_dir, "output", os.path.basename(normalized)))

    for candidate in candidate_paths:
        abs_candidate = os.path.abspath(candidate)
        if os.path.exists(abs_candidate) and abs_candidate.lower().endswith(IMAGE_EXTENSIONS):
            return abs_candidate

    return None


def extract_generated_image_paths(output: str, session_id: str) -> List[str]:
    """Extract generated image paths from execution output and resolve them on disk."""
    session_data_dir = os.path.abspath(os.path.join("code_executions", session_id, "data"))
    extracted_paths: List[str] = []
    capture_image_bullets = False

    for line in (output or "").splitlines():
        stripped = line.strip()
        if stripped in {"Saved plot(s):", "Copied image file(s) to output:"}:
            capture_image_bullets = True
            continue

        if capture_image_bullets and stripped.startswith("- "):
            resolved_path = resolve_generated_image_path(stripped[2:].strip(), session_data_dir)
            if resolved_path and resolved_path not in extracted_paths:
                extracted_paths.append(resolved_path)
            continue

        capture_image_bullets = False

    return extracted_paths


def get_files_in_directory(directory: str) -> List[str]:
    """Get list of files in a directory, excluding files starting with 'exec_code'"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return []
    files = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and not item.startswith('exec_code'):
            files.append(item)
    return sorted(files)


def list_image_files(base_directory: str) -> Set[str]:
    """List image files under base_directory, excluding the output folder."""
    image_files: Set[str] = set()
    for root, dirs, files in os.walk(base_directory):
        dirs[:] = [d for d in dirs if d != "output" and d != "__pycache__"]
        for file_name in files:
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                abs_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(abs_path, base_directory)
                image_files.add(rel_path)
    return image_files


def copy_new_images_to_output(
    images_before: Set[str],
    images_after: Set[str],
    base_directory: str,
    output_directory: str,
) -> List[str]:
    """Copy new image files generated during execution into output directory."""
    copied_files: List[str] = []
    new_images = sorted(images_after - images_before)
    for rel_path in new_images:
        src_path = os.path.join(base_directory, rel_path)
        if not os.path.exists(src_path):
            continue

        dest_base = os.path.basename(rel_path)
        dest_name, dest_ext = os.path.splitext(dest_base)
        dest_path = os.path.join(output_directory, dest_base)
        counter = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(output_directory, f"{dest_name}_{counter}{dest_ext}")
            counter += 1

        try:
            shutil.copy2(src_path, dest_path)
            copied_files.append(os.path.basename(dest_path))
        except Exception:
            continue
    return copied_files


def delete_file(file_path: str) -> bool:
    """Delete a file and return success status"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting file: {str(e)}")
        return False


def get_file_size(file_path: str) -> str:
    """Get human-readable file size"""
    try:
        size = os.path.getsize(file_path)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    except Exception:
        return "Unknown"


def display_file_section(section_title: str, directory: str, section_key: str):
    """Display a collapsible file section with delete/download functionality"""
    with st.expander(f"{section_title}", expanded=False):
        col_refresh, col_spacer = st.columns([1, 4])
        with col_refresh:
            if st.button("🔄", key=f"refresh_{section_key}", help="Refresh file list"):
                st.rerun()
        files = get_files_in_directory(directory)
        if not files:
            st.info(f"No files in {section_title.lower()}")
            return

        preview_state_key = f"active_preview_{section_key}"
        if preview_state_key not in st.session_state:
            st.session_state[preview_state_key] = None
        if st.session_state[preview_state_key] not in files:
            st.session_state[preview_state_key] = None

        st.markdown(f"**{len(files)} file(s) found:**")
        for file_name in files:
            file_path = os.path.join(directory, file_name)
            file_size = get_file_size(file_path)
            lower_name = file_name.lower()
            previewable = lower_name.endswith(('.txt', '.csv', '.json', '.log')) or lower_name.endswith(IMAGE_EXTENSIONS)
            is_active_preview = st.session_state.get(preview_state_key) == file_name

            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.markdown(f"📄 **{file_name}**")
                st.caption(f"Size: {file_size}")
            with col2:
                try:
                    with open(file_path, "rb") as f:
                        file_content = f.read()
                    st.download_button(
                        label="⬇️",
                        data=file_content,
                        file_name=file_name,
                        key=f"download_{section_key}_{file_name}",
                        help=f"Download {file_name}",
                    )
                except Exception:
                    st.error(f"Error reading {file_name}")
            with col3:
                if st.button("🗑️", key=f"delete_{section_key}_{file_name}", help=f"Delete {file_name}"):
                    if delete_file(file_path):
                        st.success(f"✅ Deleted {file_name}")
                        st.session_state.refresh_files = True
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to delete {file_name}")
            with col4:
                if previewable:
                    preview_button = "🙈" if is_active_preview else "👁️"
                    if st.button(preview_button, key=f"preview_{section_key}_{file_name}", help=f"Preview {file_name}"):
                        st.session_state[preview_state_key] = None if is_active_preview else file_name
                        st.rerun()

            if is_active_preview:
                try:
                    if lower_name.endswith(IMAGE_EXTENSIONS):
                        st.image(file_path, caption=f"Preview: {file_name}", use_container_width=True)
                    else:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        st.text_area(
                            f"Preview: {file_name}",
                            content[:1000] + ("..." if len(content) > 1000 else ""),
                            height=220,
                            key=f"preview_content_{section_key}_{file_name}",
                        )
                except Exception as e:
                    st.error(f"Error previewing {file_name}: {str(e)}")
            st.markdown("---")
