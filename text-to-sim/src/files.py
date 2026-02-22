import os
import re
import subprocess
import sys
import shutil
from typing import Dict, List

import streamlit as st
from rich import print as rprint
from streamlit.runtime.uploaded_file_manager import UploadedFile


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


def execute_python_code(code: str) -> str:
    """Execute Python code safely and return output"""
    try:
        python_bin_path = resolve_python_executable()
        rprint(f"Using Python executable: {python_bin_path}")

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

        code_hash = hash(code)
        code_path = os.path.join("code_executions", f"{st.session_state.session_id}", "data", f"exec_code_{code_hash}.py")
        with open(code_path, "w") as f:
            f.write(code)
        
        result = subprocess.run(
            [python_bin_path, f"exec_code_{code_hash}.py"],
            cwd=f"./code_executions/{st.session_state.session_id}/data",
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n" + "=" * 50 + "\nSTDERR:\n" + result.stderr
            else:
                output = result.stderr

        if result.returncode != 0:
            return f"Error (exit code {result.returncode}):\n{output}" if output else f"Process failed with exit code {result.returncode}"

        st.session_state.refresh_files = True
        return output if output else "Code executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (30 seconds)"
    except Exception as e:
        return f"Error: {str(e)}"


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
        st.markdown(f"**{len(files)} file(s) found:**")
        for file_name in files:
            file_path = os.path.join(directory, file_name)
            file_size = get_file_size(file_path)
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
                if file_name.endswith(('.txt', '.csv', '.json', '.log')):
                    if st.button("👁️", key=f"preview_{section_key}_{file_name}", help=f"Preview {file_name}"):
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            st.text_area(
                                f"Preview: {file_name}",
                                content[:1000] + ("..." if len(content) > 1000 else ""),
                                height=200,
                                key=f"preview_content_{section_key}_{file_name}",
                            )
                        except Exception as e:
                            st.error(f"Error previewing {file_name}: {str(e)}")
            st.markdown("---")
