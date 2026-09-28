import os
import sys
import subprocess

class OSModule:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    last_accessed_dir = base_dir

    @staticmethod
    def _safe_open(target):
        if hasattr(os, 'startfile'):
            os.startfile(target)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', target])
        else:
            subprocess.Popen(['xdg-open', target])

    @staticmethod
    def resolve_path(raw_path):
        if not raw_path:
            return OSModule.last_accessed_dir
        p = str(raw_path).lower().strip().replace('"', '').replace("'", "")
        # Common drive patterns and voice recognition typos like 'derive'
        if p in ["d drive", "d derive", "d:", "d:\\", "d", "drive d", "d /"]:
            return "D:\\" if sys.platform == "win32" else OSModule.base_dir
        if p in ["c drive", "c derive", "c:", "c:\\", "c", "drive c", "c /"]:
            return "C:\\" if sys.platform == "win32" else OSModule.base_dir
        if p in ["brain", "brain folder", "workspace", "here", "this"]:
            return OSModule.base_dir
        if os.path.exists(raw_path):
            return os.path.abspath(raw_path)
        # Check relative to base directory
        rel_brain = os.path.join(OSModule.base_dir, raw_path)
        if os.path.exists(rel_brain):
            return rel_brain
        return raw_path

    @staticmethod
    def open_camera():
        print("[THINKING] Executing OS Command: Open Camera")
        try:
            if sys.platform == "win32" and hasattr(os, 'startfile'):
                os.startfile("microsoft.windows.camera:")
            return "Camera opened successfully."
        except Exception as e:
            return f"Failed to open camera: {e}"

    @staticmethod
    def open_file(filepath):
        resolved = OSModule.resolve_path(filepath)
        print(f"[THINKING] Executing OS Command: Open File -> {resolved}")
        try:
            OSModule._safe_open(resolved)
            return f"File '{filepath}' opened successfully."
        except Exception as e:
            return f"Failed to open file: {e}"

    @staticmethod
    def create_folder(folder_path):
        resolved = OSModule.resolve_path(folder_path)
        print(f"[THINKING] Executing OS Command: Create Folder -> {resolved}")
        try:
            os.makedirs(resolved, exist_ok=True)
            OSModule.last_accessed_dir = resolved
            return f"Folder '{folder_path}' created successfully."
        except Exception as e:
            return f"Failed to create folder: {e}"

    @staticmethod
    def open_directory(dir_path):
        resolved = OSModule.resolve_path(dir_path)
        print(f"[THINKING] Executing OS Command: Open Directory -> {resolved}")
        try:
            if os.path.exists(resolved):
                OSModule._safe_open(resolved)
                OSModule.last_accessed_dir = resolved
                return f"Directory '{resolved}' opened successfully in File Explorer."
            else:
                return f"Directory '{resolved}' does not exist."
        except Exception as e:
            return f"Failed to open directory: {e}"

    @staticmethod
    def open_explorer(target_path=None):
        if target_path is None:
            target_path = OSModule.base_dir
        resolved = OSModule.resolve_path(target_path)
        print(f"[THINKING] Executing OS Command: Open File Explorer -> {resolved}")
        try:
            if os.path.exists(resolved):
                OSModule._safe_open(os.path.abspath(resolved))
                OSModule.last_accessed_dir = resolved
                return f"File Explorer opened at '{resolved}'."
            return f"Path '{resolved}' does not exist."
        except Exception as e:
            return f"Failed to open File Explorer: {e}"

    @staticmethod
    def list_directory(dir_path=None):
        if not dir_path or dir_path.strip() in ["it", "this", "here", "current", "available"]:
            resolved = OSModule.last_accessed_dir
        else:
            resolved = OSModule.resolve_path(dir_path)
            
        print(f"[THINKING] Executing OS Command: List Directory Contents -> {resolved}")
        try:
            if not os.path.exists(resolved):
                return f"Directory '{resolved}' was not found on the system."
            
            items = os.listdir(resolved)
            folders = [f for f in items if os.path.isdir(os.path.join(resolved, f))]
            files = [f for f in items if os.path.isfile(os.path.join(resolved, f))]
            
            OSModule.last_accessed_dir = resolved
            
            resp = f"Contents of '{resolved}':\n"
            if folders:
                folder_str = ", ".join(folders[:15]) + ("..." if len(folders) > 15 else "")
                resp += f"• Folders ({len(folders)}): {folder_str}\n"
            if files:
                file_str = ", ".join(files[:20]) + ("..." if len(files) > 20 else "")
                resp += f"• Files ({len(files)}): {file_str}\n"
            if not folders and not files:
                resp += "The folder is currently empty."
            return resp.strip()
        except Exception as e:
            return f"Could not read contents of '{resolved}': {e}"

    @staticmethod
    def open_app(app_name):
        print(f"[THINKING] Executing OS Command: Open App -> {app_name}")
        allowed_apps = {
            "notepad": "notepad.exe" if sys.platform == "win32" else "nano",
            "calc": "calc.exe" if sys.platform == "win32" else "bc",
            "calculator": "calc.exe" if sys.platform == "win32" else "bc",
            "explorer": "explorer.exe" if sys.platform == "win32" else OSModule.base_dir,
            "file explorer": "explorer.exe" if sys.platform == "win32" else OSModule.base_dir,
            "camera": "microsoft.windows.camera:" if sys.platform == "win32" else "",
            "browser": "https://www.google.com"
        }
        cmd = allowed_apps.get(app_name.lower().strip())
        if cmd:
            try:
                OSModule._safe_open(cmd)
                return f"{app_name.capitalize()} opened successfully."
            except Exception as e:
                return f"Failed to open {app_name}: {e}"
        return f"App '{app_name}' is not in allowed safe applications."
