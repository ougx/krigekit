# test_dll_load.py
import ctypes
import os

dll_path = os.path.abspath("../src/krigekit/kriging.dll")
print(f"Loading: {dll_path}")

try:
    # Try basic loading
    dll = ctypes.CDLL(dll_path, winmode=0)
    print("✓ DLL loaded successfully")
    
    # Try to find the initialize function
    init_func = getattr(dll, 'krige_initialize', None)
    if init_func:
        print("✓ Found initialize function")
    else:
        print("✗ Initialize function not found")
        # List all exports
        import subprocess
        result = subprocess.run(['dumpbin', '/exports', dll_path], capture_output=True, text=True)
        print(result.stdout[:500])
        
except Exception as e:
    print(f"✗ Failed: {e}")
