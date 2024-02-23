# runner.py - Place this in your current working directory

import sys
import importlib.util
import os

def run_script(script_path, args=[]):
    # Backup the original sys.argv
    original_argv = sys.argv
    
    # Determine module name and path
    module_name = os.path.basename(script_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    
    try:
        # Mimic command line arguments
        sys.argv = [script_path] + args
        
        # Load and execute the module
        spec.loader.exec_module(module)
        
        # If the module has a main function, call it
        if hasattr(module, 'main'):
            module.main()
    finally:
        # Restore the original sys.argv
        sys.argv = original_argv

if __name__ == "__main__":
    # Pass all arguments except the first (the script name itself) to run_script
    run_script('./jobs/fashion_mnist_train/job.py', sys.argv[1:])
