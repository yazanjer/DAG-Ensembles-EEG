import subprocess
import sys
import time
import os

# Define the exact names of your three implementation files
SCRIPTS_TO_RUN = [
    "implementation_with_test_cssp.py",
    "implementation_with_test_cssp_svmskernals.py",
    "implementation_with_test.py"
]

def run_batch():
    print("=" * 70)
    print("Starting batch execution of EEG BCI classification pipelines...")
    print("=" * 70)
    
    total_start_time = time.time()
    
    for script in SCRIPTS_TO_RUN:
        print("\n" + "=" * 50)
        print(f"🚀 EXECUTING: {script}")
        print("=" * 50)
        
        # Check if the file actually exists before trying to run it
        if not os.path.isfile(script):
            print(f"\n[ERROR] File '{script}' not found in the current directory.")
            print("Skipping to the next script...\n")
            continue
            
        script_start_time = time.time()
        
        try:
            # sys.executable ensures the script runs in the exact same Python 
            # environment (and virtual env) that is running this master script.
            subprocess.run([sys.executable, script], check=True)
            
            elapsed = time.time() - script_start_time
            print(f"\n✅ [SUCCESS] {script} finished in {elapsed:.2f} seconds.")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ [ERROR] {script} crashed with exit code {e.returncode}.")
            print("Continuing to the next script...")
            
        except KeyboardInterrupt:
            print("\n🛑 [INTERRUPTED] Batch execution stopped by user.")
            sys.exit(1)

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 70)
    print(f"🏁 ALL SCRIPTS COMPLETED. Total time: {total_elapsed:.2f} seconds.")
    print("=" * 70)

if __name__ == "__main__":
    run_batch()