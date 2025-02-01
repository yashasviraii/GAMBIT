import sys
import os
import argparse
import subprocess

sys.path.append(os.path.abspath("GAMBIT"))

#import GAMBIT

# Run the workflow

def main():
    parser = argparse.ArgumentParser(description="GAMBIT Workflow CLI")
    
    parser.add_argument("command", type=str, choices=["run"], help="Command to execute")

    parser.add_argument("--model", nargs="+", required=True, help="Specify one or more model names")
    
    args = parser.parse_args()
    
    selected_models = args.model
    
    if args.command == "run":
        script_path = os.path.join("GAMBIT", "run.py")  # Path to the target script
        
        # Prepare command list to pass model names
        command = ["python", script_path, "--model"] + selected_models

        # Run GAMBIT/run.py and pass the model arguments
        subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
