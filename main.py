from pathlib import Path
import subprocess
import sys


project_folder = Path(__file__).resolve().parent


def run_steps(script_names):
    # Check that all required scripts exist.
    for script_name in script_names:
        script_path = project_folder / script_name

        if not script_path.exists():
            print(f"Missing file: {script_name}")
            return False

    # Run each script using your current Python environment.
    for script_name in script_names:
        print(f"\nRunning {script_name}...", flush=True)

        result = subprocess.run(
            [sys.executable, str(project_folder / script_name)],
            cwd=project_folder
        )

        # Stop so the next step does not use old output.
        if result.returncode != 0:
            print(f"\n{script_name} failed. Check the error above.")
            return False

    return True


def main():
    while True:
        print("\n===== PDF RAG Bot =====")
        print("1. Ask a question")
        print("2. Build the index for the first time")
        print("3. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            run_steps(["retrieve.py", "answer.py"])

        elif choice == "2":
            print("\nThis runs extraction, chunking, embedding, and storage.")
            print("Embedding uses your Gemini API quota.")
            print("Use this only if the pdf_chunks collection does not exist.")

            confirmation = input("Type BUILD to continue: ").strip()

            if confirmation == "BUILD":
                completed = run_steps([
                    "extract.py",
                    "chunk.py",
                    "embed.py",
                    "store.py"
                ])

                if completed:
                    print("\nIndex created. Select option 1 to ask a question.")

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")