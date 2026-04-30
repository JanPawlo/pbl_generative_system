from tkinter import Tk
from pathlib import Path
import shutil

from decal_GUI import DecalApp

DEFAULT_OUTPUT_DIR = Path("output_models")


def clear_output_models():
    if not DEFAULT_OUTPUT_DIR.exists():
        return
    for item in DEFAULT_OUTPUT_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def main():
    clear_output_models()
    root = Tk()
    DecalApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
