# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""A native demo that does nothing outside its own window."""

import argparse
import tkinter as tk


class DemoApp:
    """Expose a visible button and a result for the test's independent oracle."""

    def __init__(self, root: tk.Tk, *, dark: bool = False, x: int = 80, y: int = 80) -> None:
        self.root = root
        self.completed = False
        root.title("Sightline demo | Yousif Lubbad")
        root.geometry(f"640x360+{x}+{y}")
        root.resizable(False, False)
        background, foreground = ("#14202e", "#ffffff") if dark else ("#f4f6fa", "#152332")
        self.canvas = tk.Canvas(
            root, width=640, height=360, background=background, highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(
            320, 55, text="Sightline local demo", fill=foreground, font=("Arial", 26)
        )
        self.status = self.canvas.create_text(
            320, 115, text="Ready", fill=foreground, font=("Arial", 24)
        )
        self.canvas.create_rectangle(
            200, 170, 440, 250, fill="#145db5", outline="#145db5", tags="action"
        )
        self.label = self.canvas.create_text(
            320, 210, text="Start", fill="white", font=("Arial", 30), tags="action"
        )
        self.canvas.create_text(320, 315, text="Yousif Lubbad", fill=foreground, font=("Arial", 16))
        self.canvas.tag_bind("action", "<Button-1>", self._complete)

    def _complete(self, _event: tk.Event) -> None:
        self.completed = True
        self.canvas.itemconfigure(self.status, text="Complete")
        self.canvas.itemconfigure(self.label, text="Finished")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Sightline's harmless local demo")
    parser.add_argument("--dark", action="store_true", help="Use a dark background")
    args = parser.parse_args()
    root = tk.Tk()
    DemoApp(root, dark=args.dark)
    root.mainloop()


if __name__ == "__main__":
    main()
