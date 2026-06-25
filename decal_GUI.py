#!/usr/bin/env python3

import random
import threading
from pathlib import Path
from tkinter import (
    Tk,
    Canvas,
    Frame,
    Button,
    Label,
    Scale,
    Listbox,
    Entry,
    HORIZONTAL,
    DoubleVar,
    StringVar,
    messagebox,
    filedialog,
    END,
)

from PIL import Image, ImageTk

from decal_applier_v3 import apply_decals, distance
from json_parser import parse_scene_file

DEFAULT_DECAL_DIR = Path("decals/")
DEFAULT_OUTPUT_DIR = Path("output_models/")

# Default starting directories for file dialogs
_DESKTOP = Path.home() / "Desktop"
DEFAULT_SCENE_DIR   = _DESKTOP / "FishEngine" / "res" / "scenes"
DEFAULT_MODELS_DIR  = _DESKTOP / "FishEngine" / "res" / "models"
DEFAULT_DECALS_DIR  = _DESKTOP / "pbl_generative_system" / "decals"


def _existing_dir(path: Path) -> str:
    """Return path as string if it exists, else home directory string."""
    return str(path) if path.is_dir() else str(Path.home())


class DecalApp:
    def __init__(self, master):
        self.master = master
        master.title("Biome Decal Placer")
        master.geometry("1200x700")

        self.scene_context = None
        self.scene_data = None
        self.scene_text = ""
        self.scene_format = {"indent": 4, "newline": "\n"}
        self.objects = {}
        self.object_positions = []
        self.biomes = []
        self.selected_biome_idx = None

        self.object_textures = {}
        self.texture_refs = {}
        self.has_applied = False

        self.rng = random.Random()
        self.use_seed = False
        self.seed_value = None

        self.target_percent_var = DoubleVar(value=50.0)
        self.intensity_var = DoubleVar(value=0.7)
        self.near_threshold_var = DoubleVar(value=1.05)
        self.second_chance_var = DoubleVar(value=0.25)
        self.plane_var = StringVar(value="XZ")
        self.seed_var = StringVar(value="")

        self.create_menu()
        self.create_widgets()

        self.canvas_scale = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.drag_data = {"item": None, "biome_idx": None, "x": 0, "y": 0}

        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<Button-3>", self.add_biome_at_click)

    def create_menu(self):
        from tkinter import Menu

        menubar = Menu(self.master)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Scene JSON...", command=self.load_scene)
        filemenu.add_command(label="Set Decal Folder...", command=self.set_decal_folder)
        filemenu.add_command(label="Set Output Folder...", command=self.set_output_folder)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.master.config(menu=menubar)

        self.decal_dir = DEFAULT_DECAL_DIR
        self.output_dir = DEFAULT_OUTPUT_DIR

    def set_decal_folder(self):
        d = filedialog.askdirectory(title="Decal Source Folder")
        if d:
            self.decal_dir = Path(d)

    def set_output_folder(self):
        d = filedialog.askdirectory(title="Output Assets Folder")
        if d:
            self.output_dir = Path(d)

    def load_scene(self):
        filename = filedialog.askopenfilename(
            title="Open Scene JSON",
            initialdir=_existing_dir(DEFAULT_SCENE_DIR),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            self.scene_context = parse_scene_file(Path(filename))
            self.scene_data = self.scene_context.scene_data
            self.scene_text = self.scene_context.scene_text
            self.scene_format = self.scene_context.scene_format
            self.objects = self.scene_context.all_objects
            self.object_positions = [
                (obj.object_id, obj.world_pos) for obj in self.scene_context.object3d_items
            ]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load JSON:\n{e}")
            return

        self.object_textures.clear()
        self.texture_refs.clear()
        self.has_applied = False

        if not self.object_positions:
            messagebox.showwarning(
                "No Objects", "No Object3D/Platform items with transforms found in the scene."
            )
        self.redraw()

    def create_widgets(self):
        left_frame = Frame(self.master)
        left_frame.pack(side="left", fill="both", expand=True)

        plane_frame = Frame(left_frame)
        plane_frame.pack(side="top", fill="x")
        Label(plane_frame, text="View plane:").pack(side="left")
        for p in ["XZ", "XY", "YZ"]:
            Button(plane_frame, text=p, width=3, command=lambda v=p: self.set_plane(v)).pack(
                side="left", padx=2
            )

        self.canvas = Canvas(left_frame, bg="white", width=800, height=600)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        right_frame = Frame(self.master, width=350)
        right_frame.pack(side="right", fill="y")
        right_frame.pack_propagate(False)

        seed_frame = Frame(right_frame)
        seed_frame.pack(fill="x", padx=10, pady=(10, 5))
        Label(seed_frame, text="Random Seed:").pack(side="left")
        Entry(seed_frame, textvariable=self.seed_var, width=12).pack(side="left", padx=5)
        Button(seed_frame, text="Apply Seed", command=self.apply_seed).pack(side="left")

        Label(right_frame, text="Biomes").pack(pady=(10, 0))
        self.biome_listbox = Listbox(right_frame, height=6, exportselection=0)
        self.biome_listbox.pack(fill="x", padx=5, pady=5)
        self.biome_listbox.bind("<<ListboxSelect>>", self.on_biome_select)

        btn_frame = Frame(right_frame)
        btn_frame.pack(fill="x", padx=5)
        Button(btn_frame, text="Add Biome", command=self.prepare_add_biome).pack(side="left", padx=2)
        Button(btn_frame, text="Remove Biome", command=self.remove_biome).pack(side="left", padx=2)

        Label(right_frame, text="Intensity (0-1):").pack(anchor="w", padx=10)
        Scale(
            right_frame,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=HORIZONTAL,
            variable=self.intensity_var,
            command=lambda v: self.on_slider_change(),
        ).pack(fill="x", padx=10)

        Label(right_frame, text="Near Threshold (1.0-2.0):").pack(anchor="w", padx=10)
        Scale(
            right_frame,
            from_=1.0,
            to=2.0,
            resolution=0.01,
            orient=HORIZONTAL,
            variable=self.near_threshold_var,
            command=lambda v: self.on_slider_change(),
        ).pack(fill="x", padx=10)

        Label(right_frame, text="Second Chance (0-1):").pack(anchor="w", padx=10)
        Scale(
            right_frame,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=HORIZONTAL,
            variable=self.second_chance_var,
            command=lambda v: self.on_slider_change(),
        ).pack(fill="x", padx=10)

        Label(right_frame, text="Biome Decals:").pack(anchor="w", padx=10)
        decal_list_frame = Frame(right_frame)
        decal_list_frame.pack(fill="x", padx=10)
        self.decal_listbox = Listbox(decal_list_frame, height=4, exportselection=0)
        self.decal_listbox.pack(side="left", fill="x", expand=True)
        decal_scroll = __import__("tkinter").Scrollbar(decal_list_frame, orient="vertical",
                                                       command=self.decal_listbox.yview)
        decal_scroll.pack(side="right", fill="y")
        self.decal_listbox.config(yscrollcommand=decal_scroll.set)

        decal_btn_frame = Frame(right_frame)
        decal_btn_frame.pack(fill="x", padx=10, pady=(2, 0))
        Button(decal_btn_frame, text="Add Decal...", command=self.choose_decal).pack(side="left", padx=2)
        Button(decal_btn_frame, text="Remove Selected", command=self.remove_decal).pack(side="left", padx=2)

        Label(right_frame, text="Target % affected (0-100):").pack(anchor="w", padx=10, pady=(20, 0))
        Scale(
            right_frame,
            from_=0,
            to=100,
            orient=HORIZONTAL,
            variable=self.target_percent_var,
            command=lambda v: None,
        ).pack(fill="x", padx=10)

        self.apply_btn = Button(
            right_frame,
            text="APPLY DECALS",
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.run_apply,
        )
        self.apply_btn.pack(fill="x", padx=10, pady=20)

        Button(right_frame, text="Reset Preview", command=self.reset_preview).pack(fill="x", padx=10)

    def set_plane(self, plane):
        self.plane_var.set(plane)
        self.redraw()

    def apply_seed(self):
        seed_str = self.seed_var.get().strip()
        if seed_str == "":
            self.use_seed = False
            self.seed_value = None
            self.rng = random.Random()
        else:
            try:
                self.seed_value = int(seed_str)
                self.use_seed = True
                self.rng = random.Random(self.seed_value)
            except ValueError:
                messagebox.showerror("Invalid Seed", "Seed must be an integer or blank for random.")
                return
        messagebox.showinfo("Seed Applied", f"Seed set to: {self.seed_value if self.use_seed else 'random'}")

    def reset_preview(self):
        self.object_textures.clear()
        self.texture_refs.clear()
        self.has_applied = False
        self.redraw()

    def world_to_canvas(self, wx, wy):
        cx = self.canvas.winfo_width() / 2
        cy = self.canvas.winfo_height() / 2
        x = cx + (wx + self.canvas_offset_x) * self.canvas_scale
        y = cy - (wy + self.canvas_offset_y) * self.canvas_scale
        return x, y

    def canvas_to_world(self, cx, cy):
        midx = self.canvas.winfo_width() / 2
        midy = self.canvas.winfo_height() / 2
        wx = (cx - midx) / self.canvas_scale - self.canvas_offset_x
        wy = -(cy - midy) / self.canvas_scale - self.canvas_offset_y
        return wx, wy

    def get_plane_coords(self, pos):
        x, y, z = pos
        plane = self.plane_var.get()
        if plane == "XZ":
            return x, z
        if plane == "XY":
            return x, y
        return y, z

    def redraw(self, event=None):
        self.canvas.delete("all")
        if not self.object_positions:
            return

        all_wx = []
        all_wy = []
        for obj_id, pos in self.object_positions:
            wx, wy = self.get_plane_coords(pos)
            all_wx.append(wx)
            all_wy.append(wy)
        if not all_wx:
            return

        min_x, max_x = min(all_wx), max(all_wx)
        min_y, max_y = min(all_wy), max(all_wy)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        self.canvas_offset_x = -center_x
        self.canvas_offset_y = -center_y

        for obj_id, pos in self.object_positions:
            wx, wy = self.get_plane_coords(pos)
            cx, cy = self.world_to_canvas(wx, wy)
            size = 16
            if obj_id in self.object_textures and self.object_textures[obj_id] is not None:
                photo = self.object_textures[obj_id]
                self.canvas.create_image(cx, cy, image=photo, anchor="center", tags="obj")
            else:
                if self.biomes:
                    idx = self.select_biome_preview(pos)
                else:
                    idx = None
                if idx is not None and idx < len(self.colours):
                    colour = self.colours[idx]
                else:
                    colour = "#CCCCCC"
                self.canvas.create_rectangle(
                    cx - size, cy - size, cx + size, cy + size, fill=colour, outline="black", tags="obj"
                )

        for i, biome in enumerate(self.biomes):
            bx, by = self.get_plane_coords(biome["pos"])
            bcx, bcy = self.world_to_canvas(bx, by)
            r = 10
            colour = self.colours[i % len(self.colours)]
            self.canvas.create_oval(
                bcx - r,
                bcy - r,
                bcx + r,
                bcy + r,
                fill=colour,
                outline="black",
                width=2,
                tags=("biome", f"biome_{i}"),
            )
            self.canvas.create_text(
                bcx,
                bcy,
                text=str(i),
                fill="white",
                font=("Arial", 8, "bold"),
                tags=("biome", f"biome_{i}"),
            )

    def select_biome_preview(self, world_pos):
        if not self.biomes:
            return None
        dists = [(distance(world_pos, b["pos"]), i) for i, b in enumerate(self.biomes)]
        dists.sort(key=lambda x: x[0])
        return dists[0][1]

    colours = ["#E74C3C", "#3498DB", "#2ECC71", "#F1C40F", "#9B59B6", "#1ABC9C", "#E67E22", "#95A5A6"]

    def add_biome_at_click(self, event):
        if not self.object_positions:
            return
        wx, wy = self.canvas_to_world(event.x, event.y)
        plane = self.plane_var.get()
        if plane == "XZ":
            pos = [wx, 0.0, wy]
        elif plane == "XY":
            pos = [wx, wy, 0.0]
        else:
            pos = [0.0, wx, wy]
        new_biome = {
            "pos": pos,
            "intensity": 0.7,
            "near_threshold": 1.05,
            "second_chance": 0.25,
            "decals": [],
        }
        self.biomes.append(new_biome)
        self.update_biome_listbox()
        self.redraw()
        self.select_biome(len(self.biomes) - 1)

    def prepare_add_biome(self):
        messagebox.showinfo("Add Biome", "Right-click on the canvas to place the new biome.")

    def remove_biome(self):
        if self.selected_biome_idx is not None and 0 <= self.selected_biome_idx < len(self.biomes):
            del self.biomes[self.selected_biome_idx]
            self.selected_biome_idx = None
            self.update_biome_listbox()
            self.redraw()
            self.clear_sliders()

    def update_biome_listbox(self):
        self.biome_listbox.delete(0, END)
        for i, b in enumerate(self.biomes):
            pos = b["pos"]
            self.biome_listbox.insert(END, f"Biome {i}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        if self.selected_biome_idx is not None and self.selected_biome_idx < len(self.biomes):
            self.biome_listbox.selection_set(self.selected_biome_idx)

    def on_biome_select(self, event):
        sel = self.biome_listbox.curselection()
        if sel:
            self.select_biome(sel[0])

    def select_biome(self, idx):
        if idx < 0 or idx >= len(self.biomes):
            return
        self.selected_biome_idx = idx
        b = self.biomes[idx]
        self.intensity_var.set(b["intensity"])
        self.near_threshold_var.set(b["near_threshold"])
        self.second_chance_var.set(b["second_chance"])
        self._refresh_decal_listbox(b["decals"])
        self.update_biome_listbox()

    def on_slider_change(self, *args):
        if self.selected_biome_idx is not None and 0 <= self.selected_biome_idx < len(self.biomes):
            b = self.biomes[self.selected_biome_idx]
            b["intensity"] = self.intensity_var.get()
            b["near_threshold"] = self.near_threshold_var.get()
            b["second_chance"] = self.second_chance_var.get()
            self.redraw()

    def _refresh_decal_listbox(self, decals):
        self.decal_listbox.delete(0, END)
        for d in decals:
            self.decal_listbox.insert(END, d)

    def choose_decal(self):
        f = filedialog.askopenfilename(
            title="Select Decal Image",
            initialdir=_existing_dir(DEFAULT_DECALS_DIR),
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if f:
            name = Path(f).name
            if self.selected_biome_idx is not None and 0 <= self.selected_biome_idx < len(self.biomes):
                decals = self.biomes[self.selected_biome_idx]["decals"]
                if name not in decals:
                    decals.append(name)
                self._refresh_decal_listbox(decals)

    def remove_decal(self):
        if self.selected_biome_idx is None or self.selected_biome_idx >= len(self.biomes):
            return
        sel = self.decal_listbox.curselection()
        if not sel:
            return
        decals = self.biomes[self.selected_biome_idx]["decals"]
        idx = sel[0]
        if 0 <= idx < len(decals):
            decals.pop(idx)
        self._refresh_decal_listbox(decals)

    def clear_sliders(self):
        self.intensity_var.set(0.0)
        self.near_threshold_var.set(1.0)
        self.second_chance_var.set(0.0)
        self.decal_listbox.delete(0, END)

    def on_press(self, event):
        item = self.canvas.find_closest(event.x, event.y)
        tags = self.canvas.gettags(item)
        for tag in tags:
            if tag.startswith("biome_"):
                idx = int(tag.split("_")[1])
                self.drag_data["item"] = item
                self.drag_data["biome_idx"] = idx
                self.drag_data["x"] = event.x
                self.drag_data["y"] = event.y
                self.select_biome(idx)
                return
        self.selected_biome_idx = None
        self.update_biome_listbox()
        self.clear_sliders()

    def on_release(self, event):
        self.drag_data = {"item": None, "biome_idx": None, "x": 0, "y": 0}

    def on_motion(self, event):
        if self.drag_data["item"] is None:
            return
        idx = self.drag_data["biome_idx"]
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        dwx = dx / self.canvas_scale
        dwy = -dy / self.canvas_scale
        pos = self.biomes[idx]["pos"]
        plane = self.plane_var.get()
        if plane == "XZ":
            pos[0] += dwx
            pos[2] += dwy
        elif plane == "XY":
            pos[0] += dwx
            pos[1] += dwy
        else:
            pos[1] += dwx
            pos[2] += dwy
        self.redraw()
        self.update_biome_listbox()

    def zoom(self, event):
        factor = 1.1 if event.delta > 0 else 0.9
        self.canvas_scale *= factor
        self.redraw()

    def on_canvas_resize(self, event):
        self.redraw()

    def run_apply(self):
        if not self.scene_data:
            messagebox.showwarning("No Scene", "Load a scene JSON first.")
            return
        if not self.biomes:
            messagebox.showwarning("No Biomes", "Add at least one biome.")
            return

        if not self.use_seed:
            seed_str = self.seed_var.get().strip()
            if seed_str != "":
                try:
                    self.seed_value = int(seed_str)
                    self.use_seed = True
                    self.rng = random.Random(self.seed_value)
                except ValueError:
                    messagebox.showerror("Invalid Seed", "Seed must be an integer or blank for random.")
                    return
            else:
                self.rng = random.Random()

        biomes = self.biomes
        target_percent = self.target_percent_var.get() / 100.0
        decal_dir = self.decal_dir
        output_dir = self.output_dir

        if not hasattr(self, "assets_dir") or not Path(self.assets_dir).is_dir():
            d = filedialog.askdirectory(
                title="Select Original Assets (models) Folder",
                initialdir=_existing_dir(DEFAULT_MODELS_DIR),
            )
            if not d:
                return
            self.assets_dir = d
        assets_dir = Path(self.assets_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        self.master.config(cursor="watch")
        self.apply_btn.config(state="disabled")

        rng_state = self.rng.getstate()

        def task():
            try:
                result = apply_decals(
                    scene_data=self.scene_data,
                    scene_text=self.scene_text,
                    biomes=biomes,
                    target_percent=target_percent,
                    assets_dir=assets_dir,
                    output_dir=output_dir,
                    decal_dir=decal_dir,
                    rng_state=rng_state,
                )
                self.result_message = (
                    f"Done! {result['target_count']} objects modified.\n"
                    f"Output: {result['output_dir']}\nModified JSON: {result['save_path']}"
                )
                self.generated_textures = result["generated_textures"]
                self.modified_set = result["modified_set"]
                self.source_models = result["source_models"]
                self.master.after(0, self.apply_finished, True, None)
            except Exception as e:
                self.master.after(0, self.apply_finished, False, str(e))

        threading.Thread(target=task, daemon=True).start()

    def apply_finished(self, success, error_msg):
        self.master.config(cursor="")
        self.apply_btn.config(state="normal")
        if success:
            self.has_applied = True
            self.load_texture_previews()
            self.redraw()
            messagebox.showinfo("Success", self.result_message)
        else:
            messagebox.showerror("Error", f"Application failed:\n{error_msg}")

    def load_texture_previews(self):
        self.object_textures.clear()
        self.texture_refs.clear()
        thumbnail_size = 32

        if hasattr(self, "generated_textures"):
            for obj_id, tex_path in self.generated_textures.items():
                try:
                    img = Image.open(tex_path)
                    img = img.resize((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.object_textures[obj_id] = photo
                    self.texture_refs[obj_id] = photo
                except Exception as e:
                    print(f"Failed to load preview for {obj_id}: {e}")

        if hasattr(self, "source_models") and hasattr(self, "modified_set"):
            assets_dir = Path(self.assets_dir) if hasattr(self, "assets_dir") else None
            if assets_dir:
                for obj_id, model_name in self.source_models.items():
                    if obj_id in self.modified_set:
                        continue
                    png_path = assets_dir / model_name / f"{model_name}.png"
                    if png_path.exists():
                        try:
                            img = Image.open(png_path)
                            img = img.resize((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            self.object_textures[obj_id] = photo
                            self.texture_refs[obj_id] = photo
                        except Exception as e:
                            print(f"Failed to load original preview for {obj_id}: {e}")


if __name__ == "__main__":
    root = Tk()
    app = DecalApp(root)
    root.mainloop()