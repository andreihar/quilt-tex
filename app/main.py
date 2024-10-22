import customtkinter as ctk
from PIL import Image
from tkinter import filedialog
import numpy as np
from skimage import io, util, color, img_as_ubyte
import math, heapq, threading

def randomPatch(texture, patchSize, overlapSize=None, output=None, x=None, y=None, i=None, j=None):
    h, w = texture.shape[:2]
    i = np.random.randint(h - patchSize)
    j = np.random.randint(w - patchSize)
    return texture[i:i+patchSize, j:j+patchSize]

def ssd(patch, patchSize, overlap, output, x, y):
    return (np.sum((patch[:, :overlap] - output[y:y+patchSize, x:x+overlap])**2) if x > 0 else 0) + \
           (np.sum((patch[:overlap, :] - output[y:y+overlap, x:x+patchSize])**2) if y > 0 else 0) - \
           (np.sum((patch[:overlap, :overlap] - output[y:y+overlap, x:x+overlap])**2) if x > 0 and y > 0 else 0) 

def bestPatch(texture, patchSize, overlap, output, x, y):
    h, w = texture.shape[:2]
    errors = np.array([[ssd(texture[i:i+patchSize, j:j+patchSize], patchSize, overlap, output, x, y) for j in range(w - patchSize)] for i in range(h - patchSize)])
    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+patchSize, j:j+patchSize]

def cutPatch(patch, overlapSize, output, x, y):
    def path(errors):
        h, w = errors.shape
        queue = [(error, [i]) for i,error in enumerate(errors[0])]
        heapq.heapify(queue)
        visited = set()
        while queue:
            error, path = heapq.heappop(queue)
            depth = len(path)
            if depth == h:
                return path
            for delta in -1, 0, 1:
                next = path[-1] + delta
                if 0 <= next < w:
                    if (depth, next) not in visited:
                        heapq.heappush(queue, (error + errors[depth, next], path + [next]))
                        visited.add((depth, next))
    patch = patch.copy()
    dy, dx = patch.shape[:2]
    cut = np.zeros_like(patch, dtype=bool)
    if x > 0:
        for i, j in enumerate(path(np.sum((patch[:,:overlapSize]-output[y:y+dy,x:x+overlapSize])**2, axis=2))):
            cut[i, :j] = True
    if y > 0:
        for j, i in enumerate(path(np.sum((patch[:overlapSize,:]-output[y:y+overlapSize,x:x+dx])**2, axis=2).T)):
            cut[:i, j] = True
    np.copyto(patch, output[y:y+dy, x:x+dx], where=cut)
    return patch

def corrOver(texture, corrTex, corrTar, patchSize, overlap, output, x, y, alpha=0.1):
    h, w = texture.shape[:2]
    errors = np.zeros((h - patchSize, w - patchSize))
    corrTarP = corrTar[y:y+patchSize, x:x+patchSize]
    di, dj = corrTarP.shape
    for i in range(h - patchSize):
        for j in range(w - patchSize):
            errors[i, j] = alpha * (np.sum(ssd(texture[i:i+di, j:j+dj], patchSize, overlap, output, x, y))) + (1 - alpha) * np.sum((corrTex[i:i+di, j:j+dj] - corrTarP)**2)
    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+di, j:j+dj]

def process(texture, patchSize, overlapSize=None):
    texture = util.img_as_float(texture)
    h, w = texture.shape[:2]
    if len(texture.shape) == 2:
        texture = np.stack([texture]*3, axis=-1)
    if overlapSize is None:
        patchH = math.ceil((5 * h) / patchSize)
        patchW = math.ceil((5 * w) / patchSize)
        output = np.zeros((patchH * patchSize, patchW * patchSize, texture.shape[2]))
    else:
        patchH = math.ceil((5 * h - patchSize) / (patchSize - overlapSize)) + 1
        patchW = math.ceil((5 * w - patchSize) / (patchSize - overlapSize)) + 1
        output = np.zeros(((patchH * patchSize) - (patchH - 1) * overlapSize, (patchW * patchSize) - (patchW - 1) * overlapSize, texture.shape[2]))
    return texture, patchH, patchW, output, h, w

def general_method(texture, patchSize, overlapSize, patchSelectionFunc):
    texture, patchH, patchW, output, h, w = process(texture, patchSize, overlapSize)
    for i in range(patchH):
        for j in range(patchW):
            y = i * (patchSize - overlapSize) if overlapSize else i * patchSize
            x = j * (patchSize - overlapSize) if overlapSize else j * patchSize
            patch = patchSelectionFunc(texture, patchSize, overlapSize, output, x, y, i, j)
            output[y:y+patchSize, x:x+patchSize] = patch
    return output[:5*h, :5*w]

def bestPatchSelection(texture, patchSize, overlapSize, output, x, y, i, j):
    if i == 0 and j == 0:
        return randomPatch(texture, patchSize)
    else:
        return bestPatch(texture, patchSize, overlapSize, output, x, y)

def cutPatchSelection(texture, patchSize, overlapSize, output, x, y, i, j):
    if i == 0 and j == 0:
        return randomPatch(texture, patchSize)
    else:
        return cutPatch(bestPatch(texture, patchSize, overlapSize, output, x, y), overlapSize, output, x, y)

def transfer(texture, target, patchSize, overlap, alpha):
    if texture.shape[2] == 4:
        texture = color.rgba2rgb(texture)
    if target.shape[2] == 4:
        target = color.rgba2rgb(target)
    target = util.img_as_float(target)[:,:,:3]
    h, w = target.shape[:2]
    output = np.zeros_like(target)
    for i in range((math.ceil((h - patchSize) / (patchSize - overlap)) + 1 or 1)):
        for j in range((math.ceil((w - patchSize) / (patchSize - overlap)) + 1 or 1)):
            y = i * (patchSize - overlap)
            x = j * (patchSize - overlap)
            if i == 0 and j == 0:
                patch = corrOver(util.img_as_float(texture)[:,:,:3], color.rgb2gray(texture), color.rgb2gray(target), patchSize, overlap, output, x, y, alpha)
            else:
                patch = corrOver(util.img_as_float(texture)[:,:,:3], color.rgb2gray(texture), color.rgb2gray(target), patchSize, overlap, output, x, y, alpha)
                patch = cutPatch(patch, overlap, output, x, y)
            output[y:y+patchSize, x:x+patchSize] = patch
    return output

class QuiltTexApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("QuiltTex")
        self.geometry("900x600")

        self.texture = None
        self.target = None

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images with light and dark mode image
        self.logo_image = ctk.CTkImage(Image.open("logo.png"), size=(34, 26))

        # create navigation frame
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(5, weight=1) 

        # Add other elements below the tab view
        self.navigation_frame_label = ctk.CTkLabel(self.navigation_frame, text="  QuiltTex", image=self.logo_image, compound="left", font=ctk.CTkFont(size=20, weight="bold", family="Helvetica"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        # Create tab buttons frame
        self.seg_button = ctk.CTkSegmentedButton(self.navigation_frame)
        self.seg_button.grid(row=1, column=0, padx=20, pady=20, sticky="ew")
        self.seg_button.configure(values=["Synthesis", "Transfer"], command=self.on_segment_change)
        self.seg_button.set("Synthesis")

        # Add method buttons
        for i, (ref, (name, command)) in enumerate({
            "method1_button": ("Random Patch", self.method1_button_event),
            "method2_button": ("Overlap-Constrained Patch", self.method2_button_event),
            "method3_button": ("Minimum Error Boundary Cut", self.method3_button_event)
        }.items(), start=1):
            button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text=name, fg_color="transparent",text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w", command=command)
            button.grid(row=i + 1, column=0, sticky="ew")
            setattr(self, ref, button)

        # Add appearance mode menu
        self.appearance_mode_menu = ctk.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"], command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=20, sticky="s")

        # create frames for each section
        self.method_frame = self.create_method_frame()

        # select default frame
        self.select_frame_by_name("method1")

    def create_method_frame(self):
        scrollable_frame = ctk.CTkScrollableFrame(self, corner_radius=0, fg_color="transparent")
        scrollable_frame.grid(row=0, column=1, sticky="nsew")
        frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="n")

        # Centre the widgets
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

        # Images
        self.image_texture = ctk.CTkLabel(frame, text="Texture Image", width=200, height=100)
        self.image_texture.grid(row=0, column=0, padx=20, pady=10)
        self.load_texture_button = ctk.CTkButton(frame, text="Load Texture", command=self.load_texture)
        self.load_texture_button.grid(row=1, column=0, padx=20, pady=10)

        self.image_target = ctk.CTkLabel(frame, text="Target Image", width=200, height=100)
        self.load_target_button = ctk.CTkButton(frame, text="Load Target", command=self.load_target)

        # Inputs
        self.patch_size_label = ctk.CTkLabel(frame, text="Patch Size")
        self.patch_size_label.grid(row=2, column=0, padx=20, pady=(10, 0))
        self.patch_size_entry = ctk.CTkEntry(frame)
        self.patch_size_entry.grid(row=3, column=0, padx=20, pady=(0, 10))

        self.overlap_size_label = ctk.CTkLabel(frame, text="Overlap Size")
        self.overlap_size_entry = ctk.CTkEntry(frame)

        self.alpha_label = ctk.CTkLabel(frame, text="Alpha")
        self.alpha_entry = ctk.CTkEntry(frame)

        # Button and Load
        self.go_button = ctk.CTkButton(frame)
        self.go_button.grid(row=6, column=1, pady=10)
        self.progressbar = ctk.CTkProgressBar(frame)
        self.progressbar.set(0)
        self.progressbar.grid(row=7, column=1, pady=(10, 10), sticky="ew")

        # Display Image
        ctk.CTkButton(frame, text="Save Image", command=self.save_output_image).grid(row=8, column=1, pady=10)
        self.image_output = ctk.CTkLabel(frame, text="Output Image", width=200, height=100)
        self.image_output.grid(row=9, column=1, padx=20, pady=10)

        for entry_name, entry in [("patch_size_entry", self.patch_size_entry), ("overlap_size_entry", self.overlap_size_entry), ("alpha_entry", self.alpha_entry)]:
            setattr(scrollable_frame, entry_name, entry)

        return scrollable_frame

    def select_frame_by_name(self, name):
        buttons = [self.method1_button, self.method2_button, self.method3_button]
        methods = {"method1": self.run_method1, "method2": self.run_method2, "method3": self.run_method3, "transfer": self.run_transfer}

        # Update button colors
        for i, button in enumerate(buttons, start=1):
            button.configure(fg_color=("gray75", "gray25") if f"method{i}" == name else "transparent")

        for widget in [self.image_target, self.load_target_button, self.overlap_size_label, self.overlap_size_entry, self.alpha_label, self.alpha_entry]:
            widget.grid_forget()

        method2_widgets = [
            (self.overlap_size_label, {"row": 2, "column": 1, "padx": 20, "pady": (10, 0)}),
            (self.overlap_size_entry, {"row": 3, "column": 1, "padx": 20, "pady": (0, 10)})
        ]
        transfer_widgets = [
            (self.image_target, {"row": 0, "column": 1, "padx": 20, "pady": 10}),
            (self.load_target_button, {"row": 1, "column": 1, "padx": 20, "pady": 10}),
            (self.alpha_label, {"row": 2, "column": 2, "padx": 20, "pady": (10, 0)}),
            (self.alpha_entry, {"row": 3, "column": 2, "padx": 20, "pady": (0, 10)})
        ]

        # Show method-specific widgets
        self.go_button.configure(command=methods[name], text="Synthesise")
        if name == "method2" or name == "method3":
            for widget, grid_options in method2_widgets:
                widget.grid(**grid_options)
        elif name == "transfer":
            self.go_button.configure(command=methods[name], text="Transfer")
            for widget, grid_options in (method2_widgets+transfer_widgets):
                widget.grid(**grid_options)
    
    def on_segment_change(self, value):
        if value == "Synthesis":
            self.method1_button.grid()
            self.method2_button.grid()
            self.method3_button.grid()
            self.select_frame_by_name("method1")
        elif value == "Transfer":
            self.method1_button.grid_remove()
            self.method2_button.grid_remove()
            self.method3_button.grid_remove()
            self.select_frame_by_name("transfer")
    
    def method1_button_event(self):
        self.select_frame_by_name("method1")
    def method2_button_event(self):
        self.select_frame_by_name("method2")
    def method3_button_event(self):
        self.select_frame_by_name("method3")

    def change_appearance_mode_event(self, new_appearance_mode):
        ctk.set_appearance_mode(new_appearance_mode)

    def load_texture(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        self.texture = io.imread(file_path)
        self.display_image(self.texture, self.image_texture)

    def load_target(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        self.target = io.imread(file_path)
        self.display_image(self.target, self.image_target)

    def display_image(self, image, label):
        image = Image.fromarray(img_as_ubyte(image))
        aspect_ratio = image.height / image.width
        new_height = int(200 * aspect_ratio)
        resized_image = image.resize((200, new_height), Image.LANCZOS)
        ctk_image = ctk.CTkImage(light_image=resized_image, dark_image=resized_image, size=resized_image.size)
        label.configure(image=ctk_image, text="")
        label.image = ctk_image

    def save_output_image(self):
        if self.output is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                image = Image.fromarray(img_as_ubyte(self.output))
                image.save(file_path)

    def run_general_method(self, patch_size, overlap_size=None, alpha=None, method=None):
        if self.texture is None or self.texture.size == 0:
            self.image_output.configure(text="Please load texture")
            return
        if not patch_size:
            self.image_output.configure(text="Please enter patch size")
            return
        if overlap_size is not None and not overlap_size:
            self.image_output.configure(text="Please enter overlap size")
            return
        if alpha is not None and not alpha:
            self.image_output.configure(text="Please enter alpha")
            return

        def run_transfer():
            if self.target is None or self.target.size == 0:
                self.image_output.configure(text="Please load target")
                return
            self.progressbar.configure(mode="indeterminate")
            self.progressbar.start()
            self.output = transfer(self.texture, self.target, int(patch_size), int(overlap_size), float(alpha))
            self.progressbar.stop()
            self.progressbar.configure(mode="determinate")
            self.progressbar.set(0)
            self.display_image(self.output, self.image_output)

        def run_general():
            self.progressbar.configure(mode="indeterminate")
            self.progressbar.start()
            self.output = general_method(self.texture, int(patch_size), int(overlap_size) if overlap_size else None, method)
            self.progressbar.stop()
            self.progressbar.configure(mode="determinate")
            self.progressbar.set(0)
            self.display_image(self.output, self.image_output)

        if alpha is not None:
            threading.Thread(target=run_transfer).start()
        else:
            threading.Thread(target=run_general).start()

    def run_method1(self):
        self.run_general_method(self.method_frame.patch_size_entry.get(), method=randomPatch)

    def run_method2(self):
        self.run_general_method(self.method_frame.patch_size_entry.get(), self.method_frame.overlap_size_entry.get(), method=bestPatchSelection)

    def run_method3(self):
        self.run_general_method(self.method_frame.patch_size_entry.get(), self.method_frame.overlap_size_entry.get(), method=cutPatchSelection)

    def run_transfer(self):
        self.run_general_method(self.method_frame.patch_size_entry.get(), self.method_frame.overlap_size_entry.get(), self.method_frame.alpha_entry.get())

if __name__ == "__main__":
    app = QuiltTexApp()
    app.mainloop()