import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from skimage import io, util, color, img_as_ubyte
import math
import heapq

def randomPatch(texture, patchSize):
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
            patch = patchSelectionFunc(texture, patchSize)
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

class TextureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Texture Synthesis and Transfer")

        self.texture = None
        self.target = None

        # Configure styles
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 20), padding=10)
        style.configure('TLabel', font=('Helvetica', 20))
        style.configure('TEntry', font=('Helvetica', 20))
        style.theme_use('clam')

        # Create Notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=1, fill='both')

        # Create frames for each tab
        self.synthesis_frame = ttk.Frame(self.notebook)
        self.transfer_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.synthesis_frame, text='Synthesis')
        self.notebook.add(self.transfer_frame, text='Transfer')

        # Synthesis tab with sub-tabs for methods
        self.synthesis_notebook = ttk.Notebook(self.synthesis_frame)
        self.synthesis_notebook.pack(expand=1, fill='both')

        self.method1_frame = ttk.Frame(self.synthesis_notebook)
        self.method2_frame = ttk.Frame(self.synthesis_notebook)
        self.method3_frame = ttk.Frame(self.synthesis_notebook)

        self.synthesis_notebook.add(self.method1_frame, text='Method 1')
        self.synthesis_notebook.add(self.method2_frame, text='Method 2')
        self.synthesis_notebook.add(self.method3_frame, text='Method 3')

        # Method 1 tab
        self.load_texture_btn = ttk.Button(self.method1_frame, text="Load Texture", command=self.load_texture)
        self.load_texture_btn.pack()

        self.patch_size_label = ttk.Label(self.method1_frame, text="Patch Size:")
        self.patch_size_label.pack()
        self.patch_size_entry = ttk.Entry(self.method1_frame)
        self.patch_size_entry.pack()

        self.method1_btn = ttk.Button(self.method1_frame, text="Run Method 1", command=self.run_method1)
        self.method1_btn.pack()

        # Method 2 tab
        self.load_texture_btn2 = ttk.Button(self.method2_frame, text="Load Texture", command=self.load_texture)
        self.load_texture_btn2.pack()

        self.patch_size_label2 = ttk.Label(self.method2_frame, text="Patch Size:")
        self.patch_size_label2.pack()
        self.patch_size_entry2 = ttk.Entry(self.method2_frame)
        self.patch_size_entry2.pack()

        self.overlap_size_label2 = ttk.Label(self.method2_frame, text="Overlap Size:")
        self.overlap_size_label2.pack()
        self.overlap_size_entry2 = ttk.Entry(self.method2_frame)
        self.overlap_size_entry2.pack()

        self.method2_btn = ttk.Button(self.method2_frame, text="Run Method 2", command=self.run_method2)
        self.method2_btn.pack()

        # Method 3 tab
        self.load_texture_btn3 = ttk.Button(self.method3_frame, text="Load Texture", command=self.load_texture)
        self.load_texture_btn3.pack()

        self.patch_size_label3 = ttk.Label(self.method3_frame, text="Patch Size:")
        self.patch_size_label3.pack()
        self.patch_size_entry3 = ttk.Entry(self.method3_frame)
        self.patch_size_entry3.pack()

        self.overlap_size_label3 = ttk.Label(self.method3_frame, text="Overlap Size:")
        self.overlap_size_label3.pack()
        self.overlap_size_entry3 = ttk.Entry(self.method3_frame)
        self.overlap_size_entry3.pack()

        self.method3_btn = ttk.Button(self.method3_frame, text="Run Method 3", command=self.run_method3)
        self.method3_btn.pack()

        # Transfer tab
        self.load_texture_btn_transfer = ttk.Button(self.transfer_frame, text="Load Texture", command=self.load_texture)
        self.load_texture_btn_transfer.pack()

        self.load_target_btn = ttk.Button(self.transfer_frame, text="Load Target", command=self.load_target)
        self.load_target_btn.pack()

        self.patch_size_label_transfer = ttk.Label(self.transfer_frame, text="Patch Size:")
        self.patch_size_label_transfer.pack()
        self.patch_size_entry_transfer = ttk.Entry(self.transfer_frame)
        self.patch_size_entry_transfer.pack()

        self.overlap_size_label_transfer = ttk.Label(self.transfer_frame, text="Overlap Size:")
        self.overlap_size_label_transfer.pack()
        self.overlap_size_entry_transfer = ttk.Entry(self.transfer_frame)
        self.overlap_size_entry_transfer.pack()

        self.alpha_label = ttk.Label(self.transfer_frame, text="Alpha:")
        self.alpha_label.pack()
        self.alpha_entry = ttk.Entry(self.transfer_frame)
        self.alpha_entry.pack()

        self.transfer_btn = ttk.Button(self.transfer_frame, text="Transfer", command=self.run_transfer)
        self.transfer_btn.pack()

        # Canvas to display output
        self.canvas = tk.Canvas(root, width=500, height=500)
        self.canvas.pack()

    def load_texture(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        self.texture = io.imread(file_path)
        self.display_image(self.texture)

    def load_target(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        self.target = io.imread(file_path)
        self.display_image(self.target)

    def display_image(self, image):
        image = ImageTk.PhotoImage(Image.fromarray(img_as_ubyte(image)))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.canvas.image = image

    def run_method1(self):
        output = general_method(self.texture, int(self.patch_size_entry.get()), None, randomPatch)
        self.display_image(output)

    def run_method2(self):
        output = general_method(self.texture, int(self.patch_size_entry2.get()), int(self.overlap_size_entry2.get()), bestPatchSelection)
        self.display_image(output)

    def run_method3(self):
        output = general_method(self.texture, int(self.patch_size_entry3.get()), int(self.overlap_size_entry3.get()), cutPatchSelection)
        self.display_image(output)

    def run_transfer(self):
        output = transfer(self.texture, self.target, int(self.patch_size_entry_transfer.get()), int(self.overlap_size_entry_transfer.get()), float(self.alpha_entry.get()))
        self.display_image(output)

if __name__ == "__main__":
    root = tk.Tk()
    app = TextureApp(root)
    root.mainloop()