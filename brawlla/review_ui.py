"""Revisão manual e calibração visual, sem dependência do motor de análise."""

from copy import deepcopy
import json
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk
from .dataset import CATEGORIES, load_entries, save_annotation

STATUS_LABELS = {
    "unreviewed": "Pendente",
    "accepted": "Aprovada",
    "corrected": "Corrigida",
    "rejected": "Descartada",
    "uncertain": "Incerta",
}


class ImageCanvas(tk.Canvas):
    def __init__(self, master, on_rectangle):
        super().__init__(
            master, bg="#101722", highlightthickness=0, width=900, height=500
        )
        self.frame = None
        self.boxes = []
        self.scale = 1
        self.offset = (0, 0)
        self.on_rectangle = on_rectangle
        self.anchor = None
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<ButtonPress-1>", self.begin)
        self.bind("<B1-Motion>", self.drag)
        self.bind("<ButtonRelease-1>", self.end)

    def show(self, frame, boxes):
        self.frame, self.boxes = frame, boxes
        self.redraw()

    def point(self, event):
        if self.frame is None:
            return 0, 0
        h, w = self.frame.shape[:2]
        return (
            max(0, min(w, (event.x - self.offset[0]) / self.scale)),
            max(0, min(h, (event.y - self.offset[1]) / self.scale)),
        )

    def begin(self, event):
        self.anchor = self.point(event)

    def drag(self, event):
        if self.anchor is None:
            return
        x, y = self.anchor
        xx, yy = self.point(event)
        self.delete("drag")
        self.create_rectangle(
            x * self.scale + self.offset[0],
            y * self.scale + self.offset[1],
            xx * self.scale + self.offset[0],
            yy * self.scale + self.offset[1],
            outline="#f7da55",
            width=2,
            tags="drag",
        )

    def end(self, event):
        if self.anchor is not None:
            x, y = self.anchor
            xx, yy = self.point(event)
            if abs(x - xx) >= 2 and abs(y - yy) >= 2:
                self.on_rectangle([min(x, xx), min(y, yy), abs(xx - x), abs(yy - y)])
        self.anchor = None
        self.delete("drag")

    def redraw(self):
        if self.frame is None:
            return
        h, w = self.frame.shape[:2]
        self.scale = min(
            max(1, self.winfo_width()) / w, max(1, self.winfo_height()) / h
        )
        dw, dh = max(1, round(w * self.scale)), max(1, round(h * self.scale))
        self.offset = ((self.winfo_width() - dw) / 2, (self.winfo_height() - dh) / 2)
        image = Image.fromarray(
            cv2.cvtColor(cv2.resize(self.frame, (dw, dh)), cv2.COLOR_BGR2RGB)
        )
        self.photo = ImageTk.PhotoImage(image)
        self.delete("all")
        ox, oy = self.offset
        self.create_image(ox, oy, image=self.photo, anchor="nw")
        for box in self.boxes:
            x, y, bw, bh = box["bbox"]
            self.create_rectangle(
                ox + x * self.scale,
                oy + y * self.scale,
                ox + (x + bw) * self.scale,
                oy + (y + bh) * self.scale,
                outline="#45ead4",
                width=2,
            )
            self.create_text(
                ox + x * self.scale + 2,
                oy + y * self.scale,
                text=box["category"],
                fill="#ffe975",
                anchor="sw",
            )


class CalibrationWindow(tk.Toplevel):
    def __init__(self, master, frame, callback):
        super().__init__(master)
        self.title("Calibrar área do jogo")
        self.transient(master)
        self.grab_set()
        self.geometry("1000x680")
        self.frame, self.callback, self.roi = frame, callback, None
        tk.Label(
            self,
            text="Arraste para selecionar a área inteira do jogo, incluindo o HUD. Mantenha o BlueStacks descoberto.",
        ).pack(pady=8)
        self.canvas = ImageCanvas(self, self.select)
        self.canvas.pack(fill="both", expand=True)
        self.message = tk.StringVar(value="Sem recorte selecionado")
        tk.Label(self, textvariable=self.message).pack()
        ttk.Button(self, text="Aplicar recorte", command=self.apply).pack(pady=10)
        self.after(30, lambda: self.canvas.show(frame, []))

    def select(self, rect):
        self.roi = rect
        self.canvas.boxes = [{"bbox": rect, "category": "área do jogo"}]
        self.canvas.redraw()
        self.message.set("Recorte: " + " × ".join(str(round(n)) for n in rect))

    def apply(self):
        if not self.roi or self.roi[2] < 320 or self.roi[3] < 200:
            messagebox.showwarning(
                "Recorte pequeno", "Selecione pelo menos 320 × 200 pixels.", parent=self
            )
            return
        h, w = self.frame.shape[:2]
        x, y, bw, bh = self.roi
        self.callback((x / w, y / h, bw / w, bh / h), w / h)
        self.destroy()


class ReviewWindow(tk.Toplevel):
    def __init__(self, master, folder):
        super().__init__(master)
        self.folder = Path(folder)
        self.entries = load_entries(folder)
        if not self.entries:
            self.destroy()
            raise ValueError("Esta sessão não contém amostras PNG.")
        self.index = 0
        self.dirty = False
        self.title("Revisar dados — caixas em pixels da captura original")
        self.geometry("1100x790")
        self.info = tk.StringVar()
        tk.Label(self, textvariable=self.info, wraplength=1050).pack(pady=5)
        self.canvas = ImageCanvas(self, self.add_box)
        self.canvas.pack(fill="both", expand=True)
        row = ttk.Frame(self)
        row.pack(fill="x", pady=8)
        self.category = tk.StringVar(value=CATEGORIES[0])
        ttk.Label(row, text="Categoria ao arrastar:").pack(side="left", padx=8)
        ttk.Combobox(
            row,
            textvariable=self.category,
            values=CATEGORIES,
            state="readonly",
            width=20,
        ).pack(side="left")
        ttk.Button(row, text="Remover última", command=self.remove_last).pack(
            side="left", padx=6
        )
        ttk.Button(row, text="Limpar caixas", command=self.clear).pack(side="left")
        self.overlays = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row, text="Mostrar caixas", variable=self.overlays, command=self.refresh
        ).pack(side="left", padx=8)
        owner_row = ttk.Frame(self)
        owner_row.pack(fill="x", pady=4)
        ttk.Label(owner_row, text="Dono da próxima barra:").pack(side="left", padx=8)
        self.owner_choice = tk.StringVar(value="Sem vínculo")
        self.owner_combo = ttk.Combobox(
            owner_row,
            textvariable=self.owner_choice,
            values=["Sem vínculo"],
            state="readonly",
            width=38,
        )
        self.owner_combo.pack(side="left")
        ttk.Label(
            owner_row,
            text="Desenhe o corpo antes; depois selecione-o e desenhe sua barra.",
        ).pack(side="left", padx=8)
        row = ttk.Frame(self)
        row.pack(fill="x", pady=8)
        self.status = tk.StringVar(value=STATUS_LABELS["unreviewed"])
        self.negative = tk.BooleanVar(value=False)
        ttk.Label(row, text="Revisão:").pack(side="left", padx=8)
        ttk.Combobox(
            row,
            textvariable=self.status,
            values=list(STATUS_LABELS.values()),
            state="readonly",
            width=14,
        ).pack(side="left")
        ttk.Checkbutton(
            row, text="Negativo: nenhum objeto das 4 categorias", variable=self.negative
        ).pack(side="left", padx=6)
        ttk.Button(row, text="Salvar revisão", command=self.save).pack(
            side="right", padx=8
        )
        ttk.Button(row, text="Próxima", command=lambda: self.navigate(1)).pack(
            side="right"
        )
        ttk.Button(row, text="Anterior", command=lambda: self.navigate(-1)).pack(
            side="right", padx=4
        )
        tk.Label(
            self,
            text="Arraste para desenhar. Pendente, Descartada e Incerta ficam fora da exportação.\nAprovar só barras não rotula os corpos: complete todas as categorias presentes. Salve antes de fechar.",
            wraplength=1040,
        ).pack(pady=6)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(30, self.load)

    def load(self):
        entry = self.entries[self.index]
        image = cv2.imread(str(self.folder / entry["image"]))
        if image is None:
            messagebox.showerror("Imagem inválida", entry["image"], parent=self)
            return
        path = self.folder / "labels" / (Path(entry["image"]).stem + ".json")
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            prediction = json.loads(
                (self.folder / entry["prediction"]).read_text(encoding="utf-8")
            )
            value = {
                "boxes": prediction["proposed_boxes"],
                "status": "unreviewed",
                "negative": False,
            }
        self.boxes = deepcopy(value["boxes"])
        self.status.set(STATUS_LABELS[value["status"]])
        self.negative.set(value.get("negative", False))
        self.frame = image
        self.info.set(
            f"{self.index+1}/{len(self.entries)} | Sessão {entry['session_id']} | Quadro {entry['frame_id']} | {entry['width']} × {entry['height']}\nQualidade: {entry.get('quality',{}).get('flags',[])} | Seleção: {entry.get('selection_reasons',[])}"
        )
        self.refresh()

    def refresh(self):
        if hasattr(self, "frame"):
            owners = {}
            for box in self.boxes:
                if box["category"] != "barra_vida":
                    box.setdefault("entity_id", "body_" + uuid.uuid4().hex[:8])
                    owners[box["category"] + " / " + box["entity_id"]] = box[
                        "entity_id"
                    ]
            self.owner_map = owners
            self.owner_combo.configure(values=["Sem vínculo", *owners])
            if self.owner_choice.get() not in owners:
                self.owner_choice.set("Sem vínculo")
            self.canvas.show(self.frame, self.boxes if self.overlays.get() else [])

    def add_box(self, rect):
        box = {"category": self.category.get(), "bbox": rect, "origin": "human"}
        if box["category"] == "barra_vida":
            owner = self.owner_map.get(self.owner_choice.get())
            if owner:
                box["owner_id"] = owner
        else:
            box["entity_id"] = "body_" + uuid.uuid4().hex[:8]
        self.boxes.append(box)
        self.negative.set(False)
        self.refresh()

    def remove_last(self):
        if self.boxes:
            removed = self.boxes.pop()
            for box in self.boxes:
                if box.get("owner_id") == removed.get("entity_id"):
                    box.pop("owner_id", None)
        self.refresh()

    def clear(self):
        self.boxes.clear()
        self.refresh()

    def save(self):
        try:
            save_annotation(
                self.folder,
                self.entries[self.index],
                {
                    "status": next(
                        key
                        for key, label in STATUS_LABELS.items()
                        if label == self.status.get()
                    ),
                    "negative": self.negative.get(),
                    "boxes": self.boxes,
                },
            )
            return True
        except Exception as exc:
            messagebox.showerror("Revisão não salva", str(exc), parent=self)
            return False

    def navigate(self, delta):
        if self.save():
            self.index = max(0, min(len(self.entries) - 1, self.index + delta))
            self.load()

    def close(self):
        if self.save():
            self.destroy()
