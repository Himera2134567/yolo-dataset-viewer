import json
import os
import glob
import math
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw


class YoloSegViewerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO Viewer (bbox + segmentation)")

        # --- состояние ---
        self.dataset_dir = None
        self.images_dir = None
        self.labels_dir = None
        self.image_paths = []
        self.current_index = 0
        self._resize_job = None
        self.current_photo = None  # чтобы не собирался GC

        # --- UI ---
        self._build_ui()

    def _build_ui(self):
        # Верхняя панель с кнопками
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        btn_select = tk.Button(top_frame, text="Выбрать датасет", command=self.select_dataset)
        btn_select.pack(side=tk.LEFT, padx=5)

        btn_prev = tk.Button(top_frame, text="<< Предыдущая", command=self.show_prev)
        btn_prev.pack(side=tk.LEFT, padx=5)

        btn_next = tk.Button(top_frame, text="Следующая >>", command=self.show_next)
        btn_next.pack(side=tk.LEFT, padx=5)

        self.info_label = tk.Label(top_frame, text="Датасет не загружен")
        self.info_label.pack(side=tk.LEFT, padx=10)

        # Область для изображения
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # При изменении размера окна – перерисовываем текущую картинку
        self.canvas.bind("<Configure>", self._on_resize)

    # ---------- Логика работы ----------

    def select_dataset(self):
        path = filedialog.askdirectory(title="Выберите папку с датасетом (images + labels)")
        if path:
            try:
                self.load_dataset(path)
            except (OSError, ValueError) as exc:
                messagebox.showerror("Ошибка", str(exc))

    def load_dataset(self, path):
        images_dir = os.path.join(path, "images")
        labels_dir = os.path.join(path, "labels")
        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            raise ValueError("Нужны подпапки images и labels в выбранном каталоге.")
        image_paths = sorted(item.path for item in os.scandir(images_dir) if item.is_file() and Path(item.name).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if not image_paths:
            raise ValueError("В папке images нет поддерживаемых изображений.")
        self.dataset_dir, self.images_dir, self.labels_dir = path, images_dir, labels_dir
        self.image_paths = image_paths
        self.current_index = 0
        self.show_current()

    def show_prev(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.show_current()

    def show_next(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.show_current()

    def _on_resize(self, event):
        # Перерисовать текущее изображение при изменении размеров окна
        if self.image_paths:
            if self._resize_job is not None:
                self.root.after_cancel(self._resize_job)
            self._resize_job = self.root.after(80, self._redraw)

    def _redraw(self):
        self._resize_job = None
        self.show_current(redraw_only=True)

    def show_current(self, redraw_only=False):
        if not self.image_paths:
            return

        img_path = self.image_paths[self.current_index]
        filename = os.path.basename(img_path)
        stem, _ = os.path.splitext(filename)
        label_path = os.path.join(self.labels_dir, stem + ".txt")

        try:
            with Image.open(img_path) as original:
                image = original.convert("RGB")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть изображение:\n{img_path}\n{e}")
            return

        width, height = image.size

        # Парсим labels (bbox + polygons)
        shapes = []
        has_labels = os.path.isfile(label_path)
        if has_labels:
            shapes = self._parse_yolo_labels(label_path, width, height)
            image = self._draw_shapes(image, shapes)

        # Масштабируем под размер canvas
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        image = self._scale_to_fit(image, canvas_width, canvas_height)

        self.current_photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.current_photo)

        # Обновляем инфо
        lbl_info = f"{self.current_index + 1}/{len(self.image_paths)}: {filename}"
        if has_labels:
            lbl_info += f" | labels: {len(shapes)} корректных фигур"
        else:
            lbl_info += " | labels: НЕТ"
        self.info_label.config(text=lbl_info)

    # ---------- Вспомогательные методы ----------

    @staticmethod
    def _scale_to_fit(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
        w, h = image.size
        if w <= max_w and h <= max_h:
            return image

        scale = min(max_w / w, max_h / h)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size, Image.LANCZOS)

    @staticmethod
    def _parse_yolo_labels(label_path: str, img_w: int, img_h: int):
        """Read boxes/polygons; skip malformed, non-finite or out-of-range rows."""
        if img_w <= 0 or img_h <= 0:
            return []
        try:
            lines = Path(label_path).read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError):
            return []
        shapes = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_value = float(parts[0])
                values = list(map(float, parts[1:]))
            except ValueError:
                continue
            if not math.isfinite(class_value) or class_value < 0 or not class_value.is_integer():
                continue
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                continue
            class_id = int(class_value)
            if len(values) == 4:
                x, y, w, h = values
                if w <= 0 or h <= 0:
                    continue
                points = [(max(0, (x-w/2)*img_w), max(0, (y-h/2)*img_h)), (min(img_w, (x+w/2)*img_w), min(img_h, (y+h/2)*img_h))]
                shapes.append({"class_id":class_id, "type":"rect", "points":points})
            elif len(values) >= 6 and len(values) % 2 == 0:
                points = [(values[i]*img_w, values[i+1]*img_h) for i in range(0, len(values), 2)]
                shapes.append({"class_id":class_id, "type":"polygon", "points":points})
        return shapes

    @staticmethod
    def _draw_shapes(image: Image.Image, shapes):
        """
        Рисует прямоугольники и полигоны на изображении.
        shapes: список словарей {"class_id", "type", "points"}
        """
        if not shapes:
            return image

        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Несколько цветов, по индексам будем крутить
        base_colors = [
            (255, 0, 0),      # красный
            (0, 255, 0),      # зелёный
            (0, 0, 255),      # синий
            (255, 255, 0),    # жёлтый
            (255, 0, 255),    # пурпурный
            (0, 255, 255),    # бирюзовый
            (255, 128, 0),    # оранжевый
            (128, 0, 255),    # фиолетовый
        ]

        for idx, shape in enumerate(shapes):
            shape_type = shape.get("type")
            points = shape.get("points", [])
            if not points:
                continue

            base_color = base_colors[idx % len(base_colors)]
            outline_color = (base_color[0], base_color[1], base_color[2], 255)
            fill_color = (base_color[0], base_color[1], base_color[2], 80)  # прозрачная заливка

            if shape_type == "rect" and len(points) == 2:
                (x1, y1), (x2, y2) = points
                draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color)
            elif shape_type == "polygon" and len(points) >= 3:
                draw.polygon(points, outline=outline_color, fill=fill_color)

        result = Image.alpha_composite(base, overlay).convert("RGB")
        return result


def main():
    root = tk.Tk()
    app = YoloSegViewerApp(root)
    root.geometry("1000x700")
    root.mainloop()


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        from smoke_checks import run_checks
        import faulthandler
        trace = None
        if "--self-test-report" in sys.argv:
            trace = Path(sys.argv[sys.argv.index("--self-test-report")+1]).with_suffix(".trace.txt").open("w", encoding="utf-8")
            faulthandler.dump_traceback_later(20, file=trace)
        try:
            result = run_checks(sys.modules[__name__])
        except Exception:
            import traceback
            result = {"ok":False,"error":traceback.format_exc()}
        faulthandler.cancel_dump_traceback_later()
        if trace is not None:
            trace.close()
        if "--self-test-report" in sys.argv:
            Path(sys.argv[sys.argv.index("--self-test-report")+1]).write_text(json.dumps(result), encoding="utf-8")
        if sys.stdout:
            print(result)
        raise SystemExit(0 if result["ok"] else 1)
    else:
        main()
