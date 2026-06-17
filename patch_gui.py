import sys
import os
import json

with open("gui_app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Imports
content = content.replace("import threading", "import json\nimport threading")
content = content.replace("from pathlib import Path", "from pathlib import Path\nimport cv2")

# 2. GoalSelectorDialog class
goal_selector_code = """
class GoalSelectorDialog(tk.Toplevel):
    def __init__(self, parent, video_path: str):
        super().__init__(parent)
        self.title("Définir les zones de but")
        self.geometry("800x600")
        self.result = []
        self._zones = []
        self._current_zone = None
        self._start_pos = None

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Erreur", "Impossible de lire la vidéo.")
            self.destroy()
            return
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        self.scale = min(800 / w, 500 / h)
        new_w, new_h = int(w * self.scale), int(h * self.scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))
        
        self.img = ImageTk.PhotoImage(Image.fromarray(frame_resized))
        
        ttk.Label(self, text="Dessinez DEUX rectangles pour définir les zones de but. Puis fermez la fenêtre.").pack(pady=5)
        
        self.canvas = tk.Canvas(self, width=new_w, height=new_h, cursor="cross")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        ttk.Button(self, text="Confirmer", command=self.confirm).pack(pady=10)
        
    def on_press(self, event):
        if len(self._zones) >= 2: return
        self._start_pos = (event.x, event.y)
        self._current_zone = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=2)
        
    def on_drag(self, event):
        if self._current_zone:
            self.canvas.coords(self._current_zone, self._start_pos[0], self._start_pos[1], event.x, event.y)
            
    def on_release(self, event):
        if self._current_zone:
            x1, y1 = self._start_pos
            x2, y2 = event.x, event.y
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            orig_rect = (int(x1 / self.scale), int(y1 / self.scale), int(x2 / self.scale), int(y2 / self.scale))
            self.result.append(orig_rect)
            self._zones.append(self._current_zone)
            self._current_zone = None
            
    def confirm(self):
        self.destroy()

class ScrollableFrame"""
content = content.replace("class ScrollableFrame", goal_selector_code)

# 3. Add self.goal_zones = [] in __init__
init_code = """        self.video_path: str = ""
        self.output_dir: str | None = None
        self.result: AnalysisResult | None = None
        self.goal_zones: list = []"""
content = content.replace("        self.video_path: str = \"\"\n        self.output_dir: str | None = None\n        self.result: AnalysisResult | None = None", init_code)

# 4. Buttons in UI
btn_def_buts = """
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(btn_frame, text="📁 Choisir...", command=self._pick_video).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="🥅 Définir Zones de But", command=self._define_goal_zones).pack(side=tk.LEFT, padx=(10, 0))
"""
content = content.replace("ttk.Button(file_frame, text=\"📁 Choisir...\", command=self._pick_video).pack(anchor=tk.W, pady=(0, 10))", btn_def_buts)

btn_report = """        self.open_folder_btn.pack(side=tk.LEFT, padx=5)

        self.report_btn = ttk.Button(actions, text="⚠️ Signaler une erreur (IA Apprentissage)", command=self._report_error)
        self.report_btn.pack(side=tk.LEFT, padx=5)"""
content = content.replace("self.open_folder_btn.pack(side=tk.LEFT, padx=5)", btn_report)

# 5. Add goal_zones to AnalysisConfig
run_analysis_old = """                owner_confirm_frames=int(self.owner_confirm_var.get()),
                min_possession_pass=int(self.min_poss_var.get()),
            )"""
run_analysis_new = """                owner_confirm_frames=int(self.owner_confirm_var.get()),
                min_possession_pass=int(self.min_poss_var.get()),
                goal_zones=self.goal_zones,
            )"""
content = content.replace(run_analysis_old, run_analysis_new)

# 6. Add the new methods
new_methods = """
    def _define_goal_zones(self):
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showwarning("Vidéo manquante", "Veuillez d'abord choisir une vidéo.")
            return
        dlg = GoalSelectorDialog(self.root, self.video_path)
        self.root.wait_window(dlg)
        if len(dlg.result) > 0:
            self.goal_zones = dlg.result
            self.status_var.set(f"{len(self.goal_zones)} zones de but définies manuellement.")

    def _report_error(self):
        if not self.result:
            return
            
        def on_submit():
            error_type = combo.get()
            top.destroy()
            self._learn_and_retry(error_type)
            
        top = tk.Toplevel(self.root)
        top.title("Signaler une erreur")
        ttk.Label(top, text="Quel problème avez-vous constaté ?").pack(pady=10, padx=10)
        
        options = [
            "Passe confondue avec un tir (Vitesse tir trop basse)",
            "Tir non détecté (Vitesse tir trop haute)",
            "Passes non détectées (Distance minimale trop grande)",
            "Mauvaise possession (Rayon possession trop petit)",
            "Mauvaises zones de but"
        ]
        combo = ttk.Combobox(top, values=options, width=50, state="readonly")
        combo.current(0)
        combo.pack(pady=5, padx=10)
        
        ttk.Button(top, text="Corriger automatiquement & Relancer", command=on_submit).pack(pady=10)

    def _learn_and_retry(self, error_type: str):
        if "zones de but" in error_type.lower():
            self._define_goal_zones()
            if not self.goal_zones:
                return
        
        history_dir = Path("learning_history")
        history_dir.mkdir(exist_ok=True)
        history_file = history_dir / "learning.json"
        
        history = []
        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                try:
                    history = json.load(f)
                except:
                    pass
                
        current_config = {
            "shot_speed": self.shot_speed_var.get(),
            "pass_dist": self.pass_dist_var.get(),
            "possession_radius": self.possession_var.get()
        }
        
        history.append({
            "config": current_config,
            "error_reported": error_type,
            "success": False
        })
        
        if "Vitesse tir trop basse" in error_type:
            self.shot_speed_var.set(self.shot_speed_var.get() + 3.0)
        elif "Vitesse tir trop haute" in error_type:
            self.shot_speed_var.set(max(5.0, self.shot_speed_var.get() - 3.0))
        elif "Distance minimale" in error_type:
            self.pass_dist_var.set(max(10.0, self.pass_dist_var.get() - 5.0))
        elif "Mauvaise possession" in error_type:
            self.possession_var.set(self.possession_var.get() + 10.0)
            
        new_config = {
            "shot_speed": self.shot_speed_var.get(),
            "pass_dist": self.pass_dist_var.get(),
            "possession_radius": self.possession_var.get()
        }
        history.append({
            "config": new_config,
            "error_reported": "auto_adjusted",
            "success": "pending"
        })
        
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
            
        messagebox.showinfo("Apprentissage IA", "Les paramètres ont été ajustés en fonction de votre signalement. L'analyse va redémarrer.")
        self._start_analysis()

    def _open_annotated_video(self) -> None:"""
content = content.replace("    def _open_annotated_video(self) -> None:", new_methods)

with open("gui_app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Patching successful.")
