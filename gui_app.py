"""
Analyseur de Match Football — Application bureau (sans serveur web).
Lancez avec : python gui_app.py
"""

from __future__ import annotations

import os
import shutil
import tempfile
import json
import threading
import tkinter as tk
from pathlib import Path
import cv2
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from analyzer import VideoAnalyzer
from analyzer.video_analyzer import AnalysisConfig, AnalysisResult
from analyzer.events import EventType



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

class ScrollableFrame(ttk.Frame):
    def __init__(self, container: tk.Widget, *args: object, **kwargs: object) -> None:
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg="#121214")
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Bind mousewheel events to canvas
        self.canvas.bind('<Enter>', self._bind_mousewheel)
        self.canvas.bind('<Leave>', self._unbind_mousewheel)

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _bind_mousewheel(self, event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class MatchAnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Analyseur de Match Football")
        self.geometry("1100x750")
        self.minsize(950, 650)
        self.configure(bg="#121214")

        self.video_path: str | None = None
        self.result: AnalysisResult | None = None
        self.output_dir: str | None = None
        self._preview_refs: list[ImageTk.PhotoImage] = []
        self._analyzing = False
        self.goal_zones: list = []
        self.ball_conf = 0.08
        self.person_conf = 0.32

        self._build_ui()
        self._center_window()

    def _center_window(self) -> None:
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f"+{x}+{y}")

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        # Palette de couleurs Premium Dark (Slate/Indigo)
        bg_dark = "#121214"
        bg_panel = "#1a1a1e"
        bg_card = "#22222a"
        text_light = "#f3f4f6"
        text_muted = "#a1a1aa"
        accent_color = "#4f46e5"     # Indigo
        accent_hover = "#4338ca"
        border_color = "#2a2a35"

        # Configuration globale des styles
        style.configure(".", background=bg_dark, foreground=text_light, font=("Segoe UI", 10))

        # Frames
        style.configure("TFrame", background=bg_dark)
        style.configure("Panel.TFrame", background=bg_panel)
        style.configure("Card.TFrame", background=bg_card)

        # Labels
        style.configure("TLabel", background=bg_dark, foreground=text_light)
        style.configure("Panel.TLabel", background=bg_panel, foreground=text_light)
        style.configure("Card.TLabel", background=bg_card, foreground=text_light)
        style.configure("Muted.TLabel", background=bg_dark, foreground=text_muted)
        style.configure("MutedCard.TLabel", background=bg_card, foreground=text_muted)
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=text_light)

        # LabelFrame
        style.configure("TLabelframe", background=bg_dark, bordercolor=border_color, borderwidth=1)
        style.configure("TLabelframe.Label", background=bg_dark, foreground=accent_color, font=("Segoe UI", 10, "bold"))

        # Boutons Principaux
        style.configure(
            "TButton",
            background=accent_color,
            foreground="#ffffff",
            borderwidth=1,
            bordercolor=accent_color,
            lightcolor=accent_color,
            darkcolor=accent_color,
            focusthickness=0,
            padding=[12, 6],
            font=("Segoe UI", 10, "bold")
        )
        style.map(
            "TButton",
            background=[("active", accent_hover), ("disabled", "#27272a")],
            bordercolor=[("active", accent_hover), ("disabled", "#27272a")],
            lightcolor=[("active", accent_hover), ("disabled", "#27272a")],
            darkcolor=[("active", accent_hover), ("disabled", "#27272a")],
            foreground=[("disabled", "#71717a")]
        )

        # Boutons Secondaires
        style.configure(
            "Secondary.TButton",
            background="#27272a",
            foreground=text_light,
            borderwidth=1,
            bordercolor=border_color,
            lightcolor="#27272a",
            darkcolor="#27272a",
            focusthickness=0,
            padding=[12, 6],
            font=("Segoe UI", 10)
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#3f3f46"), ("disabled", "#1c1c1e")],
            bordercolor=[("active", "#3f3f46"), ("disabled", "#1c1c1e")],
            lightcolor=[("active", "#3f3f46"), ("disabled", "#1c1c1e")],
            darkcolor=[("active", "#3f3f46"), ("disabled", "#1c1c1e")],
            foreground=[("disabled", "#71717a")]
        )

        # Combobox
        style.configure(
            "TCombobox",
            background=bg_card,
            fieldbackground=bg_card,
            foreground=text_light,
            bordercolor=border_color,
            arrowcolor=text_light
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", bg_card)],
            foreground=[("readonly", text_light)]
        )

        # Spinbox
        style.configure(
            "TSpinbox",
            background=bg_card,
            fieldbackground=bg_card,
            foreground=text_light,
            bordercolor=border_color,
            arrowcolor=text_light
        )

        # Scale
        style.configure(
            "Horizontal.TScale",
            background=bg_dark,
            troughcolor="#27272a",
            slidercolor=accent_color,
            sliderheight=14,
            sliderlength=14,
            borderwidth=0
        )
        style.map(
            "Horizontal.TScale",
            slidercolor=[("active", accent_hover)]
        )

        # Progressbar
        style.configure(
            "TProgressbar",
            background=accent_color,
            troughcolor="#27272a",
            bordercolor=border_color,
            borderwidth=1
        )

        # Notebook
        style.configure("TNotebook", background=bg_dark, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#1f1f23",
            foreground=text_muted,
            bordercolor=border_color,
            padding=[16, 8],
            font=("Segoe UI", 10)
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", bg_dark)],
            foreground=[("selected", text_light)],
            bordercolor=[("selected", border_color)]
        )

        # Treeview
        style.configure(
            "Treeview",
            background="#18181b",
            fieldbackground="#18181b",
            foreground=text_light,
            bordercolor=border_color,
            borderwidth=0,
            rowheight=26,
            font=("Segoe UI", 9)
        )
        style.configure(
            "Treeview.Heading",
            background="#27272a",
            foreground=text_light,
            bordercolor=border_color,
            borderwidth=1,
            font=("Segoe UI", 9, "bold")
        )
        style.map(
            "Treeview.Heading",
            background=[("active", "#3f3f46")]
        )
        style.map(
            "Treeview",
            background=[("selected", accent_color)],
            foreground=[("selected", "#ffffff")]
        )

        # Main Layout
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left_container = ScrollableFrame(main)
        left = left_container.scrollable_frame
        
        right = ttk.Frame(main, padding=10)
        
        main.add(left_container, weight=1)
        main.add(right, weight=3)

        # --- Panneau gauche ---
        ttk.Label(left, text="⚽ Analyseur de Match", style="Title.TLabel").pack(anchor=tk.W, pady=(4, 6))

        ttk.Label(
            left,
            text="Choisissez une courte vidéo de match.\nL'analyse se fait en local, sans internet.",
            style="Muted.TLabel",
            wraplength=260,
        ).pack(anchor=tk.W, pady=(0, 12))

        self.path_var = tk.StringVar(value="Aucune vidéo sélectionnée")
        ttk.Label(left, textvariable=self.path_var, wraplength=260, style="Muted.TLabel").pack(anchor=tk.W, pady=(0, 8))

        ttk.Button(left, text="Choisir une vidéo...", command=self._pick_video, style="Secondary.TButton").pack(fill=tk.X, pady=4)

        params = ttk.LabelFrame(left, text="Détection", padding=10)
        params.pack(fill=tk.X, pady=(10, 8))

        self.conf_var = tk.DoubleVar(value=0.32)
        self._add_scale(params, "Confiance joueurs", self.conf_var, 0.2, 0.6)

        self.ball_conf_var = tk.DoubleVar(value=0.08)
        self._add_scale(params, "Confiance ballon (YOLO)", self.ball_conf_var, 0.04, 0.25)

        self.skip_var = tk.IntVar(value=1)
        ttk.Label(params, text="Vitesse (saut de frames)").pack(anchor=tk.W, pady=(4, 0))
        skip_combo = ttk.Combobox(params, textvariable=self.skip_var, values=[1, 2, 3, 4], state="readonly", width=8)
        skip_combo.pack(anchor=tk.W, pady=(2, 8))

        self.max_frames_var = tk.IntVar(value=600)
        ttk.Label(params, text="Frames max").pack(anchor=tk.W)
        ttk.Spinbox(params, from_=100, to=2000, increment=50, textvariable=self.max_frames_var, width=10).pack(
            anchor=tk.W, pady=(2, 4)
        )

        events_params = ttk.LabelFrame(left, text="Passes / Tirs / Ballon", padding=10)
        events_params.pack(fill=tk.X, pady=8)

        self.possession_var = tk.DoubleVar(value=75.0)
        self._add_scale(events_params, "Rayon possession (px)", self.possession_var, 40, 150)

        self.pass_dist_var = tk.DoubleVar(value=28.0)
        self._add_scale(events_params, "Distance min passe (px)", self.pass_dist_var, 15, 80)

        self.pass_cooldown_var = tk.IntVar(value=8)
        ttk.Label(events_params, text="Cooldown passe (frames)").pack(anchor=tk.W, pady=(4, 0))
        ttk.Spinbox(events_params, from_=4, to=30, textvariable=self.pass_cooldown_var, width=8).pack(
            anchor=tk.W, pady=(2, 6)
        )

        self.shot_cooldown_var = tk.IntVar(value=60)
        ttk.Label(events_params, text="Cooldown tir (frames)").pack(anchor=tk.W, pady=(4, 0))
        ttk.Spinbox(events_params, from_=20, to=120, textvariable=self.shot_cooldown_var, width=8).pack(
            anchor=tk.W, pady=(2, 6)
        )

        self.owner_confirm_var = tk.IntVar(value=2)
        ttk.Label(events_params, text="Frames confirm. possession").pack(anchor=tk.W, pady=(4, 0))
        ttk.Spinbox(events_params, from_=1, to=6, textvariable=self.owner_confirm_var, width=8).pack(
            anchor=tk.W, pady=(2, 6)
        )

        self.min_poss_var = tk.IntVar(value=3)
        ttk.Label(events_params, text="Frames min avant passe").pack(anchor=tk.W, pady=(4, 0))
        ttk.Spinbox(events_params, from_=2, to=12, textvariable=self.min_poss_var, width=8).pack(
            anchor=tk.W, pady=(2, 4)
        )

        self.analyze_btn = ttk.Button(left, text="Lancer l'analyse", command=self._start_analysis)
        self.analyze_btn.pack(fill=tk.X, pady=(16, 4))

        self.open_video_btn = ttk.Button(left, text="Ouvrir vidéo annotée", command=self._open_annotated_video, state=tk.DISABLED, style="Secondary.TButton")
        self.open_video_btn.pack(fill=tk.X, pady=4)

        self.open_folder_btn = ttk.Button(left, text="Ouvrir dossier résultats", command=self._open_output_folder, state=tk.DISABLED, style="Secondary.TButton")
        self.open_folder_btn.pack(fill=tk.X, pady=4)

        self.success_btn = ttk.Button(left, text="✅ Analyse correcte (Succès)", command=self._report_success, state=tk.DISABLED)
        self.success_btn.pack(fill=tk.X, pady=(8, 2))

        self.report_btn = ttk.Button(left, text="⚠️ Signaler des erreurs (IA Apprentissage)", command=self._report_error, state=tk.DISABLED)
        self.report_btn.pack(fill=tk.X, pady=(2, 4))

        tips = ttk.LabelFrame(left, text="Conseils", padding=8)
        tips.pack(fill=tk.X, pady=(16, 10))
        ttk.Label(
            tips,
            text="• Vidéo HD 6s : frames max 180\n• Passe ratée → baisse dist. min\n• Trop de tirs → monte vitesse tir",
            justify=tk.LEFT,
            style="Muted.TLabel"
        ).pack(anchor=tk.W)

        # --- Panneau droit ---
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.summary_frame = ttk.Frame(self.notebook, padding=8)
        self.teams_frame = ttk.Frame(self.notebook, padding=8)
        self.players_frame = ttk.Frame(self.notebook, padding=8)
        self.events_frame = ttk.Frame(self.notebook, padding=8)
        self.preview_frame = ttk.Frame(self.notebook, padding=8)

        self.notebook.add(self.summary_frame, text="Résumé")
        self.notebook.add(self.teams_frame, text="Équipes")
        self.notebook.add(self.players_frame, text="Joueurs")
        self.notebook.add(self.events_frame, text="Événements")
        self.notebook.add(self.preview_frame, text="Aperçu")

        self.summary_text = tk.Text(
            self.summary_frame,
            wrap=tk.WORD,
            font=("Consolas", 11),
            height=20,
            bg="#18181b",
            fg="#f3f4f6",
            insertbackground="#ffffff",
            highlightthickness=1,
            highlightbackground="#2a2a35",
            bd=0
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        self.summary_text.insert(tk.END, "Sélectionnez une vidéo puis lancez l'analyse.\n")
        self.summary_text.config(state=tk.DISABLED)

        self.teams_tree = self._make_tree(self.teams_frame, ["Équipe", "Passes", "Tirs", "Possession (%)", "Joueurs"])
        self.players_tree = self._make_tree(
            self.players_frame,
            ["Joueur", "Équipe", "Passes", "Reçues", "Tirs", "Touches", "Distance", "Possession (%)"],
        )
        self.events_tree = self._make_tree(self.events_frame, ["Temps (s)", "Événement", "De", "Vers", "Vitesse"])

        self.preview_canvas = tk.Canvas(self.preview_frame, bg="#121214", highlightthickness=0)
        self.preview_scroll = ttk.Scrollbar(self.preview_frame, orient=tk.VERTICAL, command=self.preview_canvas.yview)
        self.preview_inner = ttk.Frame(self.preview_canvas)
        self.preview_inner.bind("<Configure>", lambda e: self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all")))
        self.preview_canvas.create_window((0, 0), window=self.preview_inner, anchor=tk.NW)
        self.preview_canvas.configure(yscrollcommand=self.preview_scroll.set)
        self.preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.preview_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Mousewheel scrolling for preview tab canvas
        def _on_preview_mousewheel(event: tk.Event) -> None:
            if event.num == 4:
                self.preview_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.preview_canvas.yview_scroll(1, "units")
            else:
                self.preview_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.preview_canvas.bind('<Enter>', lambda e: self.preview_canvas.bind_all("<MouseWheel>", _on_preview_mousewheel))
        self.preview_canvas.bind('<Leave>', lambda e: self.preview_canvas.unbind_all("<MouseWheel>"))

        # Barre de progression
        bottom = ttk.Frame(self, padding=(8, 0, 8, 8))
        bottom.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="Prêt")
        ttk.Label(bottom, textvariable=self.status_var).pack(anchor=tk.W)
        self.progress = ttk.Progressbar(bottom, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, pady=(4, 0))

    def _add_scale(self, parent: ttk.Frame, label: str, var: tk.Variable, from_: float, to: float) -> None:
        ttk.Label(parent, text=label).pack(anchor=tk.W)
        scale = ttk.Scale(parent, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL)
        scale.pack(fill=tk.X, pady=(2, 8))
        val_label = ttk.Label(parent, text="", style="Muted.TLabel")
        val_label.pack(anchor=tk.E)

        def update_label(*_args: object) -> None:
            val_label.config(text=f"{var.get():.0f}" if isinstance(var, tk.IntVar) else f"{var.get():.2f}")

        var.trace_add("write", update_label)
        update_label()

    def _make_tree(self, parent: ttk.Frame, columns: list[str]) -> ttk.Treeview:
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=18)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=110, anchor=tk.CENTER)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scroll.set)
        return tree

    def _pick_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Choisir une vidéo de match",
            filetypes=[
                ("Vidéos", "*.mp4 *.avi *.mov *.mkv"),
                ("Tous les fichiers", "*.*"),
            ],
        )
        if path:
            self.video_path = path
            self.path_var.set(Path(path).name)
            self.status_var.set(f"Vidéo sélectionnée : {Path(path).name}")

    def _start_analysis(self) -> None:
        if self._analyzing:
            return
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showwarning("Vidéo manquante", "Choisissez d'abord une vidéo.")
            return

        self._analyzing = True
        self.analyze_btn.config(state=tk.DISABLED)
        self.open_video_btn.config(state=tk.DISABLED)
        self.open_folder_btn.config(state=tk.DISABLED)
        self.success_btn.config(state=tk.DISABLED)
        self.report_btn.config(state=tk.DISABLED)
        self.progress["value"] = 0
        self.status_var.set("Analyse en cours... (1er lancement = téléchargement YOLO)")

        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()

    def _run_analysis(self) -> None:
        try:
            base_output = Path(tempfile.gettempdir()) / "match_analyzer_results"
            base_output.mkdir(parents=True, exist_ok=True)
            output_dir = tempfile.mkdtemp(prefix="analyse_", dir=base_output)

            config = AnalysisConfig(
                person_conf=self.conf_var.get(),
                ball_conf=self.ball_conf_var.get(),
                frame_skip=int(self.skip_var.get()),
                max_frames=int(self.max_frames_var.get()),
                possession_radius=self.possession_var.get(),
                pass_min_distance=self.pass_dist_var.get(),
                pass_cooldown=int(self.pass_cooldown_var.get()),
                shot_cooldown=int(self.shot_cooldown_var.get()),
                owner_confirm_frames=int(self.owner_confirm_var.get()),
                min_possession_pass=int(self.min_poss_var.get()),
                goal_zones=self.goal_zones,
            )
            analyzer = VideoAnalyzer(config)

            def on_progress(pct: float, msg: str) -> None:
                self.after(0, lambda p=pct, m=msg: self._update_progress(p, m))

            result = analyzer.analyze(self.video_path, output_dir, on_progress)
            self.after(0, lambda r=result, o=output_dir: self._on_analysis_done(r, o, None))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.after(0, lambda e=exc: self._on_analysis_done(None, None, e))

    def _update_progress(self, pct: float, msg: str) -> None:
        self.progress["value"] = pct * 100
        self.status_var.set(msg)

    def _on_analysis_done(self, result: AnalysisResult | None, output_dir: str | None, error: Exception | None) -> None:
        self._analyzing = False
        self.analyze_btn.config(state=tk.NORMAL)

        if error is not None:
            self.progress["value"] = 0
            self.status_var.set("Erreur lors de l'analyse")
            messagebox.showerror("Erreur", str(error))
            return

        self.result = result
        self.output_dir = output_dir
        self.progress["value"] = 100
        self.status_var.set("Analyse terminée !")
        self.open_video_btn.config(state=tk.NORMAL)
        self.open_folder_btn.config(state=tk.NORMAL)
        self.success_btn.config(state=tk.NORMAL)
        self.report_btn.config(state=tk.NORMAL)

        self._fill_results(result)
        messagebox.showinfo(
            "Terminé",
            f"Analyse terminée.\n"
            f"{result.stats.total_passes} passes, {result.stats.total_shots} tirs détectés.",
        )

    def _fill_results(self, result: AnalysisResult) -> None:
        stats = result.stats

        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(
            tk.END,
            f"Frames analysées : {result.frame_count}\n"
            f"Durée vidéo      : {stats.duration_sec}s\n"
            f"Passes           : {stats.total_passes}\n"
            f"Tirs             : {stats.total_shots}\n"
            f"Joueurs suivis   : {result.players_tracked}\n\n"
            f"Vidéo annotée :\n{result.annotated_video_path}\n",
        )
        self.summary_text.config(state=tk.DISABLED)

        self._fill_tree(self.teams_tree, self._teams_rows(stats))
        self._fill_tree(self.players_tree, self._players_rows(stats))
        self._fill_tree(self.events_tree, self._events_rows(stats))
        self._fill_preview(result.preview_frames)

    def _teams_rows(self, stats) -> list[tuple]:
        rows = []
        for name, t in stats.teams.items():
            if name == "inconnu" or t.players == 0:
                continue
            rows.append((name, t.passes, t.shots, round(t.possession_pct, 1), t.players))
        return rows

    def _players_rows(self, stats) -> list[tuple]:
        if stats.player_df.empty:
            return []
        rows = []
        for _, row in stats.player_df.iterrows():
            rows.append(
                (
                    row["Joueur"],
                    row["Équipe"],
                    row["Passes"],
                    row["Passes reçues"],
                    row["Tirs"],
                    row["Touches"],
                    row["Distance (px)"],
                    row["Possession (%)"],
                )
            )
        return rows

    def _events_rows(self, stats) -> list[tuple]:
        if stats.events_df.empty:
            return []
        rows = []
        for _, row in stats.events_df.iterrows():
            rows.append(
                (row["Temps (s)"], row["Événement"], row["De"], row["Vers"], row["Vitesse"])
            )
        return rows

    def _fill_tree(self, tree: ttk.Treeview, rows: list[tuple]) -> None:
        tree.delete(*tree.get_children())
        for row in rows:
            tree.insert("", tk.END, values=row)

    def _fill_preview(self, frames: list) -> None:
        for widget in self.preview_inner.winfo_children():
            widget.destroy()
        self._preview_refs.clear()

        if not frames:
            ttk.Label(self.preview_inner, text="Aucun aperçu disponible.").pack(pady=20)
            return

        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.thumbnail((480, 270))
            photo = ImageTk.PhotoImage(img)
            self._preview_refs.append(photo)
            ttk.Label(self.preview_inner, text=f"Frame {i + 1}", font=("Segoe UI", 10, "bold")).pack(pady=(12, 4))
            ttk.Label(self.preview_inner, image=photo).pack()


    def _define_goal_zones(self):
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showwarning("Vidéo manquante", "Veuillez d'abord choisir une vidéo.")
            return
        dlg = GoalSelectorDialog(self.root, self.video_path)
        self.root.wait_window(dlg)
        if len(dlg.result) > 0:
            self.goal_zones = dlg.result
            self.status_var.set(f"{len(self.goal_zones)} zones de but définies manuellement.")



    def _report_success(self) -> None:
        """Sauvegarde les paramètres actuels comme un succès pour l'apprentissage."""
        if not self.result: return
        import json
        from pathlib import Path
        import datetime
        history_file = Path("learning_history") / "learning.json"
        history_file.parent.mkdir(exist_ok=True)
        history = []
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f: history = json.load(f)
            except Exception: pass
            
        history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "video_name": Path(self.video_path).name if self.video_path else "Inconnu",
            "statut": "SUCCES",
            "stats_video": {
                "duration_sec": self.result.stats.duration_sec,
                "frame_count": self.result.frame_count,
            },
            "stats_analyse": {
                "total_passes": self.result.stats.total_passes,
                "total_shots": self.result.stats.total_shots,
                "players_tracked": self.result.players_tracked,
            },
            "erreurs_signalees": [],
            "parametres": {
                "pass_dist": self.pass_dist_var.get(),
                "possession_radius": self.possession_var.get(),
                "ball_conf": self.ball_conf_var.get(),
                "person_conf": self.conf_var.get()
            }
        })
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        messagebox.showinfo("Succès enregistré", "Merci ! L'IA a mémorisé ces paramètres comme optimaux pour cette vidéo.")

    def _report_error(self) -> None:
        """Affiche un dialogue pour signaler des événements spécifiques faux ou manquants."""
        if not self.result:
            return

        top = tk.Toplevel(self)
        top.title("Signaler des erreurs — IA Apprentissage")
        top.geometry("600x500")
        top.resizable(False, False)
        top.grab_set()

        ttk.Label(top, text="Cochez les événements qui sont FAUX (Faux Positifs) :",
                  font=("Segoe UI", 11, "bold")).pack(pady=(16, 8), padx=20, anchor=tk.W)

        # Liste des faux positifs (événements détectés mais incorrects)
        fp_vars = []
        
        container = ttk.Frame(top)
        container.pack(fill=tk.BOTH, expand=True, padx=20)
        
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        for idx, event in enumerate(self.result.events):
            var = tk.BooleanVar(value=False)
            fp_vars.append((event, var))
            lbl = f"Faux : {event.event_type.value} à {event.timestamp_sec:.1f}s (Joueur {event.from_player})"
            ttk.Checkbutton(scrollable_frame, text=lbl, variable=var).pack(anchor=tk.W, pady=2)

        if not self.result.events:
            ttk.Label(scrollable_frame, text="(Aucun événement détecté dans cette vidéo)").pack(anchor=tk.W, pady=2)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Label(top, text="Cochez ce qu'il MANQUE (Faux Négatifs) :",
                  font=("Segoe UI", 11, "bold")).pack(pady=(16, 8), padx=20, anchor=tk.W)

        miss_pass_var = tk.BooleanVar(value=False)
        miss_shot_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="L'IA a raté/ignoré une ou plusieurs Passes", variable=miss_pass_var).pack(anchor=tk.W, padx=20, pady=2)
        ttk.Checkbutton(top, text="L'IA a raté/ignoré un ou plusieurs Tirs", variable=miss_shot_var).pack(anchor=tk.W, padx=20, pady=2)

        def on_confirm():
            false_positives = [ev for ev, var in fp_vars if var.get()]
            missing_pass = miss_pass_var.get()
            missing_shot = miss_shot_var.get()
            
            if not false_positives and not missing_pass and not missing_shot:
                top.destroy()
                return
            top.destroy()
            
            # Start background optimization
            thread = threading.Thread(target=self._auto_optimize, args=(false_positives, missing_pass, missing_shot), daemon=True)
            thread.start()

        btn_frame = ttk.Frame(top)
        btn_frame.pack(pady=16)
        ttk.Button(btn_frame, text="Lancer l'Auto-Correction", command=on_confirm).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_frame, text="Annuler", command=top.destroy).pack(side=tk.LEFT, padx=8)

    def _auto_optimize(self, false_positives: list, missing_pass: bool, missing_shot: bool) -> None:
        self.after(0, lambda: self._start_opt_ui())
        
        max_attempts = 5
        attempt = 0
        success = False
        
        # Determine the initial direction of adjustment based on the errors
        adjust_pass_dist = 0
        adjust_possession = 0
        
        # Analyze what's wrong
        fp_passes = [ev for ev in false_positives if ev.event_type == EventType.PASS]
        fp_shots = [ev for ev in false_positives if ev.event_type == EventType.SHOT]
        
        if len(fp_passes) > 0:
            adjust_pass_dist = +8.0  # Fausse passe -> Augmenter la distance minimale
            adjust_possession = +2 # Fausse possession -> Augmenter le temps de confirmation
        if missing_pass:
            adjust_pass_dist = -8.0  # Passe ratée -> Baisser la distance minimale
            adjust_possession = -2 # possession ratée -> Baisser le temps de confirmation
            
        # Optimization Loop
        while attempt < max_attempts:
            attempt += 1
            self.after(0, lambda a=attempt: self.status_var.set(f"Auto-correction en cours (Essai {a}/{max_attempts})..."))
            
            # Appliquer les ajustements
            current_pd = self.pass_dist_var.get()
            current_po = int(self.owner_confirm_var.get())
            
            self.pass_dist_var.set(max(5.0, current_pd + adjust_pass_dist))
            self.owner_confirm_var.set(str(max(1, current_po + int(adjust_possession))))
            
            # Lancer analyse headless
            try:
                base_output = Path(tempfile.gettempdir()) / "match_analyzer_results"
                output_dir = tempfile.mkdtemp(prefix="analyse_opt_", dir=base_output)
                
                config = AnalysisConfig(
                    person_conf=self.conf_var.get(),
                    ball_conf=self.ball_conf_var.get(),
                    frame_skip=int(self.skip_var.get()),
                    max_frames=int(self.max_frames_var.get()),
                    possession_radius=self.possession_var.get(),
                    pass_min_distance=self.pass_dist_var.get(),
                    pass_cooldown=int(self.pass_cooldown_var.get()),
                    shot_cooldown=int(self.shot_cooldown_var.get()),
                    owner_confirm_frames=int(self.owner_confirm_var.get()),
                    min_possession_pass=int(self.min_poss_var.get()),
                    goal_zones=self.goal_zones,
                )
                analyzer = VideoAnalyzer(config)
                new_result = analyzer.analyze(self.video_path, output_dir, progress_callback=None, headless=True)
                
                # Check if the targeted false positives have disappeared
                # Since events are new objects, we check by timestamp and type
                remaining_fp = 0
                for fp in false_positives:
                    still_exists = any(
                        e.event_type == fp.event_type and abs(e.timestamp_sec - fp.timestamp_sec) < 1.0 
                        for e in new_result.events
                    )
                    if still_exists:
                        remaining_fp += 1
                        
                # Check if missing passes appeared
                new_passes = sum(1 for e in new_result.events if e.event_type == EventType.PASS)
                old_passes = self.result.stats.total_passes
                
                resolved = True
                if len(false_positives) > 0 and remaining_fp > 0:
                    resolved = False
                if missing_pass and new_passes <= old_passes:
                    resolved = False
                    
                if resolved:
                    success = True
                    break
                    
                # Not resolved, cut adjustment in half (Dichotomy) to avoid infinite bouncing
                adjust_pass_dist /= 2.0
                adjust_possession = int(adjust_possession / 2.0)
                
            except Exception as e:
                print(f"Erreur d'optimisation: {e}")
                break
                
        # Fin de la boucle
        self.after(0, lambda: self._end_opt_ui(success))

    def _start_opt_ui(self):
        self._analyzing = True
        self.analyze_btn.config(state=tk.DISABLED)
        self.open_video_btn.config(state=tk.DISABLED)
        self.open_folder_btn.config(state=tk.DISABLED)
        self.success_btn.config(state=tk.DISABLED)
        self.report_btn.config(state=tk.DISABLED)
        self.progress["mode"] = "indeterminate"
        self.progress.start()
        
    def _end_opt_ui(self, success: bool):
        self.progress.stop()
        self.progress["mode"] = "determinate"
        self._analyzing = False
        
        if success:
            messagebox.showinfo("Auto-Correction Réussie", "L'IA a trouvé les bons paramètres ! Relancement pour générer la vidéo finale...")
            self._start_analysis()
        else:
            messagebox.showwarning("Auto-Correction Échouée", "L'IA n'a pas pu corriger complètement l'erreur. Génération avec les meilleurs paramètres trouvés...")
            self._start_analysis()

    def _open_annotated_video(self) -> None:
        if self.result and os.path.exists(self.result.annotated_video_path):
            os.startfile(self.result.annotated_video_path)
        else:
            messagebox.showwarning("Fichier introuvable", "La vidéo annotée n'existe plus.")

    def _open_output_folder(self) -> None:
        if self.output_dir and os.path.isdir(self.output_dir):
            os.startfile(self.output_dir)
        else:
            messagebox.showwarning("Dossier introuvable", "Le dossier de résultats n'existe plus.")


def main() -> None:
    app = MatchAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
