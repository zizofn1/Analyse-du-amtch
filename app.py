"""
Analyseur de Match Football — Interface Streamlit
Téléversez une courte vidéo de match pour obtenir les statistiques.
"""

import os
import tempfile

import plotly.express as px
import streamlit as st

from analyzer import VideoAnalyzer
from analyzer.video_analyzer import AnalysisConfig

st.set_page_config(
    page_title="Analyseur de Match Football",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Analyseur de Match Football")
st.markdown(
    """
    Téléversez une **courte vidéo** de match (30 sec – 2 min recommandé).
    L'application détecte les joueurs et le ballon, puis calcule :
    **passes**, **tirs**, **possession** et **distance parcourue**.
    """
)

with st.sidebar:
    st.header("Paramètres")
    conf = st.slider("Seuil de confiance YOLO", 0.2, 0.8, 0.35, 0.05)
    frame_skip = st.selectbox("Vitesse d'analyse", [1, 2, 3, 4], index=1, help="Plus élevé = plus rapide, moins précis")
    max_frames = st.number_input("Frames max à analyser", 100, 2000, 600, 50)
    possession_radius = st.slider("Rayon de possession (px)", 40, 150, 80, 5)

    st.divider()
    st.info(
        "**Conseils :**\n"
        "- Vidéo stable (pas de zoom)\n"
        "- Joueurs visibles en entier\n"
        "- Ballon visible quand possible\n"
        "- Résolution 720p minimum"
    )

uploaded = st.file_uploader("Choisir une vidéo (.mp4, .avi, .mov)", type=["mp4", "avi", "mov", "mkv"])

if uploaded is not None:
    st.video(uploaded)

    if st.button("Lancer l'analyse", type="primary", use_container_width=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, uploaded.name)
            with open(video_path, "wb") as f:
                f.write(uploaded.getbuffer())

            config = AnalysisConfig(
                person_conf=conf,
                frame_skip=frame_skip,
                max_frames=max_frames,
                possession_radius=possession_radius,
            )
            analyzer = VideoAnalyzer(config)

            progress_bar = st.progress(0)
            status = st.empty()

            def on_progress(pct: float, msg: str) -> None:
                progress_bar.progress(min(pct, 1.0))
                status.text(msg)

            try:
                with st.spinner("Analyse en cours... (le modèle YOLO se télécharge au 1er lancement)"):
                    result = analyzer.analyze(video_path, tmpdir, on_progress)

                progress_bar.progress(1.0)
                status.text("Analyse terminée !")
                stats = result.stats

                st.success(f"✅ {result.frame_count} frames analysées en {stats.duration_sec}s de vidéo")

                # Métriques globales
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Passes", stats.total_passes)
                c2.metric("Tirs", stats.total_shots)
                c3.metric("Joueurs détectés", len(stats.player_df))
                c4.metric("Durée", f"{stats.duration_sec}s")

                st.divider()

                col_left, col_right = st.columns(2)

                with col_left:
                    st.subheader("Statistiques par équipe")
                    team_data = []
                    for name, t in stats.teams.items():
                        if name == "inconnu" or t.players == 0:
                            continue
                        team_data.append(
                            {
                                "Équipe": name,
                                "Passes": t.passes,
                                "Tirs": t.shots,
                                "Possession (%)": round(t.possession_pct, 1),
                                "Joueurs": t.players,
                            }
                        )
                    if team_data:
                        import pandas as pd

                        team_df = pd.DataFrame(team_data)
                        st.dataframe(team_df, use_container_width=True, hide_index=True)

                        fig = px.bar(
                            team_df.melt(id_vars="Équipe", value_vars=["Passes", "Tirs"]),
                            x="Équipe",
                            y="value",
                            color="variable",
                            barmode="group",
                            title="Passes et tirs par équipe",
                            labels={"value": "Nombre", "variable": "Stat"},
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with col_right:
                    st.subheader("Possession par équipe")
                    poss_data = [
                        {"Équipe": name, "Possession": round(t.possession_pct, 1)}
                        for name, t in stats.teams.items()
                        if name != "inconnu" and t.players > 0
                    ]
                    if poss_data:
                        import pandas as pd

                        poss_df = pd.DataFrame(poss_data)
                        fig_pie = px.pie(poss_df, values="Possession", names="Équipe", title="Répartition possession")
                        st.plotly_chart(fig_pie, use_container_width=True)

                st.subheader("Statistiques par joueur")
                if not stats.player_df.empty:
                    st.dataframe(stats.player_df, use_container_width=True, hide_index=True)

                    fig_players = px.bar(
                        stats.player_df,
                        x="Joueur",
                        y=["Passes", "Tirs", "Touches"],
                        title="Stats individuelles",
                        barmode="group",
                    )
                    st.plotly_chart(fig_players, use_container_width=True)
                else:
                    st.warning("Aucun joueur détecté. Essayez une vidéo plus claire ou baissez le seuil de confiance.")

                st.subheader("Chronologie des événements")
                if not stats.events_df.empty:
                    st.dataframe(stats.events_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucun événement détecté sur cette séquence.")

                st.subheader("Aperçu annoté")
                if result.preview_frames:
                    cols = st.columns(min(3, len(result.preview_frames)))
                    for i, frame in enumerate(result.preview_frames[:6]):
                        cols[i % 3].image(frame, caption=f"Frame {i + 1}", use_container_width=True)

                st.subheader("Vidéo annotée")
                if os.path.exists(result.annotated_video_path):
                    with open(result.annotated_video_path, "rb") as vf:
                        st.video(vf.read())

            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {e}")
                st.exception(e)

else:
    st.markdown(
        """
        ### Comment ça marche ?

        1. **Détection** — YOLOv8 détecte les joueurs (personnes) et le ballon
        2. **Suivi** — Chaque joueur reçoit un ID unique frame par frame
        3. **Événements** — Les passes sont détectées quand le ballon change de joueur
        4. **Tirs** — Détectés quand le ballon accélère soudainement
        5. **Stats** — Agrégation par joueur et par équipe (gauche/droite du terrain)

        > **Note :** C'est un prototype basé sur la vision par ordinateur.
        > Pour des stats professionnelles (comme Wyscout/StatsBomb), il faudrait
        > des modèles entraînés spécifiquement sur le football et des caméras fixes.
        """
    )
