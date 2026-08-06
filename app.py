"""Interface Streamlit para rodar o Agente de Modelagem.

Recebe a base de dados (CSV) e os metadados (YAML) do usuario, executa o
pipeline de 4 nos (entendimento_contexto -> engenharia_features -> modelagem
-> avaliacao) definido em `agente_modelagem.py` mostrando o progresso etapa a
etapa, e disponibiliza os resultados/artefatos de cada etapa para download.

O YAML de metadados deve conter as chaves:
    pergunta_negocio: pergunta de negocio a ser respondida pelo modelo
    coluna_target: nome da coluna alvo no CSV (default: "target")
As demais chaves sao passadas livremente ao LLM como contexto adicional.
"""

import json
import logging
import time
import base64
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from agente_modelagem import AgenteState, agente_modelagem

APP_DIR = Path(__file__).resolve().parent
ARTEFACT_WORDMARK = APP_DIR / "assets" / "logos" / "wordmark.png"
ARTEFACT_ICON = APP_DIR / "assets" / "icons" / "artefact_A_icon.png"

st.set_page_config(
    page_title="Artefact ML Studio",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="collapsed",
)


STAGES = [
    {
        "key": "entendimento_contexto",
        "titulo": "Contexto e framing",
        "icone": "01",
        "descricao": (
            "O LLM analisa a base de dados, a pergunta de negocio e os metadados "
            "para identificar o tipo de problema de ML e sugerir o tratamento "
            "necessario dos dados."
        ),
    },
    {
        "key": "engenharia_features",
        "titulo": "Design de features",
        "icone": "02",
        "descricao": (
            "O LLM sugere novas features e features a remover para a categoria "
            "de problema identificada; a base de dados e atualizada com essas mudancas."
        ),
    },
    {
        "key": "modelagem",
        "titulo": "Modelagem",
        "icone": "03",
        "descricao": (
            "O LLM sugere modelos candidatos e metricas de avaliacao; o PyCaret "
            "treina e compara os modelos de verdade e seleciona o vencedor."
        ),
    },
    {
        "key": "avaliacao",
        "titulo": "Leitura de negocio",
        "icone": "04",
        "descricao": (
            "O LLM traduz as metricas finais do melhor modelo em um relatorio "
            "acessivel para a area de negocio."
        ),
    },
]
STAGE_BY_KEY = {stage["key"]: stage for stage in STAGES}


def injetar_layout_artefact() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;600;700&display=swap');

        html {
            scroll-behavior: smooth;
        }

        :root {
            --af-deep: #0d1321;
            --af-ink: #141f31;
            --af-text: #edf3ff;
            --af-muted: #a2b5cf;
            --af-cyan: #56f4f4;
            --af-rose: #ff4fa3;
            --af-panel: rgba(13, 19, 33, 0.68);
            --af-border: rgba(255, 255, 255, 0.16);
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 18%, rgba(86, 244, 244, 0.23), transparent 35%),
                radial-gradient(circle at 88% 2%, rgba(255, 79, 163, 0.26), transparent 32%),
                linear-gradient(150deg, #050a14 0%, #0f1830 52%, #091123 100%);
            color: var(--af-text);
            font-family: 'Manrope', sans-serif;
        }

        [data-testid="stHeader"] {
            background: transparent;
            backdrop-filter: none;
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1180px;
            padding-top: 5.8rem;
            padding-bottom: 2.6rem;
        }

        .af-navbar {
            position: fixed;
            top: 0.45rem;
            left: 50%;
            transform: translateX(-50%);
            width: min(1180px, calc(100vw - 2rem));
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 14px;
            padding: 0.52rem 0.8rem;
            background:
                linear-gradient(92deg, rgba(86, 244, 244, 0.08), rgba(255, 79, 163, 0.16)),
                rgba(8, 14, 26, 0.88);
            backdrop-filter: none;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.34);
            isolation: isolate;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .af-nav-brand {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
        }

        .af-nav-brand img {
            display: block;
        }

        .af-nav-icon {
            width: 17px;
            height: 17px;
            object-fit: contain;
        }

        .af-nav-wordmark {
            height: 15px;
            width: auto;
            filter: brightness(0) invert(1) opacity(0.97);
        }

        .af-nav-links {
            display: inline-flex;
            align-items: center;
            gap: 0.34rem;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        .af-nav-links a {
            color: #ffd2e7;
            font-size: 0.75rem;
            font-family: 'Space Grotesk', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            text-decoration: none;
            border: 1px solid rgba(255, 79, 163, 0.38);
            border-radius: 999px;
            padding: 0.2rem 0.54rem;
            transition: transform 120ms ease, background 120ms ease;
        }

        .af-nav-links a:hover {
            background: rgba(255, 79, 163, 0.16);
            transform: translateY(-1px);
        }

        .af-anchor {
            position: relative;
            top: -88px;
            visibility: hidden;
            height: 0;
        }

        .af-hero {
            border: 1px solid var(--af-border);
            border-radius: 18px;
            background:
                linear-gradient(120deg, rgba(86, 244, 244, 0.14), rgba(255, 79, 163, 0.12)),
                var(--af-panel);
            box-shadow: inset 0 1px 0 rgba(255, 79, 163, 0.18), 0 16px 36px rgba(255, 79, 163, 0.1);
            padding: 1.35rem 1.5rem;
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
        }

        .af-hero::after {
            content: "";
            position: absolute;
            right: -120px;
            top: -80px;
            width: 360px;
            height: 260px;
            background: radial-gradient(circle, rgba(255, 79, 163, 0.24), transparent 68%);
            pointer-events: none;
        }

        .af-logo-wrap {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.28rem 0.56rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 79, 163, 0.35);
            background: rgba(255, 79, 163, 0.08);
            margin-bottom: 0.6rem;
        }

        .af-logo-icon {
            width: 20px;
            height: 20px;
            object-fit: contain;
            filter: brightness(1.12);
        }

        .af-logo-wordmark {
            height: 18px;
            width: auto;
            object-fit: contain;
            /* Force visibility even when source logo is dark. */
            filter: brightness(0) invert(1) opacity(0.98);
        }

        .af-brand-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.65rem;
            flex-wrap: wrap;
        }

        .af-brand-chip {
            border: 1px solid rgba(255, 79, 163, 0.45);
            border-radius: 999px;
            color: #ff9fce;
            padding: 0.2rem 0.6rem;
            font-size: 0.72rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
        }

        .af-hero-kicker {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--af-cyan);
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-size: 0.75rem;
            margin: 0;
            text-shadow: 0 0 14px rgba(255, 79, 163, 0.32);
        }

        .af-hero-title {
            font-family: 'Space Grotesk', sans-serif;
            margin: 0.25rem 0 0.45rem 0;
            color: #f7fbff;
            font-size: clamp(1.5rem, 2.35vw, 2.3rem);
            line-height: 1.15;
            font-weight: 700;
            text-shadow: 0 0 18px rgba(255, 79, 163, 0.2);
        }

        .af-hero-subtitle {
            color: var(--af-muted);
            margin: 0;
            font-size: 0.97rem;
            line-height: 1.45;
            max-width: 72ch;
        }

        .af-pipeline-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.8rem 0 1.05rem;
        }

        .af-pipeline-item {
            border: 1px solid var(--af-border);
            border-radius: 14px;
            background: rgba(9, 16, 30, 0.68);
            padding: 0.8rem 0.82rem;
            min-height: 100%;
        }

        .af-step-chip {
            display: inline-block;
            border: 1px solid rgba(255, 79, 163, 0.6);
            color: #ff92c6;
            border-radius: 999px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.73rem;
            padding: 0.16rem 0.42rem;
            margin-bottom: 0.35rem;
        }

        .af-step-title {
            color: #f8fbff;
            font-size: 0.95rem;
            font-weight: 700;
            margin: 0 0 0.3rem 0;
        }

        .af-step-desc {
            color: var(--af-muted);
            font-size: 0.82rem;
            line-height: 1.37;
            margin: 0;
        }

        .af-chip {
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--af-border);
            background: rgba(14, 25, 46, 0.78);
            color: #f3f7ff;
            padding: 0.2rem 0.52rem;
            border-radius: 999px;
            margin: 0.15rem 0.25rem 0.15rem 0;
            font-size: 0.78rem;
            font-weight: 600;
            line-height: 1.1;
        }

        .af-chip.--cyan {
            border-color: rgba(86, 244, 244, 0.5);
            color: var(--af-cyan);
        }

        .af-chip.--orange {
            border-color: rgba(255, 106, 61, 0.5);
            color: #ffb499;
        }

        .af-chip.--green {
            border-color: rgba(105, 226, 143, 0.55);
            color: #b7ffd1;
        }

        .af-chip.--red {
            border-color: rgba(255, 98, 121, 0.5);
            color: #ffc1cb;
        }

        .af-chip-wrap {
            margin: 0.2rem 0 0.4rem;
        }

        .af-setup-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.7rem;
            margin: 0 0 0.3rem;
            color: #ffe8f4;
            text-shadow: 0 0 18px rgba(255, 79, 163, 0.2);
        }

        .af-setup-subtitle {
            margin: 0 0 1rem;
            color: #b7c6dd;
        }

        .af-input-header {
            display: inline-block;
            font-family: 'Space Grotesk', sans-serif;
            color: #ffb6d8;
            border: 1px solid rgba(255, 79, 163, 0.42);
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            margin-bottom: 0.62rem;
            font-size: 0.75rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--af-border) !important;
            background: rgba(10, 18, 34, 0.52);
            border-radius: 14px;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(13, 23, 43, 0.74) !important;
            border: 1px dashed rgba(86, 244, 244, 0.38) !important;
            border-radius: 12px !important;
            min-height: 68px !important;
        }

        [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] {
            border-color: rgba(255, 79, 163, 0.46) !important;
            color: #ffd0e7 !important;
        }

        [data-testid="stCheckbox"] label,
        [data-testid="stNumberInput"] label {
            color: #e7eefc !important;
        }

        [data-testid="stBaseButton-primary"] {
            background: linear-gradient(90deg, rgba(86, 244, 244, 0.92), rgba(255, 79, 163, 0.9)) !important;
            color: #091126 !important;
            font-weight: 700 !important;
            border: none !important;
        }

        [data-testid="stMetric"] {
            background: rgba(10, 18, 34, 0.65);
            border: 1px solid var(--af-border);
            border-radius: 12px;
            padding: 0.65rem 0.8rem;
        }

        [data-testid="stStatusWidget"] {
            border-radius: 12px;
        }

        @media (max-width: 900px) {
            .af-navbar {
                padding: 0.45rem 0.65rem;
            }

            .af-nav-links {
                gap: 0.28rem;
            }
        }

        @media (max-width: 620px) {
            .af-navbar {
                flex-direction: column;
                align-items: flex-start;
                width: calc(100vw - 1rem);
                top: 0.2rem;
            }

            .af-nav-links {
                justify-content: flex-start;
            }

            [data-testid="stMainBlockContainer"] {
                padding-top: 7.2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_navbar() -> None:
    brand_html = '<span style="font-family: Space Grotesk, sans-serif; color: #ffffff; font-weight: 700;">Artefact</span>'
    if ARTEFACT_WORDMARK.exists() and ARTEFACT_ICON.exists():
        wordmark_b64 = base64.b64encode(ARTEFACT_WORDMARK.read_bytes()).decode("ascii")
        icon_b64 = base64.b64encode(ARTEFACT_ICON.read_bytes()).decode("ascii")
        brand_html = (
            f'<img src="data:image/png;base64,{icon_b64}" class="af-nav-icon" alt="Artefact icon" />'
            f'<img src="data:image/png;base64,{wordmark_b64}" class="af-nav-wordmark" alt="Artefact" />'
        )

    st.markdown(
        f"""
        <nav class="af-navbar">
            <div class="af-nav-brand">{brand_html}</div>
            <div class="af-nav-links">
                <a href="#overview">Visao geral</a>
                <a href="#setup">Setup</a>
                <a href="#resultados">Resultados</a>
            </div>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="af-hero">
            <div class="af-brand-row">
                <p class="af-hero-kicker">Artefact Data Intelligence</p>
                <span class="af-brand-chip">ML Studio</span>
            </div>
            <p class="af-hero-title">Agente de Modelagem para decisao de negocio</p>
            <p class="af-hero-subtitle">
                Carregue a base de dados e os metadados para executar um fluxo orientado por LLM
                com engenharia de features, comparacao de modelos e traducao executiva dos resultados.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_overview() -> None:
    cols = st.columns(4)
    for idx, stage in enumerate(STAGES):
        with cols[idx]:
            st.markdown(
                """
                <article class="af-pipeline-item">
                    <span class="af-step-chip">{icone}</span>
                    <p class="af-step-title">{titulo}</p>
                    <p class="af-step-desc">{descricao}</p>
                </article>
                """.format(
                    icone=escape(stage["icone"]),
                    titulo=escape(stage["titulo"]),
                    descricao=escape(stage["descricao"]),
                ),
                unsafe_allow_html=True,
            )


def _resumo_entendimento(o):
    return f"categoria identificada: **{o['categorizacao_problema']}**"


def _resumo_features(o):
    return f"{len(o['features_criadas'])} feature(s) criada(s), {len(o['features_removidas'])} removida(s)"


def _resumo_modelagem(o):
    return f"modelo vencedor: **{o['melhor_modelo_treinado']['nome_modelo']}**"


def _resumo_avaliacao(o):
    return "relatorio de negocio gerado"


STAGE_RESUMO = {
    "entendimento_contexto": _resumo_entendimento,
    "engenharia_features": _resumo_features,
    "modelagem": _resumo_modelagem,
    "avaliacao": _resumo_avaliacao,
}


def _fmt_metrica(valor):
    if isinstance(valor, (int, float)):
        return f"{valor:.3f}"
    return str(valor)


def render_chip(texto: str, variant: str = "") -> None:
    variant_class = f" --{variant}" if variant else ""
    st.markdown(
        f'<span class="af-chip{variant_class}">{escape(texto)}</span>',
        unsafe_allow_html=True,
    )


class PainelLogAoVivo(logging.Handler):
    """Handler que acumula mensagens de log e atualiza um placeholder do Streamlit
    ao vivo (com throttling), sem precisar de threads: cada chamada a emit() ocorre
    de forma sincrona dentro da mesma execucao do pipeline, e o Streamlit envia
    cada atualizacao ao navegador via websocket assim que ela acontece."""

    def __init__(self, placeholder, buffer: list, intervalo_min_s: float = 0.35):
        super().__init__()
        self.placeholder = placeholder
        self.buffer = buffer
        self.intervalo_min_s = intervalo_min_s
        self._ultimo_flush = 0.0
        self.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record):
        self.buffer.append(self.format(record))
        agora = time.time()
        if agora - self._ultimo_flush >= self.intervalo_min_s or len(self.buffer) <= 3:
            self._ultimo_flush = agora
            self.placeholder.code(
                "\n".join(self.buffer[-200:]), language="log", height=320
            )


def render_entendimento(o):
    render_chip(o["categorizacao_problema"], "cyan")
    st.markdown("**Sugestao de limpeza dos dados:**")
    st.info(o["sugestao_limpeza"])

    texto = (
        f"Categoria do problema: {o['categorizacao_problema']}\n\n"
        f"Sugestao de limpeza:\n{o['sugestao_limpeza']}\n"
    )
    st.download_button(
        "⬇️ Baixar resumo da etapa (.txt)",
        data=texto,
        file_name="1_entendimento_contexto.txt",
        mime="text/plain",
        key="dl_entendimento",
    )


def render_features(o):
    col_criadas, col_removidas = st.columns(2)
    with col_criadas:
        st.caption("Features criadas")
        if o["features_criadas"]:
            chips_criadas = "".join(
                f'<span class="af-chip --green">{escape(feature)}</span>'
                for feature in o["features_criadas"]
            )
            st.markdown(f'<div class="af-chip-wrap">{chips_criadas}</div>', unsafe_allow_html=True)
        else:
            st.caption("nenhuma")
    with col_removidas:
        st.caption("Features removidas")
        if o["features_removidas"]:
            chips_removidas = "".join(
                f'<span class="af-chip --red">{escape(feature)}</span>'
                for feature in o["features_removidas"]
            )
            st.markdown(f'<div class="af-chip-wrap">{chips_removidas}</div>', unsafe_allow_html=True)
        else:
            st.caption("nenhuma")

    st.markdown("**Base de dados atualizada (preview):**")
    st.dataframe(o["base_de_dados_nova"].head(20), use_container_width=True)

    csv_bytes = o["base_de_dados_nova"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Baixar base com novas features (.csv)",
        data=csv_bytes,
        file_name="2_base_dados_features.csv",
        mime="text/csv",
        key="dl_features",
    )


def render_modelagem(o):
    melhor = o["melhor_modelo_treinado"]
    render_chip(melhor["nome_modelo"], "orange")
    st.caption(f"Treinado com {melhor['n_linhas_treino']} linhas")

    metricas = melhor["metricas_validacao"]
    if metricas:
        cols = st.columns(len(metricas))
        for col, (nome, valor) in zip(cols, metricas.items()):
            col.metric(nome, _fmt_metrica(valor))

    st.markdown("**Modelos candidatos sugeridos pelo LLM:**")
    chips_modelos = "".join(
        f'<span class="af-chip">{escape(modelo)}</span>' for modelo in o["modelos_sugeridos"]
    )
    st.markdown(f'<div class="af-chip-wrap">{chips_modelos}</div>', unsafe_allow_html=True)

    with st.expander("Ver hiperparametros do modelo vencedor"):
        st.write(melhor["hiperparametros"])

    payload = json.dumps(
        {
            "modelos_sugeridos": o["modelos_sugeridos"],
            "metricas_avaliacao": o["metricas_avaliacao"],
            "melhor_modelo_treinado": melhor,
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    st.download_button(
        "⬇️ Baixar resultado da modelagem (.json)",
        data=payload,
        file_name="3_modelagem_resultado.json",
        mime="application/json",
        key="dl_modelagem",
    )

    pycaret_log_texto = o.get("pycaret_log_texto")
    if pycaret_log_texto:
        with st.expander("🪵 Ver log detalhado do PyCaret (nivel DEBUG)"):
            st.code(pycaret_log_texto[-20000:], language="log", height=320)
        st.download_button(
            "⬇️ Baixar log detalhado do PyCaret (.log)",
            data=pycaret_log_texto,
            file_name="3_pycaret_detalhado.log",
            mime="text/plain",
            key="dl_pycaret_log",
        )


def render_avaliacao(o):
    metricas = o["metricas_finais"]
    if metricas:
        cols = st.columns(len(metricas))
        for col, (nome, valor) in zip(cols, metricas.items()):
            col.metric(nome, _fmt_metrica(valor))

    st.markdown("**Relatorio para a area de negocio:**")
    st.success(o["relatorio_business"])

    st.download_button(
        "⬇️ Baixar relatorio final (.txt)",
        data=o["relatorio_business"],
        file_name="4_relatorio_negocio.txt",
        mime="text/plain",
        key="dl_avaliacao",
    )


RENDERERS = {
    "entendimento_contexto": render_entendimento,
    "engenharia_features": render_features,
    "modelagem": render_modelagem,
    "avaliacao": render_avaliacao,
}


def montar_relatorio_consolidado(outputs: dict) -> str:
    ent = outputs["entendimento_contexto"]
    feat = outputs["engenharia_features"]
    mod = outputs["modelagem"]
    ava = outputs["avaliacao"]
    melhor = mod["melhor_modelo_treinado"]

    linhas = [
        "# Relatorio do Agente de Modelagem",
        "",
        "## 1. Entendimento de contexto",
        f"- Categoria do problema: **{ent['categorizacao_problema']}**",
        f"- Sugestao de limpeza: {ent['sugestao_limpeza']}",
        "",
        "## 2. Engenharia de features",
        f"- Features criadas: {', '.join(feat['features_criadas']) or 'nenhuma'}",
        f"- Features removidas: {', '.join(feat['features_removidas']) or 'nenhuma'}",
        "",
        "## 3. Modelagem",
        f"- Modelos sugeridos: {', '.join(mod['modelos_sugeridos'])}",
        f"- Metricas recomendadas: {', '.join(mod['metricas_avaliacao'])}",
        f"- Modelo vencedor: **{melhor['nome_modelo']}** ({melhor['n_linhas_treino']} linhas de treino)",
        f"- Metricas de validacao: {melhor['metricas_validacao']}",
        "",
        "## 4. Avaliacao",
        f"- Metricas finais: {ava['metricas_finais']}",
        "",
        "### Relatorio para o negocio",
        ava["relatorio_business"],
        "",
    ]
    return "\n".join(linhas)


def render_resultados(outputs: dict, logs: list):
    st.markdown('<div id="resultados" class="af-anchor"></div>', unsafe_allow_html=True)
    st.markdown("### Resultado por etapa")
    for stage in STAGES:
        o = outputs.get(stage["key"])
        if o is None:
            continue
        with st.container(border=True):
            st.markdown(f"#### [{stage['icone']}] {stage['titulo']}")
            st.caption(stage["descricao"])
            RENDERERS[stage["key"]](o)

    with st.expander("📄 Log de execucao completo"):
        log_texto = "\n".join(logs)
        st.code(log_texto or "sem eventos registrados", language="log", height=320)
        st.download_button(
            "⬇️ Baixar log (.txt)",
            data=log_texto,
            file_name="log_execucao.txt",
            mime="text/plain",
            key="dl_log",
        )

    st.divider()
    relatorio_md = montar_relatorio_consolidado(outputs)
    st.download_button(
        "⬇️ Baixar relatorio completo (.md)",
        data=relatorio_md,
        file_name="relatorio_completo.md",
        mime="text/markdown",
        type="primary",
        key="dl_relatorio_completo",
    )


injetar_layout_artefact()
render_navbar()
st.markdown('<div id="overview" class="af-anchor"></div>', unsafe_allow_html=True)
render_hero()
render_pipeline_overview()

st.markdown('<div id="setup" class="af-anchor"></div>', unsafe_allow_html=True)
with st.container(border=True):
    st.markdown('<h2 class="af-setup-title">Setup da execucao</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="af-setup-subtitle">Configure os inputs do pipeline e execute o fluxo de modelagem.</p>',
        unsafe_allow_html=True,
    )
    col_inputs, col_cfg = st.columns((1.2, 1), gap="large")

    with col_inputs:
        st.markdown('<div class="af-input-header">Dados de entrada</div>', unsafe_allow_html=True)
        st.caption("Arquivos obrigatorios")
        csv_file = st.file_uploader("Base de dados (CSV)", type=["csv"])
        yaml_file = st.file_uploader("Metadados (YAML)", type=["yaml", "yml"])
        st.caption(
            "O arquivo YAML deve conter a chave pergunta_negocio e pode incluir "
            "contexto adicional para o LLM."
        )

    with col_cfg:
        st.markdown('<div class="af-input-header">Treinamento</div>', unsafe_allow_html=True)
        st.caption("Configuracao de treino")
        modo_rapido = st.checkbox(
            "Modo rapido para validar o fluxo ponta a ponta",
            value=True,
        )
        if modo_rapido:
            col_n_modelos, col_fold = st.columns(2)
            with col_n_modelos:
                n_modelos_max = st.number_input(
                    "Maximo de modelos", min_value=1, max_value=17, value=3
                )
            with col_fold:
                fold = st.number_input(
                    "Folds", min_value=2, max_value=10, value=3
                )
            st.caption("Prioriza velocidade para testes de fluxo.")
        else:
            n_modelos_max = None
            fold = 10
            st.caption("Executa comparacao completa de modelos com 10 folds.")

    pronto_para_rodar = csv_file is not None and yaml_file is not None
    if st.button("Executar pipeline", disabled=not pronto_para_rodar, type="primary"):
        base_de_dados = pd.read_csv(csv_file)
        metadados = yaml.safe_load(yaml_file) or {}
        pergunta_negocio = metadados.pop("pergunta_negocio", "")

        if not pergunta_negocio:
            st.error("O YAML de metadados precisa ter a chave 'pergunta_negocio'.")
            st.stop()

        estado_inicial: AgenteState = {
            "base_de_dados": base_de_dados,
            "pergunta_negocio": pergunta_negocio,
            "metadados": metadados,
        }

        st.session_state.pop("outputs", None)
        st.session_state.pop("logs", None)

        outputs = {}
        log_buffer = []
        progresso = st.container()
        with progresso:
            st.markdown("**Log ao vivo da execucao** (inclui o log detalhado do PyCaret):")
            live_placeholder = st.empty()

        handler = PainelLogAoVivo(live_placeholder, log_buffer)
        am_logger = logging.getLogger("agente_modelagem")
        am_logger.addHandler(handler)
        am_logger.setLevel(logging.INFO)

        try:
            with progresso:
                for update in agente_modelagem.stream(
                    estado_inicial,
                    stream_mode="updates",
                    config={
                        "configurable": {
                            "log_handler": handler,
                            "pycaret_fold": fold,
                            "pycaret_max_modelos": n_modelos_max,
                        }
                    },
                ):
                    (node_key, node_output), = update.items()
                    stage = STAGE_BY_KEY[node_key]
                    outputs[node_key] = node_output
                    resumo = STAGE_RESUMO[node_key](node_output)
                    with st.status(
                        f"{stage['icone']} {stage['titulo']} — concluida", state="complete"
                    ):
                        st.write(stage["descricao"])
                        st.markdown(resumo)
        except Exception as erro:
            st.error(f"Erro ao executar o pipeline: {erro}")
            st.stop()
        finally:
            am_logger.removeHandler(handler)

        st.session_state["outputs"] = outputs
        st.session_state["logs"] = log_buffer
        st.toast("Pipeline concluido!", icon="✅")

if not pronto_para_rodar and "outputs" not in st.session_state:
    st.info("Envie os arquivos CSV e YAML para habilitar a execucao do pipeline.")

if "outputs" in st.session_state:
    render_resultados(st.session_state["outputs"], st.session_state["logs"])
