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
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from fpdf import FPDF

from agente_modelagem import AgenteState, agente_modelagem

st.set_page_config(page_title="Agente de Modelagem", page_icon="🧭", layout="wide")

_FONTS_DIR = Path(__file__).parent / "assets" / "fonts"


STAGES = [
    {
        "key": "entendimento_contexto",
        "titulo": "Entendimento de contexto",
        "icone": "🧭",
        "descricao": (
            "O LLM analisa a base de dados, a pergunta de negocio e os metadados "
            "para identificar o tipo de problema de ML e sugerir o tratamento "
            "necessario dos dados."
        ),
    },
    {
        "key": "engenharia_features",
        "titulo": "Engenharia de features",
        "icone": "🛠️",
        "descricao": (
            "O LLM sugere novas features e features a remover para a categoria "
            "de problema identificada; a base de dados e atualizada com essas mudancas."
        ),
    },
    {
        "key": "modelagem",
        "titulo": "Modelagem",
        "icone": "🤖",
        "descricao": (
            "O LLM sugere modelos candidatos e metricas de avaliacao; o PyCaret "
            "treina e compara os modelos de verdade e seleciona o vencedor."
        ),
    },
    {
        "key": "avaliacao",
        "titulo": "Avaliacao",
        "icone": "📊",
        "descricao": (
            "O LLM traduz as metricas finais do melhor modelo em um relatorio "
            "acessivel para a area de negocio."
        ),
    },
]
STAGE_BY_KEY = {stage["key"]: stage for stage in STAGES}


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
    st.badge(o["categorizacao_problema"], color="blue", icon=":material/travel_explore:")
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
            for feature in o["features_criadas"]:
                st.badge(feature, color="green", icon=":material/add_circle:")
        else:
            st.caption("nenhuma")
    with col_removidas:
        st.caption("Features removidas")
        if o["features_removidas"]:
            for feature in o["features_removidas"]:
                st.badge(feature, color="red", icon=":material/remove_circle:")
        else:
            st.caption("nenhuma")

    st.markdown("**Base de dados atualizada (preview):**")
    st.dataframe(o["base_de_dados_nova"].head(20), width="stretch")

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
    st.badge(melhor["nome_modelo"], color="violet", icon=":material/trophy:")
    st.caption(f"Treinado com {melhor['n_linhas_treino']} linhas")

    metricas = melhor["metricas_validacao"]
    if metricas:
        cols = st.columns(len(metricas))
        for col, (nome, valor) in zip(cols, metricas.items()):
            col.metric(nome, _fmt_metrica(valor))

    st.markdown("**Modelos candidatos sugeridos pelo LLM:**")
    for modelo in o["modelos_sugeridos"]:
        st.badge(modelo, color="gray")

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

    st.markdown("**Resposta a pergunta de negocio:**")
    st.success(o["resposta_pergunta_negocio"])

    st.markdown("**O que as metricas significam na pratica:**")
    st.info(o["interpretacao_metricas"])

    st.markdown("**Acoes recomendadas:**")
    for acao in o["acoes_recomendadas"]:
        st.markdown(f"- {acao}")

    st.markdown("**Riscos e limitacoes:**")
    st.warning(o["riscos_e_limitacoes"])

    texto_avaliacao = (
        f"Resposta a pergunta de negocio:\n{o['resposta_pergunta_negocio']}\n\n"
        f"O que as metricas significam na pratica:\n{o['interpretacao_metricas']}\n\n"
        "Acoes recomendadas:\n"
        + "\n".join(f"- {acao}" for acao in o["acoes_recomendadas"])
        + f"\n\nRiscos e limitacoes:\n{o['riscos_e_limitacoes']}\n"
    )
    st.download_button(
        "⬇️ Baixar relatorio final (.txt)",
        data=texto_avaliacao,
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


def montar_relatorio_consolidado(outputs: dict, pergunta_negocio: str) -> str:
    ent = outputs["entendimento_contexto"]
    feat = outputs["engenharia_features"]
    mod = outputs["modelagem"]
    ava = outputs["avaliacao"]
    melhor = mod["melhor_modelo_treinado"]

    linhas = [
        "# Relatorio do Agente de Modelagem",
        "",
        f"**Pergunta de negocio:** {pergunta_negocio}",
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
        "### Resposta a pergunta de negocio",
        ava["resposta_pergunta_negocio"],
        "",
        "### O que as metricas significam na pratica",
        ava["interpretacao_metricas"],
        "",
        "### Acoes recomendadas",
        *[f"- {acao}" for acao in ava["acoes_recomendadas"]],
        "",
        "### Riscos e limitacoes",
        ava["riscos_e_limitacoes"],
        "",
    ]
    return "\n".join(linhas)


class RelatorioPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("DejaVu", "", str(_FONTS_DIR / "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", str(_FONTS_DIR / "DejaVuSans-Bold.ttf"))
        self.set_auto_page_break(auto=True, margin=18)

    def titulo(self, texto, tamanho=16):
        self.set_font("DejaVu", "B", tamanho)
        self.multi_cell(0, 8, texto, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def secao(self, texto):
        self.ln(2)
        self.set_font("DejaVu", "B", 12)
        self.multi_cell(0, 7, texto, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def paragrafo(self, texto):
        self.set_font("DejaVu", "", 10.5)
        self.multi_cell(0, 6, str(texto), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def bullet(self, texto):
        self.set_font("DejaVu", "", 10.5)
        self.multi_cell(0, 6, f"-  {texto}", new_x="LMARGIN", new_y="NEXT")


def montar_relatorio_pdf(outputs: dict, pergunta_negocio: str) -> bytes:
    ent = outputs["entendimento_contexto"]
    feat = outputs["engenharia_features"]
    mod = outputs["modelagem"]
    ava = outputs["avaliacao"]
    melhor = mod["melhor_modelo_treinado"]

    pdf = RelatorioPDF()
    pdf.add_page()

    pdf.titulo("Relatorio do Agente de Modelagem")
    pdf.set_font("DejaVu", "", 10.5)
    pdf.multi_cell(0, 6, f"Pergunta de negocio: {pergunta_negocio}", new_x="LMARGIN", new_y="NEXT")

    pdf.secao("1. Entendimento de contexto")
    pdf.paragrafo(f"Categoria do problema: {ent['categorizacao_problema']}")
    pdf.paragrafo(f"Sugestao de limpeza: {ent['sugestao_limpeza']}")

    pdf.secao("2. Engenharia de features")
    pdf.paragrafo(f"Features criadas: {', '.join(feat['features_criadas']) or 'nenhuma'}")
    pdf.paragrafo(f"Features removidas: {', '.join(feat['features_removidas']) or 'nenhuma'}")

    pdf.secao("3. Modelagem")
    pdf.paragrafo(f"Modelos sugeridos: {', '.join(mod['modelos_sugeridos'])}")
    pdf.paragrafo(f"Metricas recomendadas: {', '.join(mod['metricas_avaliacao'])}")
    pdf.paragrafo(f"Modelo vencedor: {melhor['nome_modelo']} ({melhor['n_linhas_treino']} linhas de treino)")
    metricas_fmt = ", ".join(f"{k}={_fmt_metrica(v)}" for k, v in melhor["metricas_validacao"].items())
    pdf.paragrafo(f"Metricas de validacao: {metricas_fmt}")

    pdf.secao("4. Avaliacao")
    metricas_finais_fmt = ", ".join(f"{k}={_fmt_metrica(v)}" for k, v in ava["metricas_finais"].items())
    pdf.paragrafo(f"Metricas finais: {metricas_finais_fmt}")

    pdf.secao("Resposta a pergunta de negocio")
    pdf.paragrafo(ava["resposta_pergunta_negocio"])

    pdf.secao("O que as metricas significam na pratica")
    pdf.paragrafo(ava["interpretacao_metricas"])

    pdf.secao("Acoes recomendadas")
    for acao in ava["acoes_recomendadas"]:
        pdf.bullet(acao)

    pdf.secao("Riscos e limitacoes")
    pdf.paragrafo(ava["riscos_e_limitacoes"])

    return bytes(pdf.output())


def render_resultados(outputs: dict, logs: list, pergunta_negocio: str):
    st.header("Resultados por etapa")
    for stage in STAGES:
        o = outputs.get(stage["key"])
        if o is None:
            continue
        with st.container(border=True):
            st.markdown(f"#### {stage['icone']} {stage['titulo']}")
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
    relatorio_md = montar_relatorio_consolidado(outputs, pergunta_negocio)
    relatorio_pdf = montar_relatorio_pdf(outputs, pergunta_negocio)
    col_md, col_pdf = st.columns(2)
    with col_md:
        st.download_button(
            "⬇️ Baixar relatorio completo (.md)",
            data=relatorio_md,
            file_name="relatorio_completo.md",
            mime="text/markdown",
            key="dl_relatorio_completo",
        )
    with col_pdf:
        st.download_button(
            "⬇️ Baixar relatorio completo (.pdf)",
            data=relatorio_pdf,
            file_name="relatorio_completo.pdf",
            mime="application/pdf",
            type="primary",
            key="dl_relatorio_pdf",
        )


st.title("🧭 Agente de Modelagem")
st.caption(
    "Pipeline de ML orquestrado com LangGraph: entendimento de contexto → "
    "engenharia de features → modelagem → avaliacao."
)

with st.container(border=True):
    st.markdown("##### 1. Envie seus arquivos")
    col_csv, col_yaml = st.columns(2)
    with col_csv:
        csv_file = st.file_uploader("Base de dados (CSV)", type=["csv"])
    with col_yaml:
        yaml_file = st.file_uploader("Metadados (YAML)", type=["yaml", "yml"])

    with st.expander("⚙️ Configuracoes do PyCaret (velocidade x exaustividade)"):
        modo_rapido = st.checkbox(
            "⚡ Modo rapido (menos modelos e menos folds -- ideal para testar o fluxo de ponta a ponta)",
            value=True,
        )
        if modo_rapido:
            col_n_modelos, col_fold = st.columns(2)
            with col_n_modelos:
                n_modelos_max = st.number_input(
                    "Numero maximo de modelos testados", min_value=1, max_value=17, value=3
                )
            with col_fold:
                fold = st.number_input(
                    "Folds de validacao cruzada", min_value=2, max_value=10, value=3
                )
            st.caption(
                "Assim que o fluxo estiver validado, desmarque o modo rapido para "
                "rodar a comparacao completa (todos os modelos, 10 folds)."
            )
        else:
            n_modelos_max = None
            fold = 10
            st.caption("Vai testar todos os modelos disponiveis com 10 folds -- pode levar varios minutos.")

    pronto_para_rodar = csv_file is not None and yaml_file is not None
    if st.button("▶️ Rodar pipeline", disabled=not pronto_para_rodar, type="primary"):
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
        st.session_state["pergunta_negocio"] = pergunta_negocio
        st.toast("Pipeline concluido!", icon="✅")

if not pronto_para_rodar and "outputs" not in st.session_state:
    st.info("Faca upload do CSV e do YAML de metadados para comecar.")

if "outputs" in st.session_state:
    render_resultados(
        st.session_state["outputs"], st.session_state["logs"], st.session_state["pergunta_negocio"]
    )
