"""Agente de Modelagem: pipeline sequencial de ML orquestrado com LangGraph.

Os 4 nos (entendimento_contexto -> engenharia_features -> modelagem -> avaliacao)
chamam um LLM real (via `llm_client.get_llm()`) com `.with_structured_output(...)`
para as partes qualitativas, e o no de modelagem tambem roda um AutoML real
(PyCaret) para treinar e comparar modelos candidatos.
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from llm_client import get_llm

logger = logging.getLogger("agente_modelagem")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)




class EntendimentoContextoOutput(BaseModel):
    categorizacao_problema: str = Field(
        description=(
            "Categoria do problema de machine learning identificada a partir "
            "da pergunta de negocio e dos metadados (ex.: 'classificacao_binaria', "
            "'regressao', 'series_temporais', 'clusterizacao')."
        )
    )
    sugestao_limpeza: str = Field(
        description=(
            "Sugestao textual, em linguagem de negocio, sobre como tratar a base "
            "de dados: valores faltantes, outliers, tipos de coluna incorretos, "
            "duplicatas, etc."
        )
    )


class EngenhariaFeaturesOutput(BaseModel):
    features_criadas: List[str] = Field(
        default_factory=list,
        description="Nomes das novas features criadas e uma breve justificativa de cada uma.",
    )
    features_removidas: List[str] = Field(
        default_factory=list,
        description="Nomes das features originais removidas por serem irrelevantes ou redundantes.",
    )
    justificativa: str = Field(
        description="Racional de negocio/estatistico para as transformacoes aplicadas na base.",
    )


class ModelagemOutput(BaseModel):
    modelos_sugeridos: List[str] = Field(
        description="Lista de modelos candidatos sugeridos para a categoria do problema identificada."
    )
    metricas_avaliacao: List[str] = Field(
        description="Lista de metricas recomendadas para avaliar os modelos candidatos."
    )
    melhor_modelo_treinado: Dict[str, Any] = Field(
        description=(
            "Representacao do resultado da execucao do AutoML/PyCaret local: "
            "nome do modelo vencedor, hiperparametros e metricas de validacao."
        )
    )


class ModelagemSugestaoOutput(BaseModel):
    """Saida do LLM para a etapa de modelagem (a parte qualitativa, antes do AutoML rodar)."""

    modelos_sugeridos: List[str] = Field(
        description="Lista de modelos candidatos sugeridos para a categoria do problema identificada."
    )
    metricas_avaliacao: List[str] = Field(
        description="Lista de metricas recomendadas para avaliar os modelos candidatos."
    )


class AvaliacaoRelatorioOutput(BaseModel):
    """Saida do LLM para a etapa de avaliacao -- um relatorio acionavel para um
    analista de dados, ancorado na pergunta de negocio original e nao apenas
    nas metricas cruas do modelo."""

    resposta_pergunta_negocio: str = Field(
        description=(
            "Resposta direta e objetiva a pergunta de negocio original, usando os "
            "resultados do modelo (nao um resumo tecnico generico -- responda a "
            "pergunta especificamente)."
        )
    )
    interpretacao_metricas: str = Field(
        description=(
            "Interpretacao das metricas de validacao em termos de impacto pratico "
            "para o negocio (ex.: quantos casos o modelo acerta/erra na pratica, "
            "o que isso custa ou economiza), nao apenas a definicao tecnica da metrica."
        )
    )
    acoes_recomendadas: List[str] = Field(
        description=(
            "Lista de acoes concretas e acionaveis que a area de negocio deve tomar "
            "com base neste modelo (ex.: quais segmentos/clientes priorizar, que "
            "processo ou campanha acionar, que limiar de score usar, quem deve agir)."
        )
    )
    riscos_e_limitacoes: str = Field(
        description=(
            "Riscos, limitacoes e cuidados antes de colocar o modelo em producao "
            "(ex.: qualidade/vies dos dados, necessidade de reavaliacao periodica, "
            "cenarios onde o modelo pode falhar)."
        )
    )


class AvaliacaoOutput(BaseModel):
    metricas_finais: Dict[str, float] = Field(
        description="Dicionario com as metricas finais de avaliacao do melhor modelo (ex.: accuracy, f1, rmse)."
    )
    resposta_pergunta_negocio: str
    interpretacao_metricas: str
    acoes_recomendadas: List[str]
    riscos_e_limitacoes: str


# ---------------------------------------------------------------------------
# Estado global do grafo
# ---------------------------------------------------------------------------


class AgenteState(TypedDict, total=False):
    # Inputs iniciais
    base_de_dados: pd.DataFrame
    pergunta_negocio: str
    metadados: Dict[str, Any]

    # Saida do no "Entendimento de contexto"
    categorizacao_problema: str
    sugestao_limpeza: str

    # Saida do no "Engenharia de Features"
    base_de_dados_nova: pd.DataFrame
    features_criadas: List[str]
    features_removidas: List[str]

    # Saida do no "Modelagem"
    modelos_sugeridos: List[str]
    metricas_avaliacao: List[str]
    melhor_modelo_treinado: Dict[str, Any]
    pycaret_log_texto: str

    # Saida do no "Avaliacao"
    metricas_finais: Dict[str, float]
    resposta_pergunta_negocio: str
    interpretacao_metricas: str
    acoes_recomendadas: List[str]
    riscos_e_limitacoes: str


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _resumo_dataframe(df: pd.DataFrame, linhas_amostra: int = 5) -> str:
    return (
        f"Formato: {df.shape[0]} linhas x {df.shape[1]} colunas\n"
        f"Colunas e tipos: {df.dtypes.astype(str).to_dict()}\n"
        f"Valores faltantes por coluna: {df.isna().sum().to_dict()}\n"
        f"Amostra ({min(linhas_amostra, len(df))} primeiras linhas):\n"
        f"{df.head(linhas_amostra).to_string(index=False)}"
    )


def montar_prompt_entendimento(
    base_de_dados: pd.DataFrame, pergunta_negocio: str, metadados: Dict[str, Any]
) -> str:
    return (
        "Voce e um cientista de dados senior planejando um projeto de machine learning.\n\n"
        f"Pergunta de negocio: {pergunta_negocio}\n\n"
        f"Metadados fornecidos: {json.dumps(metadados, ensure_ascii=False, default=str)}\n\n"
        f"{_resumo_dataframe(base_de_dados)}\n\n"
        "Categorize o problema de ML (ex.: 'classificacao_binaria', "
        "'classificacao_multiclasse', 'regressao', 'series_temporais', 'clusterizacao') "
        "e sugira, em linguagem de negocio, como tratar a base de dados (valores "
        "faltantes, outliers, tipos incorretos, duplicatas) antes da engenharia de features."
    )


def montar_prompt_features(
    categorizacao_problema: str, base_de_dados: pd.DataFrame, metadados: Dict[str, Any]
) -> str:
    return (
        "Voce e um cientista de dados senior fazendo engenharia de features.\n\n"
        f"Categoria do problema: {categorizacao_problema}\n"
        f"Metadados fornecidos: {json.dumps(metadados, ensure_ascii=False, default=str)}\n\n"
        f"{_resumo_dataframe(base_de_dados)}\n\n"
        "Sugira quais novas features criar (com breve justificativa) e quais features "
        "originais remover por serem irrelevantes ou redundantes para esse tipo de problema."
    )


def montar_prompt_modelagem(categorizacao_problema: str, base_de_dados_nova: pd.DataFrame) -> str:
    return (
        "Voce e um cientista de dados senior escolhendo modelos candidatos.\n\n"
        f"Categoria do problema: {categorizacao_problema}\n\n"
        f"{_resumo_dataframe(base_de_dados_nova)}\n\n"
        "Sugira uma lista de modelos candidatos adequados para essa categoria de "
        "problema e uma lista de metricas recomendadas para avalia-los."
    )


def montar_prompt_avaliacao(
    pergunta_negocio: str,
    categorizacao_problema: str,
    metadados: Dict[str, Any],
    features_criadas: List[str],
    features_removidas: List[str],
    melhor_modelo_treinado: Dict[str, Any],
) -> str:
    return (
        "Voce e um cientista de dados senior escrevendo o relatorio final de um "
        "projeto de ML para um analista de dados que vai decidir os proximos passos "
        "com a area de negocio. O relatorio precisa ser especifico e acionavel -- "
        "nunca generico ou hipotetico.\n\n"
        f"Pergunta de negocio original: {pergunta_negocio}\n"
        f"Categoria do problema: {categorizacao_problema}\n"
        f"Contexto/metadados do dominio: {json.dumps(metadados, ensure_ascii=False, default=str)}\n"
        f"Features criadas na engenharia de features: {features_criadas}\n"
        f"Features removidas na engenharia de features: {features_removidas}\n"
        f"Resultado do treinamento (modelo vencedor e metricas de validacao): "
        f"{json.dumps(melhor_modelo_treinado, ensure_ascii=False, default=str)}\n\n"
        "Com base nisso:\n"
        "1. Responda DIRETAMENTE a pergunta de negocio original, usando os resultados "
        "do modelo -- nao repita a pergunta nem de uma resposta vaga.\n"
        "2. Interprete as metricas de validacao em termos de impacto pratico no "
        "negocio (o que elas significam em decisoes reais, nao so a definicao tecnica).\n"
        "3. Liste acoes concretas e acionaveis que a area de negocio deve tomar com "
        "base neste modelo.\n"
        "4. Aponte riscos, limitacoes e cuidados antes de colocar o modelo em producao.\n"
        "Ancore tudo no contexto de negocio fornecido acima -- evite jargao generico."
    )


# ---------------------------------------------------------------------------
# AutoML (PyCaret)
# ---------------------------------------------------------------------------


def treinar_automl(
    categorizacao_problema: str,
    base_de_dados_nova: pd.DataFrame,
    coluna_target: str,
    log_handler: Optional[logging.Handler] = None,
    fold: int = 10,
    n_modelos_max: Optional[int] = None,
) -> tuple[Dict[str, Any], str]:
    """Treina e compara modelos candidatos via PyCaret, retornando o vencedor.

    `fold` controla o numero de folds da validacao cruzada e `n_modelos_max`
    limita quantos modelos candidatos sao testados no compare_models -- ambos
    servem para acelerar testes de ponta a ponta (menos folds/modelos = bem
    mais rapido, ao custo de uma comparacao menos exaustiva). Deixe
    `n_modelos_max=None` para testar todos os modelos disponiveis.

    Retorna uma tupla (melhor_modelo_treinado, log_pycaret_texto). O log
    detalhado do PyCaret (nivel DEBUG) e escrito em um arquivo temporario
    dedicado a essa execucao e devolvido como texto, para inspecao/download.
    """
    if coluna_target not in base_de_dados_nova.columns:
        raise ValueError(
            f"Coluna target '{coluna_target}' nao encontrada na base de dados "
            f"(colunas disponiveis: {list(base_de_dados_nova.columns)})."
        )

    if "classificacao" in categorizacao_problema:
        from pycaret.classification import compare_models, models, pull, setup
    elif "regressao" in categorizacao_problema:
        from pycaret.regression import compare_models, models, pull, setup
    else:
        raise ValueError(
            f"Categoria de problema '{categorizacao_problema}' nao suportada pelo "
            "AutoML automatico (esperado 'classificacao_binaria', "
            "'classificacao_multiclasse' ou 'regressao')."
        )

    logger.info(
        "Motor do PyCaret selecionado para categoria '%s': %s",
        categorizacao_problema,
        "pycaret.classification" if "classificacao" in categorizacao_problema else "pycaret.regression",
    )
    logger.info(
        "Validando coluna target '%s' -- OK (colunas disponiveis: %s)",
        coluna_target,
        list(base_de_dados_nova.columns),
    )

    log_fd, log_path = tempfile.mkstemp(prefix="agente_modelagem_pycaret_", suffix=".log")
    os.close(log_fd)
    os.environ["PYCARET_CUSTOM_LOGGING_PATH"] = log_path
    os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "DEBUG"

    logger.info(
        "Iniciando setup do PyCaret (target='%s', %d linhas, %d colunas, fold=%d)...",
        coluna_target,
        base_de_dados_nova.shape[0],
        base_de_dados_nova.shape[1],
        fold,
    )
    setup(data=base_de_dados_nova, target=coluna_target, session_id=42, fold=fold, verbose=False)
    logger.info("Setup do PyCaret concluido. Log detalhado (nivel DEBUG) em: %s", log_path)

    # A partir daqui o logger interno do PyCaret ("logs") ja foi criado pelo
    # setup() -- anexamos nosso handler para que o log DEBUG do PyCaret tambem
    # apareca ao vivo na interface, alem de ir para o arquivo acima.
    pycaret_logger = logging.getLogger("logs")
    if log_handler is not None:
        pycaret_logger.addHandler(log_handler)

    compare_kwargs: Dict[str, Any] = {"verbose": False}
    if n_modelos_max is not None:
        catalogo = models()
        # prioriza os modelos "turbo" (rapidos) na hora de montar a lista reduzida
        ids_ordenados = (
            catalogo[catalogo["Turbo"]].index.tolist()
            + catalogo[~catalogo["Turbo"]].index.tolist()
        )
        ids_selecionados = ids_ordenados[:n_modelos_max]
        compare_kwargs["include"] = ids_selecionados
        logger.info(
            "Modo rapido ativado: limitando compare_models a %d modelo(s): %s",
            n_modelos_max,
            ids_selecionados,
        )

    try:
        logger.info(
            "Iniciando compare_models (fold=%d, %s) -- isso pode levar alguns minutos...",
            fold,
            f"limitado a {n_modelos_max} modelo(s)" if n_modelos_max else "todos os modelos disponiveis",
        )
        melhor_modelo = compare_models(**compare_kwargs)
        leaderboard = pull()
        logger.info("compare_models concluido -- %d modelo(s) avaliado(s):", len(leaderboard))
        for _, linha in leaderboard.iterrows():
            metricas_str = ", ".join(
                f"{coluna}={linha[coluna]:.4f}"
                for coluna in leaderboard.columns
                if coluna != "Model" and isinstance(linha[coluna], (int, float))
            )
            logger.info("  - %s: %s", linha["Model"], metricas_str)
    finally:
        if log_handler is not None:
            pycaret_logger.removeHandler(log_handler)

    metricas_vencedor = leaderboard.iloc[0].drop(labels=["Model"], errors="ignore")
    nome_vencedor = str(leaderboard.iloc[0]["Model"])
    logger.info("Modelo vencedor selecionado: %s", nome_vencedor)

    melhor_modelo_treinado = {
        "nome_modelo": nome_vencedor,
        "hiperparametros": melhor_modelo.get_params(),
        "metricas_validacao": {k: float(v) for k, v in metricas_vencedor.items()},
        "n_linhas_treino": len(base_de_dados_nova),
    }

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as arquivo_log:
            log_pycaret_texto = arquivo_log.read()
    except OSError:
        log_pycaret_texto = ""

    return melhor_modelo_treinado, log_pycaret_texto


# ---------------------------------------------------------------------------
# Nos do grafo
# ---------------------------------------------------------------------------


def entendimento_contexto(state: AgenteState) -> Dict[str, Any]:
    """No 1: entende a base, a pergunta de negocio e os metadados."""
    base_de_dados = state["base_de_dados"]
    pergunta_negocio = state["pergunta_negocio"]
    metadados = state["metadados"]

    logger.info("Chamando LLM para entendimento de contexto...")
    prompt = montar_prompt_entendimento(base_de_dados, pergunta_negocio, metadados)
    resultado = get_llm().with_structured_output(EntendimentoContextoOutput).invoke(prompt)

    logger.info("categorizacao='%s'", resultado.categorizacao_problema)

    return {
        "categorizacao_problema": resultado.categorizacao_problema,
        "sugestao_limpeza": resultado.sugestao_limpeza,
    }


def engenharia_features(state: AgenteState) -> Dict[str, Any]:
    """No 2: cria/remove features adaptadas a categoria do problema."""
    categorizacao_problema = state["categorizacao_problema"]
    base_de_dados = state["base_de_dados"]
    metadados = state["metadados"]

    logger.info("Chamando LLM para engenharia de features...")
    prompt = montar_prompt_features(categorizacao_problema, base_de_dados, metadados)
    resultado = get_llm().with_structured_output(EngenhariaFeaturesOutput).invoke(prompt)

    base_de_dados_nova = base_de_dados.copy()
    for nova_feature in resultado.features_criadas:
        if nova_feature not in base_de_dados_nova.columns:
            base_de_dados_nova[nova_feature] = 0
    for feature_removida in resultado.features_removidas:
        if feature_removida in base_de_dados_nova.columns:
            base_de_dados_nova = base_de_dados_nova.drop(columns=[feature_removida])

    logger.info(
        "features criadas=%s, features removidas=%s",
        resultado.features_criadas,
        resultado.features_removidas,
    )

    return {
        "base_de_dados_nova": base_de_dados_nova,
        "features_criadas": resultado.features_criadas,
        "features_removidas": resultado.features_removidas,
    }


def modelagem(state: AgenteState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """No 3: sugere modelos/metricas (LLM) e executa o treinamento real (AutoML/PyCaret)."""
    categorizacao_problema = state["categorizacao_problema"]
    base_de_dados_nova = state["base_de_dados_nova"]
    metadados = state["metadados"]
    coluna_target = metadados.get("coluna_target", "target")
    configurable = (config or {}).get("configurable") or {}
    log_handler = configurable.get("log_handler")
    fold = configurable.get("pycaret_fold", metadados.get("pycaret_fold", 10))
    n_modelos_max = configurable.get("pycaret_max_modelos", metadados.get("pycaret_max_modelos"))

    logger.info("Chamando LLM para sugestao de modelos/metricas...")
    prompt = montar_prompt_modelagem(categorizacao_problema, base_de_dados_nova)
    sugestao = get_llm().with_structured_output(ModelagemSugestaoOutput).invoke(prompt)

    melhor_modelo_treinado, pycaret_log_texto = treinar_automl(
        categorizacao_problema,
        base_de_dados_nova,
        coluna_target,
        log_handler=log_handler,
        fold=fold,
        n_modelos_max=n_modelos_max,
    )

    resultado = ModelagemOutput(
        modelos_sugeridos=sugestao.modelos_sugeridos,
        metricas_avaliacao=sugestao.metricas_avaliacao,
        melhor_modelo_treinado=melhor_modelo_treinado,
    )

    logger.info("melhor_modelo='%s'", resultado.melhor_modelo_treinado["nome_modelo"])

    return {
        "modelos_sugeridos": resultado.modelos_sugeridos,
        "metricas_avaliacao": resultado.metricas_avaliacao,
        "melhor_modelo_treinado": resultado.melhor_modelo_treinado,
        "pycaret_log_texto": pycaret_log_texto,
    }


def avaliacao(state: AgenteState) -> Dict[str, Any]:
    """No 4: avalia o melhor modelo e gera o relatorio para negocio."""
    melhor_modelo_treinado = state["melhor_modelo_treinado"]
    metricas_validacao = melhor_modelo_treinado["metricas_validacao"]

    logger.info("Chamando LLM para relatorio de avaliacao...")
    prompt = montar_prompt_avaliacao(
        pergunta_negocio=state["pergunta_negocio"],
        categorizacao_problema=state["categorizacao_problema"],
        metadados=state["metadados"],
        features_criadas=state["features_criadas"],
        features_removidas=state["features_removidas"],
        melhor_modelo_treinado=melhor_modelo_treinado,
    )
    sugestao = get_llm().with_structured_output(AvaliacaoRelatorioOutput).invoke(prompt)

    resultado = AvaliacaoOutput(
        metricas_finais=metricas_validacao,
        resposta_pergunta_negocio=sugestao.resposta_pergunta_negocio,
        interpretacao_metricas=sugestao.interpretacao_metricas,
        acoes_recomendadas=sugestao.acoes_recomendadas,
        riscos_e_limitacoes=sugestao.riscos_e_limitacoes,
    )

    logger.info("metricas_finais=%s", resultado.metricas_finais)

    return {
        "metricas_finais": resultado.metricas_finais,
        "resposta_pergunta_negocio": resultado.resposta_pergunta_negocio,
        "interpretacao_metricas": resultado.interpretacao_metricas,
        "acoes_recomendadas": resultado.acoes_recomendadas,
        "riscos_e_limitacoes": resultado.riscos_e_limitacoes,
    }


# ---------------------------------------------------------------------------
# Construcao do grafo
# ---------------------------------------------------------------------------


def construir_grafo():
    grafo = StateGraph(AgenteState)

    grafo.add_node("entendimento_contexto", entendimento_contexto)
    grafo.add_node("engenharia_features", engenharia_features)
    grafo.add_node("modelagem", modelagem)
    grafo.add_node("avaliacao", avaliacao)

    grafo.add_edge(START, "entendimento_contexto")
    grafo.add_edge("entendimento_contexto", "engenharia_features")
    grafo.add_edge("engenharia_features", "modelagem")
    grafo.add_edge("modelagem", "avaliacao")
    grafo.add_edge("avaliacao", END)

    return grafo.compile()


agente_modelagem = construir_grafo()


if __name__ == "__main__":
    print(
        "Este modulo agora faz chamadas reais de LLM e treina modelos reais via "
        "PyCaret -- use a interface web para rodar o pipeline com seu proprio "
        "CSV e YAML de metadados:\n\n"
        "    streamlit run app.py\n"
    )
