"""
CBR Prediction Dashboard — Streamlit
Dissertação de Mestrado | PUC Goiás 2025

Run: streamlit run dashboard.py
"""

import os
import sys
import subprocess

# O Streamlit executa este script fora da thread principal, e o backend
# interativo do matplotlib cria objetos Tk — que só podem nascer e morrer na
# thread do laço de eventos. Fora dela, cada figura coletada derruba
# `RuntimeError: main thread is not in main loop`. Aqui as figuras vão para o
# navegador via `st.pyplot`, então nenhuma janela é necessária. Precisa vir
# antes de qualquer importação de pyplot, inclusive a indireta via `core`.
os.environ.setdefault("MLL_SHOW", "0")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────
# CAMINHOS E CONSTANTES
# ─────────────────────────────────────────────

CODE_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from core import metrics, paths, plots  # noqa: E402

# Mesma paleta, mesma grade e mesma ausência de títulos dos gráficos gerados
# pelos scripts de treino — o dashboard e a dissertação mostram a mesma coisa.
plots.apply_style()

DATA_PATH = str(paths.DATASET)
RF_PT_DIR = str(paths.RF_DIR)
RF_EN_DIR = str(paths.RF_EN_DIR)
MLP_DIR   = str(paths.MLP_DIR)
QNT_DIR   = str(paths.RF_QUINTIS_DIR)
RF_ENS_DIR = str(paths.RF_ENSEMBLE_DIR)
SUBSETS_DIR = str(paths.SUBSETS_DIR)

FEATURES = [
    "25.4mm", "9.5mm", "4.8mm", "2.0mm", "0.42mm", "0.076mm",
    "LL", "IP", "Umidade Ótima", "Densidade máxima",
]
FEATURES_LABELS = [
    "25.4mm (IG)", "9.5mm (EXP)", "4.8mm (D3)", "2.0mm (D4)",
    "0.42mm (D5)", "0.076mm (D6)", "LL", "IP",
    "Umidade Ótima (CH)", "Densidade máxima (CY)",
]
TARGET_COL = "CBR "
SEED       = 42
TEST_SIZE  = 0.20
VAL_SIZE   = 0.15
LOG_TARGET = True
# Meta declarada em R²: não depende da escala nem da faixa do alvo, e por isso
# vale igual para o modelo global e para um quintil isolado (ver
# core/metrics.py). O limiar de MSE equivalente muda com a variância de cada
# conjunto, então não cabe como constante aqui.
META_R2    = metrics.META_R2

LIMITS = {
    "25.4mm":          (0.0,    100.0),
    "9.5mm":           (0.0,    100.0),
    "4.8mm":           (0.0,    100.0),
    "2.0mm":           (0.0,    100.0),
    "0.42mm":          (0.0,    100.0),
    "0.076mm":         (0.0,    100.0),
    "LL":              (0.0,    100.0),
    "IP":              (0.0,     80.0),
    "Umidade Ótima":   (5.0,     40.0),
    "Densidade máxima":(1200.0, 2200.0),
}

SCRIPTS = {
    "RF Padrão (PT)":      "RANDOM_FOREST.py",
    "RF Standard (EN)":    "RANDOM_FOREST_EN.py",
    "RF Quintis Simples":  "RANDOM_FLOREST_Dn.py",
    "RF Quintis Completo": "RANDOM_FLOREST_Separado.py",
    "MLP — Rede Neural":   "MLL.py",
}

CORES_QUINTIL = {
    "D1": "#1D4ED8", "D2": "#0891B2", "D3": "#16A34A",
    "D4": "#D97706", "D5": "#DC2626",
}

PALETTE = {
    "blue":   "#2563EB", "blue2":  "#60A5FA",
    "orange": "#EA580C", "green":  "#16A34A",
    "red":    "#DC2626", "bg":     "#F8FAFC",
    "grid":   "#E2E8F0",
}

# ─────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="CBR ML Dashboard",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stButton > button { width: 100%; }
    h2 { border-bottom: 2px solid #E2E8F0; padding-bottom: 6px; }
    .metric-label { font-size: 0.75rem; color: #64748B; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("🪨 CBR Prediction")
    st.caption("Dissertação de Mestrado · PUC Goiás · 2025")
    st.divider()
    pagina = st.radio(
        "Navegação",
        ["📊 Visão Geral", "🌳 Resultados RF", "🔢 Quintis D1–D5",
         "🔮 Previsão Manual", "⚙️ Treinamento"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Meta: R² >= {META_R2:.2f}")

# ─────────────────────────────────────────────
# FUNÇÕES COMPARTILHADAS
# ─────────────────────────────────────────────

@st.cache_data
def carregar_dados():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    _map = {
        "Ll":"LL","ll":"LL","L.L":"LL",
        "Ip":"IP","ip":"IP","I.P":"IP",
        "Densidade Maxima":"Densidade máxima","Densidade Máxima":"Densidade máxima",
        "d_max":"Densidade máxima","Dmax":"Densidade máxima",
        "Wot":"Umidade Ótima","wot":"Umidade Ótima",
        "Umidade otima":"Umidade Ótima","Umidade Otima":"Umidade Ótima",
        "CBR":"CBR ","cbr":"CBR ","Cbr":"CBR ",
    }
    return df.rename(columns=_map)[FEATURES + [TARGET_COL]]


def dividir_dados(df):
    """Reproduz o mesmo split train/val/test dos scripts de treinamento."""
    Y = df[TARGET_COL].values.ravel()
    X = df[FEATURES].values
    Y_orig = Y.copy()
    if LOG_TARGET:
        Y = np.log1p(Y)
    X_tv, X_test, Y_tv, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=SEED)
    X_train, X_val, Y_train, Y_val = train_test_split(X_tv, Y_tv, test_size=VAL_SIZE, random_state=SEED)
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "Y_train": Y_train, "Y_val":  Y_val, "y_test": y_test,
        "Y_orig":  Y_orig,
    }


def calcular_metricas(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100
    r2   = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R²": r2}


def exibir_metricas(met_val, met_test):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Validação**")
        cols = st.columns(5)
        for col, (k, v) in zip(cols, met_val.items()):
            fmt = f"{v:.2f}%" if k == "MAPE (%)" else f"{v:.4f}"
            col.metric(k, fmt)
    with c2:
        st.markdown("**Teste Final**")
        cols = st.columns(5)
        for col, (k, v) in zip(cols, met_test.items()):
            fmt = f"{v:.2f}%" if k == "MAPE (%)" else f"{v:.4f}"
            if k == "R²":
                ok = v >= META_R2
                col.metric(k, fmt, delta="✓ Meta" if ok else "✗ Abaixo",
                           delta_color="normal" if ok else "inverse")
            else:
                col.metric(k, fmt)


def plot_scatter(y_true, y_pred, titulo, cor):
    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    lim = [min(y_true.min(), y_pred.min()) * 0.92,
           max(y_true.max(), y_pred.max()) * 1.08]
    ax.scatter(y_true, y_pred, alpha=0.55, color=cor,
               edgecolors="white", linewidths=0.4, s=40)
    ax.plot(lim, lim, "--", color=PALETTE["orange"], lw=1.5, label="Ideal")
    ax.text(0.05, 0.93, f"R² = {r2_score(y_true, y_pred):.4f}",
            transform=ax.transAxes, fontsize=10, color=cor, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#E2E8F0", alpha=0.93))
    ax.set_xlim(lim); ax.set_ylim(lim)

    ax.set_xlabel("Real (CBR %)"); ax.set_ylabel("Previsto (CBR %)")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_residuos(y_true, y_pred, titulo, cor):
    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.scatter(y_pred, y_true - y_pred, alpha=0.55, color=cor,
               edgecolors="white", linewidths=0.4, s=40)
    ax.axhline(0, color=PALETTE["orange"], lw=1.5, linestyle="--")

    ax.set_xlabel("Previsto (CBR %)"); ax.set_ylabel("Resíduo")
    plt.tight_layout()
    return fig


def plot_importancia(importancias, titulo):
    idx   = np.argsort(importancias)[::-1]
    cores = plt.cm.Blues(np.linspace(0.4, 0.9, len(importancias)))[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.barh(range(len(importancias)), importancias[idx], color=cores, edgecolor="white")
    ax.set_yticks(range(len(importancias)))
    ax.set_yticklabels([FEATURES_LABELS[i] for i in idx], fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Importância (Gini)")
    plt.tight_layout()
    return fig


def carregar_rf(model_dir, rf_filename):
    """Carrega RF + scaler e retorna predições no val/test."""
    model_path  = os.path.join(model_dir, rf_filename)
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None
    model  = load(model_path)
    scaler = load(scaler_path)
    df     = carregar_dados()
    s      = dividir_dados(df)
    pred_val  = model.predict(scaler.transform(s["X_val"]))
    pred_test = model.predict(scaler.transform(s["X_test"]))
    if LOG_TARGET:
        pred_val  = np.expm1(pred_val)
        pred_test = np.expm1(pred_test)
        Y_val_met  = np.expm1(s["Y_val"])
        y_test_met = np.expm1(s["y_test"])
    else:
        Y_val_met, y_test_met = s["Y_val"], s["y_test"]
    return {
        "model":     model,
        "Y_val":     Y_val_met, "pred_val":  pred_val,
        "y_test":    y_test_met, "pred_test": pred_test,
    }


# ─────────────────────────────────────────────
# PÁGINA: VISÃO GERAL
# ─────────────────────────────────────────────

if pagina == "📊 Visão Geral":
    st.title("📊 Visão Geral do Projeto")
    st.markdown("""
    **Previsão de CBR** (California Bearing Ratio) de solos com Machine Learning.
    *Dissertação de Mestrado · PUC Goiás · Engenharia Industrial e IA · 2025*
    """)

    try:
        df = carregar_dados()
        cbr = df[TARGET_COL]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Amostras",       df.shape[0])
        col2.metric("Features",       len(FEATURES))
        col3.metric("CBR Mínimo",     f"{cbr.min():.1f}%")
        col4.metric("CBR Máximo",     f"{cbr.max():.1f}%")
        col5.metric("CBR Médio",      f"{cbr.mean():.1f}%")

        st.subheader("Distribuição do CBR")
        col_hist, col_stats = st.columns([2, 1])
        with col_hist:
            fig, ax = plt.subplots(figsize=(8, 4), facecolor=PALETTE["bg"])
            ax.set_facecolor(PALETTE["bg"])
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.hist(cbr, bins=25, color=PALETTE["blue"], alpha=0.75, edgecolor="white")
            ax.axvline(cbr.mean(), color=PALETTE["orange"], lw=2, linestyle="--",
                       label=f"Média = {cbr.mean():.1f}%")
            ax.axvline(cbr.median(), color=PALETTE["green"], lw=1.5, linestyle=":",
                       label=f"Mediana = {cbr.median():.1f}%")
            ax.set_xlabel("CBR (%)"); ax.set_ylabel("Frequência")

            ax.legend()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_stats:
            st.markdown("**Estatísticas Descritivas**")
            desc = cbr.describe().rename({
                "count": "N", "mean": "Média", "std": "Std",
                "min": "Mín", "25%": "Q1", "50%": "Mediana",
                "75%": "Q3", "max": "Máx",
            })
            st.dataframe(desc.to_frame().round(2), use_container_width=True)

        st.subheader("Correlação das Features com CBR")
        corr = df.corr()[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
        fig2, ax2 = plt.subplots(figsize=(8, 3.5), facecolor=PALETTE["bg"])
        ax2.set_facecolor(PALETTE["bg"])
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
        cores_corr = [PALETTE["blue"] if v > 0 else PALETTE["red"] for v in corr.values]
        labels_ord = [FEATURES_LABELS[FEATURES.index(f)] for f in corr.index]
        ax2.barh(labels_ord, corr.values, color=cores_corr, edgecolor="white", alpha=0.85)
        ax2.axvline(0, color="#374151", lw=0.8)

        ax2.set_xlabel("Coeficiente de Correlação")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        st.subheader("Features de Entrada")
        feat_df = pd.DataFrame({
            "Coluna CSV":  FEATURES,
            "Rótulo":      FEATURES_LABELS,
            "Unidade":     ["%"]*8 + ["%", "kg/m³"],
            "Intervalo":   [f"{LIMITS[f][0]} – {LIMITS[f][1]}" for f in FEATURES],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.error(f"Dataset não encontrado: `{DATA_PATH}`")
        st.info("Execute `LEITURA.py` para gerar o arquivo de dados processados.")


# ─────────────────────────────────────────────
# PÁGINA: RESULTADOS RF
# ─────────────────────────────────────────────

elif pagina == "🌳 Resultados RF":
    st.title("🌳 Random Forest — Resultados")

    aba_pt, aba_en, aba_comp = st.tabs(["🇧🇷 Português", "🇬🇧 English", "⚖️ Comparação"])

    def pagina_rf(model_dir, rf_filename, lang):
        resultado = carregar_rf(model_dir, rf_filename)
        if resultado is None:
            st.warning(f"Modelo não encontrado em `{model_dir}`.")
            st.info("Vá para **⚙️ Treinamento** e execute o modelo.")
            return None

        model    = resultado["model"]
        Y_val    = resultado["Y_val"];   pred_val  = resultado["pred_val"]
        y_test   = resultado["y_test"];  pred_test = resultado["pred_test"]
        met_val  = calcular_metricas(Y_val, pred_val)
        met_test = calcular_metricas(y_test, pred_test)

        st.subheader(f"Métricas — RF ({lang})")
        exibir_metricas(met_val, met_test)

        st.subheader("Gráficos")
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = plot_scatter(Y_val, pred_val, "Previsto vs Real — Validação", PALETTE["blue"])
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        with c2:
            fig = plot_scatter(y_test, pred_test, "Previsto vs Real — Teste", PALETTE["blue2"])
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        with c3:
            fig = plot_importancia(model.feature_importances_, "Importância das Features")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        c4, c5 = st.columns(2)
        with c4:
            fig = plot_residuos(Y_val, pred_val, "Resíduos — Validação", PALETTE["blue"])
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        with c5:
            fig = plot_residuos(y_test, pred_test, "Resíduos — Teste", PALETTE["blue2"])
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        with st.expander("Parâmetros do modelo"):
            st.json(model.get_params())

        return met_test

    with aba_pt:
        met_pt = pagina_rf(RF_PT_DIR, "rf_model.joblib", "Português")
    with aba_en:
        met_en = pagina_rf(RF_EN_DIR, "rf_model.joblib", "English")
    with aba_comp:
        st.subheader("Comparação PT vs EN")
        if met_pt is None or met_en is None:
            st.info("Treine ambos os modelos (PT e EN) para ver a comparação.")
        else:
            metricas_list = list(met_pt.keys())
            valores_pt    = [met_pt[k] for k in metricas_list]
            valores_en    = [met_en[k] for k in metricas_list]

            fig, ax = plt.subplots(figsize=(8, 4), facecolor=PALETTE["bg"])
            ax.set_facecolor(PALETTE["bg"])
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            x = np.arange(len(metricas_list))
            w = 0.35
            bars1 = ax.bar(x - w/2, valores_pt, w, label="RF PT", color=PALETTE["blue"], edgecolor="white")
            bars2 = ax.bar(x + w/2, valores_en, w, label="RF EN", color=PALETTE["blue2"], edgecolor="white")
            ax.set_xticks(x); ax.set_xticklabels(metricas_list)

            ax.legend()
            for bar in list(bars1) + list(bars2):
                v = bar.get_height()
                fmt = f"{v:.2f}%" if v > 10 else f"{v:.4f}"
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                        fmt, ha="center", va="bottom", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

            comp_df = pd.DataFrame({
                "Métrica": metricas_list,
                "RF PT":   [f"{v:.4f}" for v in valores_pt],
                "RF EN":   [f"{v:.4f}" for v in valores_en],
                "Δ (EN−PT)": [f"{e-p:+.4f}" for p, e in zip(valores_pt, valores_en)],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PÁGINA: QUINTIS D1–D5
# ─────────────────────────────────────────────

elif pagina == "🔢 Quintis D1–D5":
    st.title("🔢 Random Forest por Faixas de CBR (D1–D5)")

    try:
        df      = carregar_dados()
        Y_orig  = df[TARGET_COL].values.ravel()
        limites = np.percentile(Y_orig, [0, 20, 40, 60, 80, 100])
        limites[0]  = Y_orig.min() - 1e-6
        limites[-1] = Y_orig.max() + 1e-6
        rotulos = ["D1", "D2", "D3", "D4", "D5"]

        resultados = []
        for i, rot in enumerate(rotulos):
            pasta = os.path.join(QNT_DIR, rot)
            m_path = os.path.join(pasta, "rf_model.joblib")
            s_path = os.path.join(pasta, "scaler.joblib")

            mask = (Y_orig > limites[i]) & (Y_orig <= limites[i+1])
            X_g = df[FEATURES].values[mask]
            Y_g = Y_orig[mask]

            if not os.path.exists(m_path):
                resultados.append({"Grupo": rot, "Faixa CBR": f"{Y_g.min():.1f}–{Y_g.max():.1f}",
                                   "N": int(mask.sum()), "Status": "⚠️ Não treinado",
                                   "MSE": None, "R²": None})
                continue

            model  = load(m_path)
            scaler = load(s_path)
            Y_g_proc = np.log1p(Y_g) if LOG_TARGET else Y_g

            X_tv, X_test, Y_tv, y_test, Y_orig_tv, Y_orig_test = train_test_split(
                X_g, Y_g_proc, Y_g, test_size=0.25, random_state=SEED
            )
            pred = model.predict(scaler.transform(X_test))
            if LOG_TARGET:
                pred = np.expm1(pred)
            m = calcular_metricas(Y_orig_test, pred)
            resultados.append({
                "Grupo": rot, "Faixa CBR": f"{Y_g.min():.1f}–{Y_g.max():.1f}",
                "N": int(mask.sum()), "Status": "✅",
                "MSE": m["MSE"], "RMSE": m["RMSE"],
                "MAE": m["MAE"], "R²": m["R²"],
                "_y_true": Y_orig_test, "_pred": pred,
            })

        # Tabela resumo
        st.subheader("Resumo por Grupo")
        tab_dados = [{k: (f"{v:.4f}" if isinstance(v, float) else v)
                      for k, v in r.items() if not k.startswith("_")}
                     for r in resultados]
        st.dataframe(pd.DataFrame(tab_dados), use_container_width=True, hide_index=True)

        completos = [r for r in resultados if isinstance(r.get("MSE"), float)]
        if completos:
            # Barras MSE e R²
            col1, col2 = st.columns(2)
            rots  = [r["Grupo"] for r in completos]
            cores = [CORES_QUINTIL[r] for r in rots]

            with col1:
                fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PALETTE["bg"])
                ax.set_facecolor(PALETTE["bg"])
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                bars = ax.bar(rots, [r["MSE"] for r in completos], color=cores, edgecolor="white", width=0.6)
                # Sem linha única de meta: o limiar de MSE equivalente a
                # R² >= META_R2 muda de quintil para quintil, junto com a
                # variância do CBR dentro de cada grupo.
                for b, v in zip(bars, [r["MSE"] for r in completos]):
                    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

                ax.set_ylabel("MSE"); ax.legend()
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PALETTE["bg"])
                ax.set_facecolor(PALETTE["bg"])
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                bars2 = ax.bar(rots, [r["R²"] for r in completos], color=cores, edgecolor="white", width=0.6)
                ax.axhline(0.8, color=PALETTE["green"], lw=1.2, linestyle="--", label="R²=0.8")
                ax.axhline(0.0, color=PALETTE["red"], lw=0.8, linestyle=":", alpha=0.5)
                for b, v in zip(bars2, [r["R²"] for r in completos]):
                    ax.text(b.get_x()+b.get_width()/2, max(v, 0)+0.01,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

                ax.set_ylabel("R²"); ax.legend()
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close(fig)

            # Scatter sobrepostos
            st.subheader("Previsto vs Real — Todos os Grupos")
            fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=PALETTE["bg"])
            ax.set_facecolor(PALETTE["bg"])
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            todos_real = np.concatenate([r["_y_true"] for r in completos])
            todos_pred = np.concatenate([r["_pred"]   for r in completos])
            lim = [todos_real.min() * 0.9, todos_real.max() * 1.1]
            for r in completos:
                ax.scatter(r["_y_true"], r["_pred"], alpha=0.6,
                           color=CORES_QUINTIL[r["Grupo"]], edgecolors="white",
                           linewidths=0.3, s=35,
                           label=f"{r['Grupo']} — {r['Faixa CBR']}%")
            ax.plot(lim, lim, "--", color=PALETTE["orange"], lw=1.5, label="Ideal")
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("Real (CBR %)"); ax.set_ylabel("Previsto (CBR %)")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        else:
            st.info("Nenhum modelo de quintil encontrado. Execute o treinamento em **⚙️ Treinamento**.")

    except FileNotFoundError:
        st.error(f"Dataset não encontrado: `{DATA_PATH}`")


# ─────────────────────────────────────────────
# PÁGINA: PREVISÃO MANUAL
# ─────────────────────────────────────────────

elif pagina == "🔮 Previsão Manual":
    st.title("🔮 Previsão Manual de CBR")
    st.markdown("Informe os parâmetros granulométricos e geotécnicos do solo.")

    col_form, col_result = st.columns([3, 2])

    with col_form:
        st.subheader("Parâmetros do Solo")
        valores = {}
        col_a, col_b = st.columns(2)
        metade = len(FEATURES) // 2
        for col, feats in zip([col_a, col_b], [FEATURES[:metade], FEATURES[metade:]]):
            with col:
                for f in feats:
                    lo, hi = LIMITS[f]
                    label  = FEATURES_LABELS[FEATURES.index(f)]
                    default = round((lo + hi) / 2, 1) if f != "Densidade máxima" else 1700.0
                    step    = 10.0 if f == "Densidade máxima" else 0.5
                    valores[f] = st.number_input(
                        label,
                        min_value=float(lo), max_value=float(hi),
                        value=float(default), step=step,
                    )
        prever = st.button("🔮 Calcular CBR Previsto", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Resultado")
        if prever:
            X_inp = np.array([[valores[f] for f in FEATURES]])
            previsoes = {}

            for nome, (mdir, mfile) in {
                "RF (Português)": (RF_PT_DIR, "rf_model.joblib"),
                "RF (English)":   (RF_EN_DIR, "rf_model.joblib"),
            }.items():
                mp = os.path.join(mdir, mfile)
                sp = os.path.join(mdir, "scaler.joblib")
                if os.path.exists(mp) and os.path.exists(sp):
                    m  = load(mp); sc = load(sp)
                    p  = float(m.predict(sc.transform(X_inp))[0])
                    if LOG_TARGET:
                        p = float(np.expm1(p))
                    previsoes[nome] = p

            if previsoes:
                for nome, cbr in previsoes.items():
                    cor = "normal"
                    if cbr < 2 or cbr > 100:
                        cor = "inverse"
                    st.metric(f"CBR — {nome}", f"{cbr:.2f} %", delta_color=cor)
                    if cbr < 2:
                        st.warning("CBR muito baixo — verifique os dados inseridos.")
                    elif cbr > 100:
                        st.warning("CBR acima de 100% — valor atípico.")

                if len(previsoes) == 2:
                    vals = list(previsoes.values())
                    st.info(f"Diferença entre modelos: **{abs(vals[0]-vals[1]):.2f}%**")
                    media = np.mean(vals)
                    st.success(f"Média dos modelos: **{media:.2f}%**")
            else:
                st.warning("Nenhum modelo RF encontrado.")
                st.info("Execute o treinamento em **⚙️ Treinamento**.")
        else:
            st.info("Preencha os parâmetros e clique em **Calcular CBR Previsto**.")
            st.markdown("""
**Referência — Classificação CBR (DNIT):**

| CBR (%)  | Classificação |
|----------|---------------|
| < 2      | Muito fraco   |
| 2 – 7    | Fraco         |
| 7 – 20   | Regular       |
| 20 – 50  | Bom           |
| > 50     | Excelente     |
            """)


# ─────────────────────────────────────────────
# PÁGINA: TREINAMENTO
# ─────────────────────────────────────────────

elif pagina == "⚙️ Treinamento":
    st.title("⚙️ Treinamento de Modelos")
    st.info("Execute os scripts de treinamento. Os modelos são salvos automaticamente na pasta `models/`.")

    col_conf, col_saida = st.columns([1, 2])

    with col_conf:
        st.subheader("Configuração")
        modelo_sel = st.selectbox("Modelo", list(SCRIPTS.keys()))
        modo = st.radio("Modo de execução", [
            "Separado — modelo escolhido",
            "Juntos — apenas RF (PT + EN)",
            "Juntos — todos os modelos",
        ])
        st.divider()
        st.markdown("**Status dos modelos:**")
        checks = {
            "RF PT":          os.path.join(RF_PT_DIR, "rf_model.joblib"),
            "RF EN":          os.path.join(RF_EN_DIR, "rf_model.joblib"),
            "RF Quintis D1":  os.path.join(QNT_DIR, "D1", "rf_model.joblib"),
            "MLP":            os.path.join(MLP_DIR, "mlp_model.keras"),
        }
        for nome, caminho in checks.items():
            icon = "✅" if os.path.exists(caminho) else "⬜"
            st.markdown(f"{icon} {nome}")

        executar = st.button("▶ Executar", type="primary", use_container_width=True)

    with col_saida:
        st.subheader("Saída do Terminal")
        area = st.empty()

        if executar:
            if modo == "Separado — modelo escolhido":
                scripts_rodar = [SCRIPTS[modelo_sel]]
            elif modo == "Juntos — apenas RF (PT + EN)":
                scripts_rodar = [SCRIPTS["RF Padrão (PT)"], SCRIPTS["RF Standard (EN)"]]
            else:
                scripts_rodar = list(SCRIPTS.values())

            saida_total = ""
            for script in scripts_rodar:
                caminho = os.path.join(CODE_DIR, script)
                saida_total += f"\n{'='*50}\n  Executando: {script}\n{'='*50}\n"
                area.code(saida_total + "⏳ Processando...", language="text")

                try:
                    resultado = subprocess.run(
                        [sys.executable, caminho],
                        capture_output=True, text=True,
                        cwd=CODE_DIR, timeout=900,
                        encoding="utf-8", errors="replace",
                    )
                    saida_total += resultado.stdout or ""
                    if resultado.returncode != 0:
                        saida_total += f"\n[STDERR]\n{resultado.stderr}\n"
                        saida_total += "❌ Encerrado com erro.\n"
                    else:
                        saida_total += "\n✅ Concluído com sucesso.\n"
                except subprocess.TimeoutExpired:
                    saida_total += "\n⏱️ Tempo limite excedido (15 min).\n"

                area.code(saida_total, language="text")

            st.success("Execução finalizada. Os gráficos foram exibidos no terminal.")
            st.cache_data.clear()
            st.info("Navegue para **🌳 Resultados RF** para ver as métricas e gráficos interativos.")
