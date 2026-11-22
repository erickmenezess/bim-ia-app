"""Visualizador de dados BIM (IFC) com assistente de IA.
Atividade 06 - BIM e IA - Especializacao BIM UFPE.
"""
import io, json, tempfile, datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as uel

st.set_page_config(page_title="BIM + IA | Visualizador IFC", layout="wide")

CLASSES_3D = ["IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcDoor", "IfcWindow", "IfcStair", "IfcRoof", "IfcColumn", "IfcBeam"]
CORES = {"IfcWall": "#b8b8b8", "IfcWallStandardCase": "#b8b8b8", "IfcSlab": "#8c8c8c", "IfcDoor": "#a0522d",
         "IfcWindow": "#6fa8dc", "IfcStair": "#999966", "IfcRoof": "#a64d4d", "IfcColumn": "#777777", "IfcBeam": "#777777"}


@st.cache_data(show_spinner=False)
def ler_ifc(dados: bytes, nome: str):
    with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as f:
        f.write(dados)
        caminho = f.name
    m = ifcopenshell.open(caminho)
    linhas = []
    for e in m.by_type("IfcProduct"):
        if e.is_a("IfcOpeningElement") or e.is_a("IfcSite") or e.is_a("IfcBuilding") or e.is_a("IfcBuildingStorey") or e.is_a("IfcGrid"):
            continue
        cont = uel.get_container(e) or uel.get_aggregate(e)
        tipo = uel.get_type(e)
        psets = uel.get_psets(e)
        q = {}
        for nome_ps, props in psets.items():
            if nome_ps.startswith("Qto_") or nome_ps == "BaseQuantities":
                q.update({k: v for k, v in props.items() if isinstance(v, (int, float))})
        n_props = sum(len([v for k, v in p.items() if k != "id" and v not in (None, "")]) for p in psets.values())
        linhas.append({
            "modelo": nome,
            "GlobalId": e.GlobalId,
            "classe": e.is_a(),
            "nome": e.Name or "",
            "tipo": (e.LongName if e.is_a("IfcSpace") else (tipo.Name if tipo else (e.ObjectType or ""))) or "",
            "pavimento": cont.Name if cont and cont.is_a("IfcBuildingStorey") else "(sem pavimento)",
            "n_psets": len(psets),
            "n_props_preenchidas": n_props,
            "area_m2": q.get("NetSideArea") or q.get("NetArea") or q.get("GrossArea") or q.get("NetFloorArea") or q.get("Area"),
            "volume_m3": q.get("NetVolume") or q.get("GrossVolume"),
        })
    df = pd.DataFrame(linhas)
    malhas = []
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    for cls in CLASSES_3D:
        for e in m.by_type(cls):
            try:
                sh = ifcopenshell.geom.create_shape(s, e)
                v = np.array(sh.geometry.verts).reshape(-1, 3)
                fcs = np.array(sh.geometry.faces).reshape(-1, 3)
                malhas.append({"classe": e.is_a(), "modelo": nome, "v": v, "f": fcs})
            except Exception:
                pass
    pj = m.by_type("IfcProject")
    projeto = (pj[0].LongName or pj[0].Name) if pj else nome
    return df, malhas, m.schema, projeto


def figura_3d(malhas):
    fig = go.Figure()
    for k in malhas:
        v, f = k["v"], k["f"]
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color=CORES.get(k["classe"], "#cccccc"), opacity=0.55 if k["classe"] == "IfcWindow" else 1.0,
                                flatshading=True, name=k["classe"], showscale=False, hoverinfo="name"))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                                 camera=dict(eye=dict(x=1.3, y=-1.3, z=1.0))),
                      margin=dict(l=0, r=0, t=0, b=0), height=520, showlegend=False)
    return fig


def grafico(df, spec):
    tipo = spec.get("tipo", "barra")
    x, cor, agg, val = spec.get("x", "classe"), spec.get("cor"), spec.get("agregacao", "contagem"), spec.get("valor")
    cols = [x] + ([cor] if cor and cor != x else [])
    if agg == "soma" and val in df.columns:
        d = df.groupby(cols, dropna=False)[val].sum().reset_index(name="valor")
    else:
        d = df.groupby(cols, dropna=False).size().reset_index(name="valor")
    titulo = spec.get("titulo", "")
    if tipo == "pizza":
        return px.pie(d, names=x, values="valor", title=titulo)
    if tipo == "treemap":
        return px.treemap(d, path=cols, values="valor", title=titulo)
    if tipo == "barra_horizontal":
        return px.bar(d, y=x, x="valor", color=cor, orientation="h", title=titulo)
    return px.bar(d, x=x, y="valor", color=cor, title=titulo)


def pedir_grafico_ia(prompt, df, chave, modelo="gemini-2.5-flash"):
    """Assistente de IA: o Gemini traduz o pedido em linguagem natural para uma especificacao de grafico."""
    import requests
    instr = (
        "Voce e um assistente de dados BIM. Colunas disponiveis: " + ", ".join(df.columns) +
        ". Responda SOMENTE um JSON com as chaves: tipo (barra, barra_horizontal, pizza ou treemap), "
        "x (coluna), cor (coluna ou null), agregacao (contagem ou soma), valor (coluna numerica ou null), titulo. "
        "Pedido do usuario: " + prompt)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent"
    r = requests.post(url, params={"key": chave}, json={"contents": [{"parts": [{"text": instr}]}]}, timeout=60)
    r.raise_for_status()
    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    txt = txt[txt.find("{"): txt.rfind("}") + 1]
    return json.loads(txt)


def relatorio_pdf(df, projeto, aluno, obs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, "Especializacao BIM UFPE - Relatorio automatico IFC", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for linha in [f"Projeto: {projeto}", f"Aluno: {aluno}", f"Data: {datetime.date.today():%d/%m/%Y}", f"Observacao: {obs}",
                  f"Elementos: {len(df)} | Classes: {df.classe.nunique()} | Tipos: {df.tipo.nunique()} | Pavimentos: {df.pavimento.nunique()}"]:
        pdf.cell(0, 6, linha.encode("latin-1", "replace").decode("latin-1"), new_x="LMARGIN", new_y="NEXT")
    for col, tit in (("classe", "Elementos por classe IFC"), ("tipo", "Elementos por tipo")):
        c = df[col].value_counts().head(12)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        c.sort_values().plot.barh(ax=ax, color="#4a6fa5")
        ax.set_title(tit); ax.set_xlabel("quantidade")
        fig.tight_layout()
        b = io.BytesIO(); fig.savefig(b, format="png", dpi=150); plt.close(fig); b.seek(0)
        pdf.image(b, w=180)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Completude de propriedades por classe", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    comp = df.groupby("classe").agg(qtd=("GlobalId", "count"), psets=("n_psets", "mean"), props=("n_props_preenchidas", "mean"))
    for cls, r in comp.iterrows():
        pdf.cell(0, 5, f"{cls}: {int(r.qtd)} elementos | media {r.psets:.1f} Psets | media {r.props:.0f} propriedades preenchidas",
                 new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


# ---------------- interface ----------------
st.sidebar.title("BIM + IA")
arquivos = st.sidebar.file_uploader("Carregue um ou mais arquivos .ifc", type=["ifc"], accept_multiple_files=True)
chave = st.sidebar.text_input("Chave de API do Gemini (assistente de IA)", type="password")
modelo_ia = st.sidebar.text_input("Modelo Gemini", "gemini-2.5-flash")

if not arquivos:
    st.title("Visualizador de dados BIM com IA")
    st.info("Carregue um arquivo IFC na barra lateral para comecar.")
    st.stop()

dfs, malhas, projeto, schema = [], [], "", ""
for a in arquivos:
    d, mm, schema, projeto = ler_ifc(a.getvalue(), a.name)
    dfs.append(d); malhas += mm
df = pd.concat(dfs, ignore_index=True)

modelos = st.sidebar.multiselect("Modelos", sorted(df.modelo.unique()), default=sorted(df.modelo.unique()))
pavs = st.sidebar.multiselect("Pavimentos", sorted(df.pavimento.unique()), default=sorted(df.pavimento.unique()))
df = df[df.modelo.isin(modelos) & df.pavimento.isin(pavs)]
malhas = [m for m in malhas if m["modelo"] in modelos]

st.title(f"{projeto}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Elementos", len(df)); c2.metric("Classes IFC", df.classe.nunique())
c3.metric("Tipos", df.tipo.nunique()); c4.metric("Pavimentos", df.pavimento.nunique())

aba3d, abadash, abaia, abarel = st.tabs(["Modelo 3D", "Dashboard", "Assistente de IA", "Relatorio"])
with aba3d:
    st.plotly_chart(figura_3d(malhas), use_container_width=True)
    st.caption(f"Esquema {schema} | {len(malhas)} elementos com geometria")
with abadash:
    a, b = st.columns(2)
    a.plotly_chart(grafico(df, {"tipo": "barra_horizontal", "x": "classe", "titulo": "Elementos por classe IFC"}), use_container_width=True)
    b.plotly_chart(grafico(df, {"tipo": "pizza", "x": "tipo", "titulo": "Distribuicao por tipo"}), use_container_width=True)
    comp = df.groupby("classe").agg(elementos=("GlobalId", "count"), media_psets=("n_psets", "mean"),
                                    media_props=("n_props_preenchidas", "mean")).round(1)
    st.subheader("Completude de propriedades (Psets)")
    st.dataframe(comp, use_container_width=True)
    if df.area_m2.notna().any():
        st.plotly_chart(grafico(df, {"tipo": "barra", "x": "tipo", "agregacao": "soma", "valor": "area_m2",
                                     "titulo": "Area (m2) por tipo"}), use_container_width=True)
with abaia:
    st.write("Peca um grafico em linguagem natural. O Gemini interpreta o pedido e o app desenha o grafico.")
    pedido = st.text_input("Pedido", "Crie um grafico de barras horizontais das entidades IFC por pavimento")
    if st.button("Gerar grafico com IA"):
        if not chave:
            st.warning("Informe a chave de API do Gemini na barra lateral.")
        else:
            try:
                spec = pedir_grafico_ia(pedido, df, chave, modelo_ia)
                st.json(spec)
                st.plotly_chart(grafico(df, spec), use_container_width=True)
            except Exception as e:
                st.error(f"Nao consegui gerar o grafico: {e}")
with abarel:
    aluno = st.text_input("Nome do aluno", "Erick Lima")
    obs = st.text_input("Observacao", "Revisao 00")
    st.download_button("Gerar relatorio PDF", relatorio_pdf(df, projeto, aluno, obs), file_name="relatorio_ifc.pdf", mime="application/pdf")
    st.download_button("Baixar dados (CSV)", df.to_csv(index=False).encode("utf-8"), file_name="dados_ifc.csv")
