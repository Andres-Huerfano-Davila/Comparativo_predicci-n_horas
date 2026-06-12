# -*- coding: utf-8 -*-
"""
Comparativo y predicción de horas de nómina - V10 final
Creado para comparar CCNómina pagada (mes vencido) vs provisión vs proyección,
con homologación basada en Maestro Posición -> Función -> Cargo homologado.
"""

import io
import os
import re
import math
import zipfile
import hashlib
import unicodedata
from datetime import datetime
from functools import reduce
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st

# ==============================
# Configuración general
# ==============================
st.set_page_config(
    page_title="Comparativo y predicción de horas de nómina",
    page_icon="🦜",
    layout="wide",
)

APP_VERSION = "V10 final - Maestro Posición → Función"
ORANGE = "#F26A21"
BLUE = "#005AA9"
GREEN = "#2E8B57"
RED = "#D62828"
YELLOW = "#F4B400"
GRAY_BG = "#F4F6F8"
DARK = "#333333"

CONCEPTOS = {
    "Y220": "Rec. Noc.",
    "Y221": "Rec. Dom noc",
    "Y300": "Hora Extra",
    "Y305": "Hora Extra",
    "Y310": "Hora Extra",
    "Y315": "Hora Extra",
    "Y350": "Compensatorio",
    "YM01": "Rec. Dom",
}
CONCEPTOS_SET = set(CONCEPTOS.keys())

INTERFAZ_MAP = {
    "Y540": "Y220",
    "Y541": "Y221",
    "Y542": "Y300",
    "Y543": "Y305",
    "Y544": "Y310",
    "Y545": "Y315",
    "Y546": "Y350",
    "Y547": "YM01",
}

KEY_DETAIL = ["periodo_novedad", "area_negocio", "cargo_homologado", "ceco", "concepto", "tipo_hora"]
KEY_EXEC = ["periodo_novedad", "area_negocio", "cargo_homologado", "concepto", "tipo_hora"]
KEY_HC = ["periodo_novedad", "area_negocio", "cargo_homologado"]
MAX_SCREEN_ROWS = 5000

# ==============================
# CSS / Branding
# ==============================
st.markdown(
    f"""
    <style>
    .main .block-container {{ padding-top: 1.0rem; }}
    .jmc-header {{
        background: linear-gradient(90deg, {ORANGE} 0%, #FF8A3D 100%);
        padding: 18px 22px;
        border-radius: 18px;
        color: white;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }}
    .jmc-title {{ font-size: 29px; font-weight: 800; margin: 0; }}
    .jmc-subtitle {{ font-size: 14px; opacity: .95; margin-top: 4px; }}
    .small-note {{ color: #6b7280; font-size: 13px; }}
    .stButton button {{ border-radius: 10px; font-weight: 700; }}
    div[data-testid="stMetric"] {{
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="jmc-header">
      <div class="jmc-title">🦜 Comparativo y predicción de horas de nómina</div>
      <div class="jmc-subtitle">{APP_VERSION} · Pagado real mes vencido · Provisión · Proyección · Headcount · Homologación por función</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Utilidades base
# ==============================
def strip_accents(text: Any) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def norm_key(text: Any) -> str:
    s = strip_accents(text).upper().strip()
    s = re.sub(r"[\n\r\t]+", " ", s)
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    return str(text).strip()


def clean_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s.endswith(".0") and re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    s = re.sub(r"\s+", "", s)
    return s


def clean_sap(value: Any) -> str:
    s = clean_code(value)
    if not s or norm_key(s) in {"ERROR", "NAN", "NONE", "NULL"}:
        return ""
    digits = re.sub(r"\D", "", s)
    return digits if len(digits) >= 5 else ""


def clean_concept(value: Any) -> str:
    s = clean_code(value).upper()
    return s


def parse_number(x: Any) -> float:
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        if math.isnan(float(x)):
            return 0.0
        return float(x)
    s = str(x).strip()
    if not s or norm_key(s) in {"NAN", "NONE", "NULL", "-"}:
        return 0.0
    s = s.replace("COP", "").replace("$", "").replace(" ", "")
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    # europeo: 1.234.567,89
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    elif "." in s:
        # Si parece separador de miles: 55.713 o 1.234.567
        if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
            s = s.replace(".", "")
    try:
        val = float(s)
        return -val if neg else val
    except Exception:
        return 0.0


def format_money(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return "$ " + f"{v:,.0f}".replace(",", ".")


def format_qty(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    s = f"{v:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def format_int(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return f"{v:,.0f}".replace(",", ".")


def format_pct(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    s = f"{v:,.2f}%"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def parse_period_any(value: Any, fallback_name: str = "") -> str:
    """Retorna MM.YYYY."""
    if pd.notna(value) and not isinstance(value, (bytes, bytearray)):
        if isinstance(value, (pd.Timestamp, datetime)):
            return f"{value.month:02d}.{value.year}"
        s = str(value).strip()
        if s and norm_key(s) not in {"NAN", "NONE", "NULL", "ERROR"}:
            # 202602
            m = re.search(r"(20\d{2})(0[1-9]|1[0-2])", s)
            if m:
                return f"{m.group(2)}.{m.group(1)}"
            # 02.2026 o 02/2026 o 02-2026
            m = re.search(r"(0?[1-9]|1[0-2])[\.\-/ ](20\d{2})", s)
            if m:
                return f"{int(m.group(1)):02d}.{m.group(2)}"
            # 022026
            m = re.search(r"\b(0[1-9]|1[0-2])(20\d{2})\b", s)
            if m:
                return f"{m.group(1)}.{m.group(2)}"
    if fallback_name:
        return parse_period_any(str(fallback_name), "")
    return ""


def prev_period(period: str) -> str:
    if not period:
        return ""
    m = re.match(r"(0[1-9]|1[0-2])\.(20\d{2})", str(period))
    if not m:
        return ""
    month = int(m.group(1)); year = int(m.group(2))
    month -= 1
    if month == 0:
        month = 12; year -= 1
    return f"{month:02d}.{year}"


def period_sort_key(period: str) -> int:
    m = re.match(r"(0[1-9]|1[0-2])\.(20\d{2})", str(period))
    if not m:
        return 999999
    return int(m.group(2))*100 + int(m.group(1))


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = False) -> Optional[str]:
    normalized_cols = {norm_key(c): c for c in df.columns}
    cand_norm = [norm_key(c) for c in candidates]
    for cn in cand_norm:
        if cn in normalized_cols:
            return normalized_cols[cn]
    # fuzzy contains, prefer shortest
    for cn in cand_norm:
        matches = [orig for nk, orig in normalized_cols.items() if cn and (cn in nk or nk in cn)]
        if matches:
            return sorted(matches, key=lambda x: len(str(x)))[0]
    if required:
        raise ValueError(f"No encontré columna para: {candidates}")
    return None


def coalesce_cols(df: pd.DataFrame, cols: List[Optional[str]]) -> pd.Series:
    out = pd.Series([""] * len(df), index=df.index, dtype="object")
    for c in cols:
        if c and c in df.columns:
            s = df[c].apply(clean_text)
            out = out.mask(out.eq(""), s)
    return out


def drop_duplicated_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.loc[:, ~pd.Index(df.columns).duplicated()].copy()


def classify_area(ceco: Any = "", division: Any = "", tipo: Any = "", area_nomina: Any = "", cargo: Any = "") -> str:
    ceco_s = clean_code(ceco)
    combo = norm_key(" ".join([clean_text(division), clean_text(tipo), clean_text(area_nomina), clean_text(cargo)]))
    if "BODEGA CANASTO" in combo or re.search(r"\bBDC\b", combo):
        return "BDC"
    if "LOGISTICA" in combo or "DISTR CENTER" in combo or "CENTRO DE DISTRIB" in combo or "CEDI" in combo:
        return "CEDI"
    if "ADMINISTRAT" in combo or "OFICINA" in combo or "SOPORTE" in combo or "HQ" in combo:
        return "Oficina Soporte"
    if ceco_s.startswith("102"):
        return "CEDI"
    if ceco_s.startswith("103"):
        return "Oficina Soporte"
    if ceco_s.startswith("101"):
        return "Tiendas"
    return "Sin clasificar"


def is_manager_excl(area_personal: Any = "", area_nomina: Any = "", cargo: Any = "") -> bool:
    s = norm_key(" ".join([clean_text(area_personal), clean_text(area_nomina), clean_text(cargo)]))
    # No excluir Non Manager
    if "NON MANAGER" in s:
        return False
    # Manager I, II, III, IV o Manager 1-4
    if re.search(r"\bMANAGER\s*(I|II|III|IV|1|2|3|4)\b", s):
        return True
    return False


def ensure_key_types(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    df = df.copy()
    for k in keys:
        if k not in df.columns:
            df[k] = ""
        df[k] = df[k].fillna("").astype(str)
    return df

# ==============================
# Lectura de archivos
# ==============================
def get_bytes(uploaded_file) -> bytes:
    if uploaded_file is None:
        return b""
    return uploaded_file.getvalue()


def _read_excel_bytes(data: bytes, name: str, sheet_name=0, nrows=None, usecols=None) -> pd.DataFrame:
    ext = os.path.splitext(name)[1].lower()
    bio = io.BytesIO(data)
    # xlsb
    if ext == ".xlsb":
        return pd.read_excel(bio, sheet_name=sheet_name, engine="pyxlsb", nrows=nrows, usecols=usecols)
    # xlsx / xlsm / xls
    for engine in ["calamine", "openpyxl", None]:
        try:
            bio.seek(0)
            if engine:
                return pd.read_excel(bio, sheet_name=sheet_name, engine=engine, nrows=nrows, usecols=usecols)
            return pd.read_excel(bio, sheet_name=sheet_name, nrows=nrows, usecols=usecols)
        except Exception:
            continue
    # último intento genérico
    bio.seek(0)
    return pd.read_excel(bio, sheet_name=sheet_name, nrows=nrows, usecols=usecols)


def read_excel_upload(uploaded_file, sheet_name=0, nrows=None, usecols=None) -> pd.DataFrame:
    return _read_excel_bytes(uploaded_file.getvalue(), uploaded_file.name, sheet_name=sheet_name, nrows=nrows, usecols=usecols)


def read_sap_text_bytes(data: bytes) -> pd.DataFrame:
    # SAP list export usually UTF-16 LE, tab separated, with header line after a few title rows.
    decoded = None
    for enc in ["utf-16", "latin1", "utf-8-sig", "utf-8"]:
        try:
            decoded = data.decode(enc, errors="ignore")
            if "Nº pers." in decoded or "N° pers." in decoded or "CC-n." in decoded:
                break
        except Exception:
            continue
    if decoded is None:
        decoded = data.decode("latin1", errors="ignore")
    lines = decoded.splitlines(True)
    header_idx = None
    for i, line in enumerate(lines):
        if ("Nº pers." in line or "N° pers." in line or "N pers" in line) and ("CC-n." in line or "Per.para" in line):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("No encontré encabezado SAP en archivo texto")
    text = "".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str, engine="python")
    df = df.dropna(axis=1, how="all").dropna(how="all")
    # remover columnas unnamed/vacías
    keep = [c for c in df.columns if not str(c).startswith("Unnamed") and str(c).strip() != ""]
    df = df[keep].copy()
    return df


def read_any_upload(uploaded_file, sheet_name=0) -> pd.DataFrame:
    data = uploaded_file.getvalue()
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower()
    if ext in [".txt", ".csv"] or ext == ".xls":
        # Muchas salidas SAP vienen .XLS pero realmente son texto UTF-16 tabulado.
        try:
            return read_sap_text_bytes(data)
        except Exception:
            pass
    return read_excel_upload(uploaded_file, sheet_name=sheet_name)


def list_sheets(uploaded_file) -> List[str]:
    data = uploaded_file.getvalue(); name = uploaded_file.name; ext = os.path.splitext(name)[1].lower()
    if ext == ".xlsb":
        xl = pd.ExcelFile(io.BytesIO(data), engine="pyxlsb")
    else:
        xl = pd.ExcelFile(io.BytesIO(data))
    return xl.sheet_names

# ==============================
# Homologación
# ==============================
def read_detalle_horas(detalle_file) -> Tuple[Dict[str, str], Dict[str, str], pd.DataFrame]:
    """Retorna (concepto->tipo_hora, funcion_key->cargo_homologado, tabla)."""
    concept_map = dict(CONCEPTOS)
    func_map: Dict[str, str] = {}
    audit_rows = []
    if detalle_file is not None:
        try:
            df = read_excel_upload(detalle_file, sheet_name="Homologación", nrows=5000)
        except Exception:
            df = read_excel_upload(detalle_file, sheet_name=0, nrows=5000)
        df = drop_duplicated_columns(df).dropna(how="all")
        # limpiar filas vacías masivas de xlsb
        df = df.loc[df.notna().sum(axis=1) > 0].copy()
        c_cc = find_col(df, ["CC-n.", "CC-n", "Concepto"], False)
        c_hora = find_col(df, ["Hora", "Tipo hora"], False)
        if c_cc and c_hora:
            tmp = df[[c_cc, c_hora]].dropna(how="all").copy()
            for _, r in tmp.iterrows():
                cc = clean_concept(r.get(c_cc))
                hora = clean_text(r.get(c_hora))
                if cc and hora:
                    concept_map[cc] = hora
        # funciones a cargos homologados
        c_func_code = find_col(df, ["Función", "Funcion"], False)
        c_func_text = find_col(df, ["Función_4", "Funcion_4", "Nombre función", "Nombre funcion"], False)
        c_cargo = find_col(df, ["Cargo"], False)
        if c_cargo:
            for _, r in df.iterrows():
                cargo = clean_text(r.get(c_cargo))
                if not cargo:
                    continue
                if c_func_code:
                    fc = clean_code(r.get(c_func_code))
                    if fc:
                        func_map[norm_key(fc)] = cargo
                        audit_rows.append({"origen": "Detalle Horas", "tipo_llave": "codigo_funcion", "llave": fc, "funcion": clean_text(r.get(c_func_text)) if c_func_text else "", "cargo_homologado": cargo})
                if c_func_text:
                    ft = clean_text(r.get(c_func_text))
                    if ft:
                        func_map[norm_key(ft)] = cargo
                        audit_rows.append({"origen": "Detalle Horas", "tipo_llave": "texto_funcion", "llave": ft, "funcion": ft, "cargo_homologado": cargo})
    # manual conceptos
    concept_map.setdefault("Y350", "Compensatorio")
    audit = pd.DataFrame(audit_rows).drop_duplicates() if audit_rows else pd.DataFrame(columns=["origen","tipo_llave","llave","funcion","cargo_homologado"])
    return concept_map, func_map, audit


def fuzzy_cargo_from_function(funcion: Any, area_negocio: str = "") -> str:
    s = norm_key(funcion)
    a = norm_key(area_negocio)
    if not s:
        return "Sin homologar"
    if "APRENDIZ" in s:
        return "Aprendiz"
    if "PART TIME" in s or "PARTTIME" in s:
        return "Part time"
    if "JEFE" in s and "TIENDA" in s:
        return "Jefe Tienda"
    if "SUPERVISOR" in s and ("JR" in s or "JUNIOR" in s):
        return "Supervisor Jr"
    if "SUPERVISOR" in s and "TIENDA" in s:
        return "Supervisor Tienda"
    if "OPERADOR" in s and "TIENDA" in s:
        return "Operador Tienda"
    if "MONTACARGA" in s and ("BDC" in s or a == "BDC"):
        return "Op . Montacarga BDC"
    if "MONTACARGA" in s:
        return "Montacarga Cedi"
    if "CENTRO DE DISTRIB" in s or "CEDI" in s or "DISTRIBUCION" in s or ("OPERADOR" in s and a == "CEDI"):
        return "Op. Cedi"
    if a == "OFICINA SOPORTE" or "ANALISTA" in s or "COORDINADOR" in s or "ESPECIALISTA" in s or "ASISTENTE" in s:
        return "Oficina Soporte"
    return "Sin homologar"


def homologate_function(func_code: Any, func_text: Any, func_map: Dict[str, str], area_negocio: str = "") -> Tuple[str, str]:
    fc = clean_code(func_code)
    ft = clean_text(func_text)
    if fc and norm_key(fc) in func_map:
        return func_map[norm_key(fc)], "funcion_codigo_detalle"
    if ft and norm_key(ft) in func_map:
        return func_map[norm_key(ft)], "funcion_texto_detalle"
    # si el texto que llega ya es un cargo homologado conocido
    if ft:
        fuzzy = fuzzy_cargo_from_function(ft, area_negocio)
        if fuzzy != "Sin homologar":
            return fuzzy, "funcion_fuzzy"
        return ft, "funcion_sin_mapeo"
    return "Sin homologar", "sin_funcion"


def build_position_function_master(hc_df: pd.DataFrame, poshom_df: Optional[pd.DataFrame] = None) -> Tuple[Dict[str, Dict[str, str]], pd.DataFrame]:
    rows = []
    def add(pos_code, pos_text, func_code, func_text, source, periodo=""):
        pos_code = clean_code(pos_code); pos_text = clean_text(pos_text); func_code = clean_code(func_code); func_text = clean_text(func_text)
        if not func_text and not func_code:
            return
        for key_val, key_type in [(pos_text, "posicion_texto"), (pos_code, "posicion_codigo"), (func_text, "funcion_texto"), (func_code, "funcion_codigo")]:
            if key_val:
                rows.append({
                    "key": norm_key(key_val), "llave_original": key_val, "tipo_llave": key_type,
                    "posicion_codigo": pos_code, "posicion_nombre": pos_text,
                    "funcion_codigo": func_code, "funcion_nombre": func_text,
                    "origen": source, "periodo": periodo,
                })
    if poshom_df is not None and not poshom_df.empty:
        poshom_df = drop_duplicated_columns(poshom_df)
        cp_code = find_col(poshom_df, ["Posición", "Posicion"], False)
        cp_text = find_col(poshom_df, ["Posición_3", "Posicion_3", "Nombre posición", "Nombre posicion", "Posición nombre"], False)
        cf_code = find_col(poshom_df, ["Función", "Funcion"], False)
        cf_text = find_col(poshom_df, ["Función_4", "Funcion_4", "Nombre función", "Nombre funcion"], False)
        for _, r in poshom_df.iterrows():
            add(r.get(cp_code) if cp_code else "", r.get(cp_text) if cp_text else "", r.get(cf_code) if cf_code else "", r.get(cf_text) if cf_text else "", "Posiciones_homologadas")
    if hc_df is not None and not hc_df.empty:
        for _, r in hc_df.iterrows():
            add(r.get("posicion_codigo", ""), r.get("posicion_nombre", ""), r.get("funcion_codigo", ""), r.get("funcion_nombre", ""), "Headcount", r.get("periodo_novedad", ""))
    if not rows:
        return {}, pd.DataFrame()
    raw = pd.DataFrame(rows)
    # Prioridad: Posiciones_homologadas > Headcount; dentro de llave, tomar la combinación más frecuente
    raw["prioridad"] = np.where(raw["origen"].eq("Posiciones_homologadas"), 0, 1)
    raw["combo"] = raw["funcion_codigo"].fillna("").astype(str) + "|" + raw["funcion_nombre"].fillna("").astype(str)
    counts = raw.groupby(["key", "combo"], dropna=False).size().reset_index(name="conteo")
    raw = raw.merge(counts, on=["key", "combo"], how="left")
    raw = raw.sort_values(["key", "prioridad", "conteo"], ascending=[True, True, False])
    best = raw.drop_duplicates("key", keep="first").copy()
    mapping = {
        r["key"]: {
            "funcion_codigo": r.get("funcion_codigo", ""),
            "funcion_nombre": r.get("funcion_nombre", ""),
            "posicion_codigo": r.get("posicion_codigo", ""),
            "posicion_nombre": r.get("posicion_nombre", ""),
            "origen_maestro": r.get("origen", ""),
        }
        for _, r in best.iterrows()
    }
    return mapping, raw.drop(columns=["combo"], errors="ignore")


def resolve_function_from_master(value: Any, master: Dict[str, Dict[str, str]]) -> Tuple[str, str, str]:
    key = norm_key(value)
    if key and key in master:
        info = master[key]
        return info.get("funcion_codigo", ""), info.get("funcion_nombre", ""), info.get("origen_maestro", "maestro")
    return "", "", "sin_maestro"

# ==============================
# Procesadores de fuentes
# ==============================
def process_headcount(files: List[Any], concept_map: Dict[str, str], func_map: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    all_rows = []
    alerts = []
    for f in files or []:
        try:
            df = read_any_upload(f)
            df = drop_duplicated_columns(df)
            c_sap = find_col(df, ["Nº pers.", "N° pers.", "SAP"], True)
            c_status = find_col(df, ["Status ocupación", "Status ocupacion", "Status"], False)
            c_div = find_col(df, ["División de personal", "Division de personal"], False)
            c_area_pers = find_col(df, ["Área de personal", "Area de personal"], False)
            c_area_nom = find_col(df, ["Área de nómina", "Area de nomina", "Texto área nómina"], False)
            c_ceco = find_col(df, ["Ce.coste", "Ce coste", "CECO"], False)
            c_pos_code = find_col(df, ["Posición", "Posicion"], False)
            c_pos_text = find_col(df, ["Posición.1", "Posicion.1", "Posición_3", "Posicion_3"], False)
            c_func_code = find_col(df, ["Función", "Funcion"], False)
            c_func_text = find_col(df, ["Función.1", "Funcion.1", "Función_4", "Funcion_4"], False)
            periodo = parse_period_any("", f.name)
            out = pd.DataFrame()
            out["periodo_novedad"] = periodo
            out["sap"] = df[c_sap].apply(clean_sap)
            out["status"] = df[c_status].apply(clean_text) if c_status else ""
            out["division"] = df[c_div].apply(clean_text) if c_div else ""
            out["area_personal"] = df[c_area_pers].apply(clean_text) if c_area_pers else ""
            out["area_nomina"] = df[c_area_nom].apply(clean_text) if c_area_nom else ""
            out["ceco"] = df[c_ceco].apply(clean_code) if c_ceco else ""
            out["posicion_codigo"] = df[c_pos_code].apply(clean_code) if c_pos_code else ""
            out["posicion_nombre"] = df[c_pos_text].apply(clean_text) if c_pos_text else ""
            out["funcion_codigo"] = df[c_func_code].apply(clean_code) if c_func_code else ""
            out["funcion_nombre"] = df[c_func_text].apply(clean_text) if c_func_text else ""
            out["area_negocio"] = [classify_area(ceco, div, "", area, pos) for ceco, div, area, pos in zip(out["ceco"], out["division"], out["area_nomina"], out["posicion_nombre"])]
            out["manager_excluido"] = [is_manager_excl(ap, an, pos) for ap, an, pos in zip(out["area_personal"], out["area_nomina"], out["posicion_nombre"])]
            out["cargo_homologado"], out["metodo_cargo"] = zip(*[homologate_function(fc, ft, func_map, area) for fc, ft, area in zip(out["funcion_codigo"], out["funcion_nombre"], out["area_negocio"])])
            all_rows.append(out)
            alerts.append({"tipo":"Cargue", "mensaje":f"Headcount {f.name}: {len(out):,} registros leídos"})
        except Exception as e:
            alerts.append({"tipo":"Error", "mensaje":f"Error procesando Headcount {getattr(f,'name','archivo')}: {e}"})
    hc_full = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if hc_full.empty:
        return hc_full, pd.DataFrame(), pd.DataFrame(), alerts
    hc_full = hc_full[hc_full["sap"].ne("")].copy()
    excl = hc_full[hc_full["manager_excluido"]].copy()
    hc_valid = hc_full[~hc_full["manager_excluido"]].copy()
    # HC por mes, área, cargo. Contar SAP único.
    hc_group = hc_valid.groupby(KEY_HC, dropna=False)["sap"].nunique().reset_index(name="hc")
    hc_group = ensure_key_types(hc_group, KEY_HC)
    if len(excl):
        alerts.append({"tipo":"Headcount", "mensaje":f"Headcount: se excluyeron {len(excl):,} registros Manager I-IV/no aplican horas"})
    return hc_full, hc_group, excl, alerts


def add_function_and_cargo(df: pd.DataFrame, source: str, master: Dict[str, Dict[str, str]], func_map: Dict[str, str], hc_full: pd.DataFrame, alerts: List[Dict[str, Any]], prefer_sap: bool = False) -> pd.DataFrame:
    """Añade función_periodo y cargo_homologado. Prioridad:
    1) SAP+periodo en HC si prefer_sap=True y está disponible.
    2) función directa si viene.
    3) cargo/posición contra Maestro Posición→Función.
    4) fallback fuzzy.
    """
    df = df.copy()
    for c in ["sap", "periodo_novedad", "posicion_original", "funcion_codigo", "funcion_nombre", "area_negocio"]:
        if c not in df.columns:
            df[c] = ""
    df["funcion_codigo_final"] = df["funcion_codigo"].apply(clean_code)
    df["funcion_nombre_final"] = df["funcion_nombre"].apply(clean_text)
    df["origen_funcion"] = np.where(df["funcion_nombre_final"].ne("") | df["funcion_codigo_final"].ne(""), "funcion_fuente", "")

    # SAP + periodo en HC, útil sobre todo para provisión
    if prefer_sap and hc_full is not None and not hc_full.empty:
        hc_sap = hc_full[["sap", "periodo_novedad", "funcion_codigo", "funcion_nombre", "posicion_nombre", "area_negocio"]].drop_duplicates(["sap","periodo_novedad"])
        before_cols = list(df.columns)
        df = df.merge(hc_sap, on=["sap","periodo_novedad"], how="left", suffixes=("", "_hc"))
        mask = (df["funcion_nombre_final"].eq("") | df["funcion_nombre_final"].isna()) & df["funcion_nombre_hc"].fillna("").ne("")
        df.loc[mask, "funcion_nombre_final"] = df.loc[mask, "funcion_nombre_hc"].apply(clean_text)
        df.loc[mask, "funcion_codigo_final"] = df.loc[mask, "funcion_codigo_hc"].apply(clean_code)
        df.loc[mask, "origen_funcion"] = "sap_periodo_headcount"
        # Completar área desde HC si venía vacía/sin clasificar
        mask_area = df["area_negocio"].isin(["", "Sin clasificar"]) & df.get("area_negocio_hc", pd.Series([""]*len(df))).fillna("").ne("")
        if "area_negocio_hc" in df.columns:
            df.loc[mask_area, "area_negocio"] = df.loc[mask_area, "area_negocio_hc"]
        # limpiar columnas _hc auxiliares excepto información que no moleste
        drop_cols = [c for c in df.columns if c.endswith("_hc")]
        df = df.drop(columns=drop_cols, errors="ignore")

    # Usar maestro posición -> función si aún no hay función
    for idx in df.index[df["funcion_nombre_final"].fillna("").eq("") & df["posicion_original"].fillna("").ne("")]:
        fc, ft, origin = resolve_function_from_master(df.at[idx, "posicion_original"], master)
        if ft or fc:
            df.at[idx, "funcion_codigo_final"] = fc
            df.at[idx, "funcion_nombre_final"] = ft
            df.at[idx, "origen_funcion"] = f"maestro_posicion_funcion:{origin}"

    # Si lo que viene como posicion_original es realmente función y existe en detalle, también sirve.
    for idx in df.index[df["funcion_nombre_final"].fillna("").eq("") & df["posicion_original"].fillna("").ne("")]:
        txt = df.at[idx, "posicion_original"]
        if norm_key(txt) in func_map:
            df.at[idx, "funcion_nombre_final"] = txt
            df.at[idx, "origen_funcion"] = "texto_ya_es_funcion"

    # Homologar función -> cargo homologado
    cargos = []
    methods = []
    for fc, ft, area in zip(df["funcion_codigo_final"], df["funcion_nombre_final"], df["area_negocio"]):
        cargo, metodo = homologate_function(fc, ft, func_map, area)
        cargos.append(cargo)
        methods.append(metodo)
    df["cargo_homologado"] = cargos
    df["metodo_cargo"] = methods

    pendientes = df[df["cargo_homologado"].isin(["", "Sin homologar"])]
    if len(pendientes):
        alerts.append({"tipo":"Homologación", "mensaje":f"{source}: {len(pendientes):,} registros quedaron sin cargo homologado. Revise hoja Pendientes_Homologacion."})
    return df


def process_pagado(cc_files: List[Any], comp_files: List[Any], concept_map: Dict[str, str], func_map: Dict[str, str], master: Dict[str, Dict[str, str]], hc_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    rows = []
    alerts = []
    all_files = [(f, "CCNómina") for f in (cc_files or [])] + [(f, "Compensatorios") for f in (comp_files or [])]
    for f, source in all_files:
        try:
            df = read_any_upload(f)
            df = drop_duplicated_columns(df)
            c_sap = find_col(df, ["Nº pers.", "N° pers.", "SAP"], False)
            c_period = find_col(df, ["Per.para", "Periodo", "Periodo para nómina", "Periodo para nomina"], False)
            c_fecha_pago = find_col(df, ["Fecha pago", "Fecha de pago"], False)
            c_concept = find_col(df, ["CC-n.", "CC-n", "Concepto", "Valores"], True)
            c_text = find_col(df, ["Texto expl.CC-nómina", "Texto expl.CC-nomina", "Txt.expl.", "Texto concepto"], False)
            c_qty = find_col(df, ["Cantidad", "Total"], True)
            c_value = find_col(df, ["Importe", "     Importe", "   Importe", "Valor"], True)
            c_ceco = find_col(df, ["Ce.coste", "CECO", "Ce coste"], False)
            c_div = find_col(df, ["Texto división de personal", "Texto division de personal", "División de personal"], False)
            c_area_nom = find_col(df, ["Texto área nómina", "Texto area nomina", "Área de nómina", "Area de nomina"], False)
            c_func_code = find_col(df, ["Función", " Funcion", "Función ", "Funcion"], False)
            c_func_text = find_col(df, ["Denominación función", "Denominacion funcion", "Función.1", "Funcion.1"], False)

            out = pd.DataFrame()
            out["fuente"] = source
            out["archivo"] = f.name
            out["sap"] = df[c_sap].apply(clean_sap) if c_sap else ""
            out["periodo_pago"] = df[c_period].apply(lambda x: parse_period_any(x, f.name)) if c_period else parse_period_any("", f.name)
            # si no detecta periodo_pago por columna, intenta fecha pago o nombre
            if c_fecha_pago:
                mask_empty = out["periodo_pago"].eq("")
                out.loc[mask_empty, "periodo_pago"] = df.loc[mask_empty, c_fecha_pago].apply(lambda x: parse_period_any(x, f.name))
            out["periodo_novedad"] = out["periodo_pago"].apply(prev_period)
            out["concepto"] = df[c_concept].apply(clean_concept)
            out["concepto"] = out["concepto"].map(lambda x: INTERFAZ_MAP.get(x, x))
            out["tipo_hora"] = out["concepto"].map(concept_map).fillna(out["concepto"].map(CONCEPTOS)).fillna("Sin tipo hora")
            out["cantidad_pagada"] = df[c_qty].apply(parse_number)
            out["valor_pagado"] = df[c_value].apply(parse_number)
            out["ceco"] = df[c_ceco].apply(clean_code) if c_ceco else ""
            div = df[c_div].apply(clean_text) if c_div else pd.Series([""]*len(df))
            area_nom = df[c_area_nom].apply(clean_text) if c_area_nom else pd.Series([""]*len(df))
            func_text = df[c_func_text].apply(clean_text) if c_func_text else pd.Series([""]*len(df))
            out["funcion_codigo"] = df[c_func_code].apply(clean_code) if c_func_code else ""
            out["funcion_nombre"] = func_text
            out["posicion_original"] = func_text  # En CC nómina normalmente viene función; si no, se tratará como texto
            out["area_negocio"] = [classify_area(ceco, d, "", an, ft) for ceco, d, an, ft in zip(out["ceco"], div, area_nom, func_text)]
            out = out[out["concepto"].isin(CONCEPTOS_SET)].copy()
            out = out[(out["cantidad_pagada"].abs() > 0) | (out["valor_pagado"].abs() > 0)].copy()
            out = add_function_and_cargo(out, source, master, func_map, hc_full, alerts, prefer_sap=False)
            rows.append(out)
            alerts.append({"tipo":"Cargue", "mensaje":f"{source} {f.name}: {len(out):,} registros útiles procesados"})
        except Exception as e:
            alerts.append({"tipo":"Error", "mensaje":f"Error procesando {source} {getattr(f,'name','archivo')}: {e}"})
    full = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if full.empty:
        return full, pd.DataFrame(), alerts
    agg = full.groupby(KEY_DETAIL, dropna=False).agg(cantidad_pagada=("cantidad_pagada","sum"), valor_pagado=("valor_pagado","sum")).reset_index()
    agg = ensure_key_types(agg, KEY_DETAIL)
    return full, agg, alerts


def process_provision(files: List[Any], concept_map: Dict[str, str], func_map: Dict[str, str], master: Dict[str, Dict[str, str]], hc_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    rows = []
    alerts = []
    for f in files or []:
        try:
            # Buscar hoja Horas_Provisión, si no existe primera hoja
            try:
                df = read_excel_upload(f, sheet_name="Horas_Provisión")
            except Exception:
                df = read_any_upload(f)
            df = drop_duplicated_columns(df)
            c_source = find_col(df, ["Source.Name", "Source Name", "MES", "Periodo"], False)
            c_sap = find_col(df, ["IdentificacionEmpleado", "IdentificaciónEmpleado", "SAP", "Nº pers."], False)
            c_ceco = find_col(df, ["CECO", "Ce.coste", "Ce coste"], False)
            c_tipo = find_col(df, ["TIPO", "Tipo"], False)
            c_cargo = find_col(df, ["CARGO", "Cargo"], True)
            c_concept = find_col(df, ["Valores", "Concepto", "CC-n."], True)
            c_qty = find_col(df, ["Total", "Cantidad"], True)
            c_value = find_col(df, ["PROVISIÓN", "PROVISION", "Provisión", "Provision"], True)
            c_region = find_col(df, ["REGION", "Región", "Division"], False)

            out = pd.DataFrame()
            out["fuente"] = "Provisión"
            out["archivo"] = f.name
            out["periodo_novedad"] = df[c_source].apply(lambda x: parse_period_any(x, f.name)) if c_source else parse_period_any("", f.name)
            out["sap"] = df[c_sap].apply(clean_sap) if c_sap else ""
            out["ceco"] = df[c_ceco].apply(clean_code) if c_ceco else ""
            cargo = df[c_cargo].apply(clean_text)
            tipo = df[c_tipo].apply(clean_text) if c_tipo else pd.Series([""]*len(df))
            region = df[c_region].apply(clean_text) if c_region else pd.Series([""]*len(df))
            out["posicion_original"] = cargo
            out["funcion_codigo"] = ""
            out["funcion_nombre"] = ""
            out["concepto"] = df[c_concept].apply(clean_concept)
            out["tipo_hora"] = out["concepto"].map(concept_map).fillna(out["concepto"].map(CONCEPTOS)).fillna("Sin tipo hora")
            out["cantidad_provisionada"] = df[c_qty].apply(parse_number)
            out["valor_provisionado"] = df[c_value].apply(parse_number)
            out["area_negocio"] = [classify_area(ceco, reg, ti, "", cg) for ceco, reg, ti, cg in zip(out["ceco"], region, tipo, cargo)]
            out = out[out["concepto"].isin(CONCEPTOS_SET)].copy()
            out = out[(out["cantidad_provisionada"].abs() > 0) | (out["valor_provisionado"].abs() > 0)].copy()
            out = add_function_and_cargo(out, "Provisión", master, func_map, hc_full, alerts, prefer_sap=True)
            rows.append(out)
            alerts.append({"tipo":"Cargue", "mensaje":f"Provisión {f.name}: {len(out):,} registros útiles procesados"})
        except Exception as e:
            alerts.append({"tipo":"Error", "mensaje":f"Error procesando provisión {getattr(f,'name','archivo')}: {e}"})
    full = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if full.empty:
        return full, pd.DataFrame(), alerts
    agg = full.groupby(KEY_DETAIL, dropna=False).agg(cantidad_provisionada=("cantidad_provisionada","sum"), valor_provisionado=("valor_provisionado","sum")).reset_index()
    agg = ensure_key_types(agg, KEY_DETAIL)
    return full, agg, alerts


def process_proyeccion(files: List[Any], concept_map: Dict[str, str], func_map: Dict[str, str], master: Dict[str, Dict[str, str]], hc_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    rows = []
    alerts = []
    for f in files or []:
        try:
            try:
                df = read_excel_upload(f, sheet_name="Horas_Proyección")
            except Exception:
                df = read_any_upload(f)
            df = drop_duplicated_columns(df)
            c_mes = find_col(df, ["MES", "Source.Name", "Periodo"], False)
            c_sap = find_col(df, ["SAP", "Nº pers."], False)
            c_func = find_col(df, ["Funcion", "Función", "Cargo"], False)
            c_ceco = find_col(df, ["Ce.coste", "CECO"], False)
            c_tipo = find_col(df, ["Tipo", "TIPO"], False)
            c_area_nom = find_col(df, ["Área de nómina", "Area de nomina"], False)
            base_cols = [c for c in [c_mes, c_sap, c_func, c_ceco, c_tipo, c_area_nom] if c]
            tmp_base = df[base_cols].copy() if base_cols else pd.DataFrame(index=df.index)
            long_rows = []
            for concepto in CONCEPTOS_SET:
                q_col = None; v_col = None
                # Preferir nombre exacto
                for c in df.columns:
                    if norm_key(c) == norm_key(f"{concepto}_Q"):
                        q_col = c
                    if norm_key(c) == norm_key(f"{concepto}_$") or str(c).strip().upper() == f"{concepto}_$":
                        v_col = c
                if q_col is None and f"{concepto}_Q" in df.columns:
                    q_col = f"{concepto}_Q"
                if v_col is None and f"{concepto}_$" in df.columns:
                    v_col = f"{concepto}_$"
                if q_col is None and v_col is None:
                    continue
                part = tmp_base.copy()
                part["concepto"] = concepto
                part["cantidad_proyectada"] = df[q_col].apply(parse_number) if q_col else 0.0
                part["valor_proyectado"] = df[v_col].apply(parse_number) if v_col else 0.0
                part = part[(part["cantidad_proyectada"].abs() > 0) | (part["valor_proyectado"].abs() > 0)]
                long_rows.append(part)
            if not long_rows:
                alerts.append({"tipo":"Proyección", "mensaje":f"Proyección {f.name}: no se encontraron columnas *_Q / *_$ con movimiento"})
                continue
            long = pd.concat(long_rows, ignore_index=True)
            out = pd.DataFrame()
            out["fuente"] = "Proyección"
            out["archivo"] = f.name
            out["periodo_novedad"] = long[c_mes].apply(lambda x: parse_period_any(x, f.name)) if c_mes else parse_period_any("", f.name)
            out["sap"] = long[c_sap].apply(clean_sap) if c_sap else ""
            out["ceco"] = long[c_ceco].apply(clean_code) if c_ceco else ""
            func = long[c_func].apply(clean_text) if c_func else pd.Series([""]*len(long))
            tipo = long[c_tipo].apply(clean_text) if c_tipo else pd.Series([""]*len(long))
            area_nom = long[c_area_nom].apply(clean_text) if c_area_nom else pd.Series([""]*len(long))
            out["posicion_original"] = func
            out["funcion_codigo"] = ""
            out["funcion_nombre"] = func  # Proyección suele traer función textual
            out["concepto"] = long["concepto"].apply(clean_concept)
            out["tipo_hora"] = out["concepto"].map(concept_map).fillna(out["concepto"].map(CONCEPTOS)).fillna("Sin tipo hora")
            out["cantidad_proyectada"] = long["cantidad_proyectada"].astype(float)
            out["valor_proyectado"] = long["valor_proyectado"].astype(float)
            out["area_negocio"] = [classify_area(ceco, "", ti, an, fu) for ceco, ti, an, fu in zip(out["ceco"], tipo, area_nom, func)]
            invalid_sap = out["sap"].eq("").sum()
            if invalid_sap:
                alerts.append({"tipo":"Proyección", "mensaje":f"{invalid_sap:,} registros de proyección tenían SAP inválido/Error; se homologaron por función/cargo."})
            out = add_function_and_cargo(out, "Proyección", master, func_map, hc_full, alerts, prefer_sap=False)
            rows.append(out)
            alerts.append({"tipo":"Cargue", "mensaje":f"Proyección {f.name}: {len(out):,} registros útiles procesados"})
        except Exception as e:
            alerts.append({"tipo":"Error", "mensaje":f"Error procesando proyección {getattr(f,'name','archivo')}: {e}"})
    full = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if full.empty:
        return full, pd.DataFrame(), alerts
    agg = full.groupby(KEY_DETAIL, dropna=False).agg(cantidad_proyectada=("cantidad_proyectada","sum"), valor_proyectado=("valor_proyectado","sum")).reset_index()
    agg = ensure_key_types(agg, KEY_DETAIL)
    return full, agg, alerts

# ==============================
# Agregación y reportes
# ==============================
def empty_agg(cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=KEY_DETAIL + cols)


def aggregate_sources(pagado: pd.DataFrame, provision: pd.DataFrame, proyeccion: pd.DataFrame, hc: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    p = pagado.copy() if pagado is not None and not pagado.empty else empty_agg(["cantidad_pagada", "valor_pagado"])
    pr = provision.copy() if provision is not None and not provision.empty else empty_agg(["cantidad_provisionada", "valor_provisionado"])
    py = proyeccion.copy() if proyeccion is not None and not proyeccion.empty else empty_agg(["cantidad_proyectada", "valor_proyectado"])
    for df in [p, pr, py]:
        for k in KEY_DETAIL:
            if k not in df.columns:
                df[k] = ""
            df[k] = df[k].fillna("").astype(str)
    comp = p.merge(pr, on=KEY_DETAIL, how="outer")
    comp = comp.merge(py, on=KEY_DETAIL, how="outer")
    for c in ["cantidad_pagada", "valor_pagado", "cantidad_provisionada", "valor_provisionado", "cantidad_proyectada", "valor_proyectado"]:
        if c not in comp.columns:
            comp[c] = 0.0
        comp[c] = comp[c].fillna(0.0).astype(float)
    # HC por cargo/área/mes
    hc_use = hc.copy() if hc is not None and not hc.empty else pd.DataFrame(columns=KEY_HC + ["hc"])
    hc_use = ensure_key_types(hc_use, KEY_HC)
    hc_use["hc"] = hc_use.get("hc", 0).fillna(0).astype(float)
    comp = comp.merge(hc_use, on=KEY_HC, how="left")
    comp["hc"] = comp["hc"].fillna(0.0)
    comp = add_diff_cols(comp)
    # Ejecutivo por mes + área + cargo + concepto + tipo hora (sin CECO)
    exec_df = comp.groupby(KEY_EXEC, dropna=False).agg(
        cantidad_pagada=("cantidad_pagada","sum"),
        valor_pagado=("valor_pagado","sum"),
        cantidad_provisionada=("cantidad_provisionada","sum"),
        valor_provisionado=("valor_provisionado","sum"),
        cantidad_proyectada=("cantidad_proyectada","sum"),
        valor_proyectado=("valor_proyectado","sum"),
    ).reset_index()
    exec_df = exec_df.merge(hc_use, on=KEY_HC, how="left")
    exec_df["hc"] = exec_df["hc"].fillna(0.0)
    exec_df = add_diff_cols(exec_df)
    exec_sin_ceros = exec_df[(exec_df["valor_pagado"].abs() + exec_df["valor_provisionado"].abs() + exec_df["valor_proyectado"].abs() + exec_df["cantidad_pagada"].abs() + exec_df["cantidad_provisionada"].abs() + exec_df["cantidad_proyectada"].abs()) > 0].copy()
    cargo_df = exec_df.groupby(KEY_HC, dropna=False).agg(
        cantidad_pagada=("cantidad_pagada","sum"),
        valor_pagado=("valor_pagado","sum"),
        cantidad_provisionada=("cantidad_provisionada","sum"),
        valor_provisionado=("valor_provisionado","sum"),
        cantidad_proyectada=("cantidad_proyectada","sum"),
        valor_proyectado=("valor_proyectado","sum"),
        hc=("hc", "max"),
    ).reset_index()
    cargo_df = add_diff_cols(cargo_df)
    indicadores = cargo_df.copy()
    indicadores["horas_pagadas_por_hc"] = np.where(indicadores["hc"] > 0, indicadores["cantidad_pagada"] / indicadores["hc"], 0)
    indicadores["horas_provisionadas_por_hc"] = np.where(indicadores["hc"] > 0, indicadores["cantidad_provisionada"] / indicadores["hc"], 0)
    indicadores["horas_proyectadas_por_hc"] = np.where(indicadores["hc"] > 0, indicadores["cantidad_proyectada"] / indicadores["hc"], 0)
    indicadores["valor_pagado_por_hc"] = np.where(indicadores["hc"] > 0, indicadores["valor_pagado"] / indicadores["hc"], 0)
    indicadores["valor_provisionado_por_hc"] = np.where(indicadores["hc"] > 0, indicadores["valor_provisionado"] / indicadores["hc"], 0)
    indicadores["valor_proyectado_por_hc"] = np.where(indicadores["hc"] > 0, indicadores["valor_proyectado"] / indicadores["hc"], 0)
    resumen_mes = exec_df.groupby("periodo_novedad", dropna=False).agg(
        cantidad_pagada=("cantidad_pagada","sum"),
        valor_pagado=("valor_pagado","sum"),
        cantidad_provisionada=("cantidad_provisionada","sum"),
        valor_provisionado=("valor_provisionado","sum"),
        cantidad_proyectada=("cantidad_proyectada","sum"),
        valor_proyectado=("valor_proyectado","sum"),
    ).reset_index()
    resumen_mes = add_diff_cols(resumen_mes)
    for df in [comp, exec_df, exec_sin_ceros, cargo_df, indicadores, resumen_mes]:
        if "periodo_novedad" in df.columns:
            df["periodo_orden"] = df["periodo_novedad"].apply(period_sort_key)
            df.sort_values(["periodo_orden"] + [c for c in ["area_negocio","cargo_homologado","concepto","tipo_hora"] if c in df.columns], inplace=True)
            df.drop(columns=["periodo_orden"], inplace=True)
    return {
        "Detalle_Comparativo": comp,
        "Resumen_Ejecutivo": exec_df,
        "Resumen_Ejecutivo_Sin_Ceros": exec_sin_ceros,
        "Resumen_Cargo_Homologado": cargo_df,
        "Indicadores_HC": indicadores,
        "Resumen_Mes": resumen_mes,
    }


def add_diff_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["valor_pagado","valor_provisionado","valor_proyectado","cantidad_pagada","cantidad_provisionada","cantidad_proyectada"]:
        if c not in df.columns:
            df[c] = 0.0
    df["dif_pagado_vs_provision"] = df["valor_pagado"] - df["valor_provisionado"]
    df["dif_pagado_vs_proyeccion"] = df["valor_pagado"] - df["valor_proyectado"]
    df["dif_cant_pagada_vs_provision"] = df["cantidad_pagada"] - df["cantidad_provisionada"]
    df["dif_cant_pagada_vs_proyeccion"] = df["cantidad_pagada"] - df["cantidad_proyectada"]
    df["desv_pct_vs_provision"] = np.where(df["valor_provisionado"].abs() > 0, df["dif_pagado_vs_provision"] / df["valor_provisionado"].abs() * 100, np.where(df["valor_pagado"].abs() > 0, 100.0, 0.0))
    df["desv_pct_vs_proyeccion"] = np.where(df["valor_proyectado"].abs() > 0, df["dif_pagado_vs_proyeccion"] / df["valor_proyectado"].abs() * 100, np.where(df["valor_pagado"].abs() > 0, 100.0, 0.0))
    return df


def build_alerts(report: Dict[str, pd.DataFrame], alerts: List[Dict[str, Any]], umbral: float = 15.0) -> pd.DataFrame:
    out = list(alerts)
    detalle = report.get("Detalle_Comparativo", pd.DataFrame())
    if not detalle.empty:
        p_sin_pr = detalle[(detalle["valor_pagado"].abs() > 0) & (detalle["valor_provisionado"].abs() == 0)]
        p_sin_py = detalle[(detalle["valor_pagado"].abs() > 0) & (detalle["valor_proyectado"].abs() == 0)]
        pr_sin_p = detalle[(detalle["valor_provisionado"].abs() > 0) & (detalle["valor_pagado"].abs() == 0)]
        py_sin_p = detalle[(detalle["valor_proyectado"].abs() > 0) & (detalle["valor_pagado"].abs() == 0)]
        hc_cero = detalle[(detalle["hc"].fillna(0) == 0) & ((detalle["valor_pagado"].abs()+detalle["valor_provisionado"].abs()+detalle["valor_proyectado"].abs()) > 0)]
        desv = detalle[(detalle["valor_pagado"].abs() > 0) & ((detalle["desv_pct_vs_provision"].abs() > umbral) | (detalle["desv_pct_vs_proyeccion"].abs() > umbral))]
        out += [
            {"tipo":"Cruce", "mensaje":f"Pagado sin provisión: {len(p_sin_pr):,} combinaciones"},
            {"tipo":"Cruce", "mensaje":f"Pagado sin proyección: {len(p_sin_py):,} combinaciones"},
            {"tipo":"Cruce", "mensaje":f"Provisión sin pagado: {len(pr_sin_p):,} combinaciones"},
            {"tipo":"Cruce", "mensaje":f"Proyección sin pagado: {len(py_sin_p):,} combinaciones"},
            {"tipo":"Headcount", "mensaje":f"Combinaciones con movimiento y HC en cero: {len(hc_cero):,}"},
            {"tipo":"Desviación", "mensaje":f"Combinaciones que superan {umbral:.0f}% de desviación: {len(desv):,}"},
        ]
    return pd.DataFrame(out)

# ==============================
# Presentación y Excel
# ==============================
FRIENDLY = {
    "periodo_novedad": "Mes novedad",
    "periodo_pago": "Mes pago",
    "area_negocio": "Área negocio",
    "cargo_homologado": "Cargo homologado",
    "ceco": "CECO",
    "concepto": "Concepto",
    "tipo_hora": "Tipo hora",
    "cantidad_pagada": "Cantidad pagada",
    "valor_pagado": "Pagado",
    "cantidad_provisionada": "Cantidad provisión",
    "valor_provisionado": "Provisión",
    "cantidad_proyectada": "Cantidad proyección",
    "valor_proyectado": "Proyección",
    "dif_pagado_vs_provision": "Dif. pagado vs provisión",
    "dif_pagado_vs_proyeccion": "Dif. pagado vs proyección",
    "dif_cant_pagada_vs_provision": "Dif. cant. vs provisión",
    "dif_cant_pagada_vs_proyeccion": "Dif. cant. vs proyección",
    "desv_pct_vs_provision": "% desv. vs provisión",
    "desv_pct_vs_proyeccion": "% desv. vs proyección",
    "hc": "HC",
    "horas_pagadas_por_hc": "Horas pagadas por HC",
    "horas_provisionadas_por_hc": "Horas provisión por HC",
    "horas_proyectadas_por_hc": "Horas proyección por HC",
    "valor_pagado_por_hc": "Pagado por HC",
    "valor_provisionado_por_hc": "Provisión por HC",
    "valor_proyectado_por_hc": "Proyección por HC",
}


def pretty_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # ordenar columnas importantes
    preferred = [c for c in ["periodo_novedad","area_negocio","cargo_homologado","ceco","concepto","tipo_hora","hc","cantidad_pagada","valor_pagado","cantidad_provisionada","valor_provisionado","cantidad_proyectada","valor_proyectado","dif_pagado_vs_provision","dif_pagado_vs_proyeccion","desv_pct_vs_provision","desv_pct_vs_proyeccion"] if c in out.columns]
    rest = [c for c in out.columns if c not in preferred]
    out = out[preferred + rest]
    for c in out.columns:
        lc = c.lower()
        if any(token in lc for token in ["valor", "pagado", "provision", "proyectado", "dif_pagado", "costo"]):
            # no formatear cantidades que tengan pagada/proyectada en nombre
            if "cantidad" not in lc and "desv" not in lc and "pct" not in lc and "horas" not in lc:
                if pd.api.types.is_numeric_dtype(out[c]):
                    out[c] = out[c].apply(format_money)
        if "cantidad" in lc or "horas" in lc:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].apply(format_qty)
        if c == "hc" or lc == "hc":
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].apply(format_int)
        if "pct" in lc or "desv_pct" in lc:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].apply(format_pct)
    out = out.rename(columns=FRIENDLY)
    return out


def display_df(df: pd.DataFrame, name: str = "tabla", max_rows: int = MAX_SCREEN_ROWS):
    if df is None or df.empty:
        st.info("No hay datos para mostrar.")
        return
    total = len(df)
    show = df.head(max_rows).copy()
    if total > max_rows:
        st.warning(f"Se muestran las primeras {max_rows:,} filas de {total:,}. Descarga el Excel para revisar el detalle completo.")
    height = min(650, max(240, 38 * (len(show) + 1)))
    st.dataframe(pretty_df(show), use_container_width=True, height=int(height))


def filter_df(df: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col, vals in filters.items():
        if vals and col in out.columns:
            out = out[out[col].astype(str).isin(vals)]
    return out


def make_excel(report: Dict[str, pd.DataFrame], extras: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book
        fmt_header = wb.add_format({"bold": True, "bg_color": ORANGE, "font_color": "white", "border": 1})
        fmt_money = wb.add_format({"num_format": '$ #,##0', "border": 1})
        fmt_qty = wb.add_format({"num_format": '#,##0.00', "border": 1})
        fmt_int = wb.add_format({"num_format": '#,##0', "border": 1})
        fmt_pct = wb.add_format({"num_format": '0.00%', "border": 1})
        fmt_text = wb.add_format({"border": 1})
        for sheet, df in {**report, **extras}.items():
            if df is None or df.empty:
                df = pd.DataFrame({"Mensaje": ["Sin datos"]})
            sheet_name = re.sub(r"[\[\]\:\*\?\/\\]", "_", sheet)[:31]
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            ws = writer.sheets[sheet_name]
            for i, col in enumerate(df.columns):
                ws.write(0, i, col, fmt_header)
                width = min(max(12, len(str(col)) + 2), 34)
                ws.set_column(i, i, width, fmt_text)
                lc = str(col).lower()
                if any(t in lc for t in ["valor", "pagado", "provision", "proyectado", "dif_pagado", "costo"]):
                    if "cantidad" not in lc and "pct" not in lc and "desv" not in lc and "horas" not in lc:
                        ws.set_column(i, i, max(width, 16), fmt_money)
                if "cantidad" in lc or "horas" in lc:
                    ws.set_column(i, i, max(width, 14), fmt_qty)
                if lc == "hc":
                    ws.set_column(i, i, max(width, 10), fmt_int)
                if "pct" in lc or "desv_pct" in lc:
                    ws.set_column(i, i, max(width, 12), fmt_pct)
            ws.autofilter(0, 0, max(len(df), 1), max(len(df.columns)-1, 0))
            ws.freeze_panes(1, 0)
    return output.getvalue()

# ==============================
# Procesamiento orquestador
# ==============================
def process_all(inputs: Dict[str, List[Any]], umbral: float = 15.0) -> Dict[str, Any]:
    progress = st.progress(0, text="Iniciando procesamiento...")
    alerts = []
    progress.progress(5, text="Leyendo Detalle Horas y homologación final...")
    detalle_file = inputs.get("detalle", [None])[0] if inputs.get("detalle") else None
    concept_map, func_map, detalle_audit = read_detalle_horas(detalle_file)

    progress.progress(15, text="Procesando Headcount...")
    hc_full, hc_group, hc_excl, a = process_headcount(inputs.get("headcount", []), concept_map, func_map)
    alerts.extend(a)

    progress.progress(25, text="Leyendo posiciones homologadas opcionales...")
    poshom_df = None
    if inputs.get("posiciones"):
        try:
            try:
                poshom_df = read_excel_upload(inputs["posiciones"][0], sheet_name="HEAD_COUNT")
            except Exception:
                poshom_df = read_any_upload(inputs["posiciones"][0])
            alerts.append({"tipo":"Cargue", "mensaje":f"Posiciones homologadas: {len(poshom_df):,} registros leídos"})
        except Exception as e:
            alerts.append({"tipo":"Error", "mensaje":f"Error leyendo Posiciones homologadas: {e}"})

    progress.progress(35, text="Construyendo Maestro Posición → Función...")
    master, master_audit = build_position_function_master(hc_full, poshom_df)
    alerts.append({"tipo":"Homologación", "mensaje":f"Maestro Posición → Función construido con {len(master):,} llaves únicas"})

    progress.progress(45, text="Procesando pagado real CCNómina + compensatorios...")
    pagado_full, pagado_agg, a = process_pagado(inputs.get("ccnomina", []), inputs.get("compensatorios", []), concept_map, func_map, master, hc_full)
    alerts.extend(a)

    progress.progress(60, text="Procesando provisión...")
    provision_full, provision_agg, a = process_provision(inputs.get("provision", []), concept_map, func_map, master, hc_full)
    alerts.extend(a)

    progress.progress(75, text="Procesando proyección...")
    proyeccion_full, proyeccion_agg, a = process_proyeccion(inputs.get("proyeccion", []), concept_map, func_map, master, hc_full)
    alerts.extend(a)

    progress.progress(88, text="Generando comparativos y resúmenes...")
    report = aggregate_sources(pagado_agg, provision_agg, proyeccion_agg, hc_group)
    alertas_df = build_alerts(report, alerts, umbral=umbral)

    progress.progress(98, text="Preparando auditorías...")
    pendientes = pd.concat([
        pagado_full.assign(origen_base="Pagado") if pagado_full is not None and not pagado_full.empty else pd.DataFrame(),
        provision_full.assign(origen_base="Provisión") if provision_full is not None and not provision_full.empty else pd.DataFrame(),
        proyeccion_full.assign(origen_base="Proyección") if proyeccion_full is not None and not proyeccion_full.empty else pd.DataFrame(),
    ], ignore_index=True)
    if not pendientes.empty:
        pendientes = pendientes[pendientes["cargo_homologado"].isin(["", "Sin homologar"])][[c for c in ["origen_base","archivo","periodo_novedad","sap","posicion_original","funcion_codigo_final","funcion_nombre_final","area_negocio","concepto","valor_pagado","valor_provisionado","valor_proyectado"] if c in pendientes.columns]].copy()
    extras = {
        "Alertas": alertas_df,
        "Maestro_Posicion_Funcion": master_audit,
        "Homologacion_Detalle_Horas": detalle_audit,
        "Headcount_Usado": hc_group,
        "Headcount_Excluido": hc_excl,
        "Pendientes_Homologacion": pendientes,
    }
    progress.progress(100, text="Listo.")
    return {
        "report": report,
        "extras": extras,
        "raw": {
            "pagado": pagado_full, "provision": provision_full, "proyeccion": proyeccion_full,
            "headcount": hc_full,
        },
        "metrics": {
            "pagado_registros": len(pagado_full) if pagado_full is not None else 0,
            "provision_registros": len(provision_full) if provision_full is not None else 0,
            "proyeccion_registros": len(proyeccion_full) if proyeccion_full is not None else 0,
            "hc_registros": len(hc_full) if hc_full is not None else 0,
            "maestro_llaves": len(master),
        }
    }

# ==============================
# UI
# ==============================
if "resultado_v10" not in st.session_state:
    st.session_state["resultado_v10"] = None

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Ara_macao_-Costa_Rica-8.jpg/320px-Ara_macao_-Costa_Rica-8.jpg", width=110)
    st.markdown("### Menú")
    page = st.radio("Ir a", ["1. Cargue y procesamiento", "2. Comparativo histórico", "3. Diagnóstico y alertas", "4. Instructivo"], label_visibility="collapsed")
    st.divider()
    umbral = st.number_input("Umbral desviación alerta (%)", min_value=1.0, max_value=100.0, value=15.0, step=1.0)

if page == "1. Cargue y procesamiento":
    st.subheader("1. Cargue de archivos")
    st.markdown("Carga cada bloque por separado. El comparativo se calcula por **mes de novedad**; CCNómina y compensatorios se corren un mes atrás por pago mes vencido.")
    c1, c2 = st.columns(2)
    with c1:
        detalle = st.file_uploader("Detalle Horas / Homologación (obligatorio)", type=["xlsb", "xlsx", "xlsm"], accept_multiple_files=False)
        posiciones = st.file_uploader("Posiciones homologadas (opcional)", type=["xlsx", "xlsm"], accept_multiple_files=False)
        ccnomina = st.file_uploader("CCNóminas - Pagado real (cargue múltiple)", type=["xlsx", "xlsm", "xls", "txt", "csv"], accept_multiple_files=True)
        compensatorios = st.file_uploader("Compensatorios - Pagado Y350 (cargue múltiple)", type=["xls", "xlsx", "txt", "csv"], accept_multiple_files=True)
    with c2:
        headcount = st.file_uploader("Headcount mensual (cargue múltiple)", type=["xlsx", "xlsm", "xls"], accept_multiple_files=True)
        provision = st.file_uploader("Consolidado Provisión", type=["xlsx", "xlsm"], accept_multiple_files=True)
        proyeccion = st.file_uploader("Consolidado Proyección", type=["xlsx", "xlsm"], accept_multiple_files=True)
    st.info("Tip: Posiciones homologadas es opcional. Si no se carga, la app construye el maestro Posición → Función desde los Headcount cargados.")
    if st.button("🚀 Procesar archivos", type="primary", use_container_width=True):
        if not detalle:
            st.error("Debes cargar Detalle Horas / Homologación.")
        elif not headcount:
            st.error("Debes cargar al menos un Headcount para construir el maestro Posición → Función y calcular HC.")
        else:
            inputs = {
                "detalle": [detalle] if detalle else [],
                "posiciones": [posiciones] if posiciones else [],
                "ccnomina": ccnomina or [],
                "compensatorios": compensatorios or [],
                "headcount": headcount or [],
                "provision": provision or [],
                "proyeccion": proyeccion or [],
            }
            try:
                st.session_state["resultado_v10"] = process_all(inputs, umbral=umbral)
                st.success("Procesamiento finalizado. Ve a Comparativo histórico para revisar resultados.")
            except Exception as e:
                st.exception(e)
    if st.session_state["resultado_v10"]:
        m = st.session_state["resultado_v10"]["metrics"]
        st.subheader("Control de registros procesados")
        cols = st.columns(5)
        cols[0].metric("Pagado", format_int(m.get("pagado_registros",0)))
        cols[1].metric("Provisión", format_int(m.get("provision_registros",0)))
        cols[2].metric("Proyección", format_int(m.get("proyeccion_registros",0)))
        cols[3].metric("Headcount", format_int(m.get("hc_registros",0)))
        cols[4].metric("Maestro", format_int(m.get("maestro_llaves",0)))

elif page == "2. Comparativo histórico":
    st.subheader("2. Comparativo histórico")
    res = st.session_state.get("resultado_v10")
    if not res:
        st.warning("Primero procesa los archivos en la pantalla de cargue.")
    else:
        report = res["report"]
        resumen_mes = report["Resumen_Mes"]
        total_pagado = resumen_mes["valor_pagado"].sum() if not resumen_mes.empty else 0
        total_prov = resumen_mes["valor_provisionado"].sum() if not resumen_mes.empty else 0
        total_proy = resumen_mes["valor_proyectado"].sum() if not resumen_mes.empty else 0
        cols = st.columns(4)
        cols[0].metric("Pagado", format_money(total_pagado))
        cols[1].metric("Provisión", format_money(total_prov))
        cols[2].metric("Proyección", format_money(total_proy))
        cols[3].metric("Dif. pagado vs provisión", format_money(total_pagado-total_prov))
        base = report["Resumen_Ejecutivo_Sin_Ceros"].copy()
        with st.form("filtros_comparativo"):
            st.markdown("#### Filtros — vacío = todos")
            fcols = st.columns(5)
            meses = sorted(base["periodo_novedad"].dropna().astype(str).unique(), key=period_sort_key) if "periodo_novedad" in base.columns else []
            areas = sorted(base["area_negocio"].dropna().astype(str).unique()) if "area_negocio" in base.columns else []
            cargos = sorted(base["cargo_homologado"].dropna().astype(str).unique()) if "cargo_homologado" in base.columns else []
            conceptos = sorted(base["concepto"].dropna().astype(str).unique()) if "concepto" in base.columns else []
            tipos = sorted(base["tipo_hora"].dropna().astype(str).unique()) if "tipo_hora" in base.columns else []
            sel_meses = fcols[0].multiselect("Mes novedad", meses, default=[])
            sel_areas = fcols[1].multiselect("Área negocio", areas, default=[])
            sel_cargos = fcols[2].multiselect("Cargo homologado", cargos, default=[])
            sel_conceptos = fcols[3].multiselect("Concepto", conceptos, default=[])
            sel_tipos = fcols[4].multiselect("Tipo hora", tipos, default=[])
            submitted = st.form_submit_button("Aplicar filtros", type="primary")
        filters = {"periodo_novedad": sel_meses, "area_negocio": sel_areas, "cargo_homologado": sel_cargos, "concepto": sel_conceptos, "tipo_hora": sel_tipos}
        filtered_exec = filter_df(report["Resumen_Ejecutivo_Sin_Ceros"], filters)
        st.markdown("### Resumen Ejecutivo sin ceros")
        display_df(filtered_exec)
        st.markdown("### Resumen por mes")
        display_df(filter_df(report["Resumen_Mes"], {"periodo_novedad": sel_meses}))
        st.markdown("### Resumen por cargo homologado")
        display_df(filter_df(report["Resumen_Cargo_Homologado"], {"periodo_novedad": sel_meses, "area_negocio": sel_areas, "cargo_homologado": sel_cargos}))
        excel_bytes = make_excel(report, res["extras"])
        st.download_button("📥 Descargar Excel completo", excel_bytes, file_name=f"comparativo_horas_nomina_{datetime.now():%Y%m%d_%H%M}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

elif page == "3. Diagnóstico y alertas":
    st.subheader("3. Diagnóstico y alertas")
    res = st.session_state.get("resultado_v10")
    if not res:
        st.warning("Primero procesa los archivos en la pantalla de cargue.")
    else:
        tabs = st.tabs(["Alertas", "Maestro Posición → Función", "Pendientes homologación", "Headcount usado", "Detalle auditoría"])
        with tabs[0]:
            display_df(res["extras"].get("Alertas", pd.DataFrame()), max_rows=2000)
        with tabs[1]:
            display_df(res["extras"].get("Maestro_Posicion_Funcion", pd.DataFrame()), max_rows=5000)
        with tabs[2]:
            display_df(res["extras"].get("Pendientes_Homologacion", pd.DataFrame()), max_rows=5000)
        with tabs[3]:
            display_df(res["extras"].get("Headcount_Usado", pd.DataFrame()), max_rows=5000)
        with tabs[4]:
            st.markdown("#### Muestra detalle comparativo")
            display_df(res["report"].get("Detalle_Comparativo", pd.DataFrame()), max_rows=5000)

elif page == "4. Instructivo":
    st.subheader("4. Instructivo de uso y lectura")
    st.markdown(
        """
### Objetivo
La herramienta compara **Pagado real vs Provisión vs Proyección** por mes, área de negocio, cargo homologado, concepto y tipo de hora.

### Regla principal: pago mes vencido
Los archivos **CCNómina** y **Compensatorios** representan el mes de pago. Para comparar, la app los corre un mes atrás:

| Mes pago | Mes novedad |
|---|---|
| 02.2026 | 01.2026 |
| 03.2026 | 02.2026 |
| 04.2026 | 03.2026 |
| 05.2026 | 04.2026 |

### Homologación correcta
La app no compara directamente por cargo textual del reporte. Primero construye un maestro:

**Posición → Función → Cargo homologado**

1. Si se carga `Posiciones_homologadas`, se usa como maestro principal.
2. Si no se carga, se construye desde los Headcount.
3. Luego cada base se lleva a función y finalmente a cargo homologado con `Detalle Horas`.

### Headcount
El HC se calcula por:

**Mes + Área negocio + Cargo homologado**

Se excluyen Manager I, II, III y IV porque no aplican para horas/tiempo suplementario.

### Hojas del Excel
- `Detalle_Comparativo`: granularidad máxima por CECO.
- `Resumen_Ejecutivo`: Mes + Área + Cargo + Concepto + Tipo hora.
- `Resumen_Ejecutivo_Sin_Ceros`: igual al ejecutivo, pero sin filas vacías.
- `Resumen_Cargo_Homologado`: análisis por cargo.
- `Indicadores_HC`: horas y valores por HC.
- `Alertas`: calidad de cruces y datos.
- `Maestro_Posicion_Funcion`: auditoría del maestro construido.
- `Pendientes_Homologacion`: nombres que no se pudieron homologar.
        """
    )
