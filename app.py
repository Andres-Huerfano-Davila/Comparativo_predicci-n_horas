import io
import re
import math
import calendar
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import holidays
except Exception:
    holidays = None

# =============================================================
# CONFIGURACIÓN GENERAL
# =============================================================
st.set_page_config(
    page_title="Comparativo y predicción de horas nómina",
    page_icon="🦜",
    layout="wide",
)

CONCEPTOS = ["Y220", "Y221", "Y300", "Y305", "Y310", "Y315", "Y350", "YM01"]

# Conceptos que componen el Salario Total para calcular valor hora desde MD SAP.
# Se pueden ampliar si JMC incluye nuevos conceptos salariales en el query.
CONCEPTOS_SALARIO_MD = ["Y010", "Y011", "Y020", "Y050", "Y051", "Y090", "Y506", "Y610", "Y617", "Y618"]

TIPO_HORA_DEFAULT = {
    "Y220": "Rec. Noc.",
    "Y221": "Rec. Dom noc",
    "Y300": "Hora Extra",
    "Y305": "Hora Extra",
    "Y310": "Hora Extra",
    "Y315": "Hora Extra",
    "Y350": "Compensatorio",
    "YM01": "Rec. Dom",
}

INTERFAZ_A_CCNOMINA = {
    "Y540": "Y220",
    "Y541": "Y221",
    "Y542": "Y300",
    "Y543": "Y305",
    "Y544": "Y310",
    "Y545": "Y315",
    "Y546": "Y350",
    "Y547": "YM01",
}

FACTORES_DEFAULT = pd.DataFrame(
    [
        {"concepto": "Y220", "tipo_hora": "Rec. Noc.", "factor": 0.35},
        {"concepto": "Y221", "tipo_hora": "Rec. Dom noc", "factor": 1.10},
        {"concepto": "Y300", "tipo_hora": "Hora Extra", "factor": 1.25},
        {"concepto": "Y305", "tipo_hora": "Hora Extra", "factor": 1.75},
        {"concepto": "Y310", "tipo_hora": "Hora Extra", "factor": 2.00},
        {"concepto": "Y315", "tipo_hora": "Hora Extra", "factor": 2.50},
        {"concepto": "Y350", "tipo_hora": "Compensatorio", "factor": 1.00},
        {"concepto": "YM01", "tipo_hora": "Rec. Dom", "factor": 0.75},
    ]
)

CUENTAS_TEMPLATE = pd.DataFrame(
    [
        {"concepto": c, "area_negocio": a, "prefijo_ceco": p, "cuenta": "", "descripcion_cuenta": ""}
        for c in CONCEPTOS
        for a, p in [("Tiendas", "101"), ("CEDI", "102"), ("Oficina soporte", "103"), ("BDC", "101")]
    ]
)

KEY_COMPARATIVO = ["periodo_novedad", "concepto", "tipo_hora", "cargo_homologado", "area_negocio", "ceco"]
KEY_HC = ["periodo_novedad", "cargo_homologado", "area_negocio"]
# El HC se cruza a nivel cargo homologado + área + mes.
# No se cruza por CECO para evitar ceros cuando el mismo cargo/función aparece con CECO distinto
# entre pago, provisión, proyección y headcount.

# =============================================================
# UTILIDADES
# =============================================================
def quitar_tildes(txt: str) -> str:
    if txt is None:
        return ""
    rep = str.maketrans("áéíóúÁÉÍÓÚñÑüÜ", "aeiouAEIOUnNuU")
    return str(txt).translate(rep)


def norm_col(x) -> str:
    x = quitar_tildes(str(x)).strip().upper()
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[^A-Z0-9]+", "", x)
    return x


def norm_text(x) -> str:
    x = quitar_tildes(str(x) if x is not None else "").strip().upper()
    x = re.sub(r"\s+", " ", x)
    return x


def clean_code(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def clean_ceco(x) -> str:
    s = clean_code(x)
    s = re.sub(r"\D", "", s)
    return s


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = False) -> Optional[str]:
    if df is None or len(getattr(df, "columns", [])) == 0:
        if required:
            raise ValueError(f"DataFrame vacío o sin columnas. No encontré columna para: {candidates}")
        return None
    norm_to_orig = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        n = norm_col(cand)
        if n in norm_to_orig:
            return norm_to_orig[n]
    # Búsqueda por contenido normalizado, preferimos columnas cortas para evitar que CC-n. encuentre texto explicativo.
    for cand in candidates:
        n = norm_col(cand)
        possibles = []
        for c in df.columns:
            nc = norm_col(c)
            if n and (n in nc or nc in n):
                possibles.append(c)
        if possibles:
            possibles = sorted(possibles, key=lambda c: len(str(c)))
            return possibles[0]
    if required:
        raise ValueError(f"No encontré columna para: {candidates}. Columnas disponibles: {list(df.columns)[:30]}")
    return None


def to_num(s) -> pd.Series:
    if isinstance(s, pd.Series):
        ser = s.copy()
    else:
        ser = pd.Series(s)
    if pd.api.types.is_numeric_dtype(ser):
        return pd.to_numeric(ser, errors="coerce").fillna(0.0)
    def parse_one(v):
        if pd.isna(v):
            return 0.0
        txt = str(v).strip().replace("$", "").replace("COP", "").replace(" ", "")
        if txt == "" or txt == "-":
            return 0.0
        # SAP en texto suele traer miles con punto y decimal con coma: 111.426 o 7,00
        if "," in txt:
            txt = txt.replace(".", "").replace(",", ".")
        else:
            # Si solo hay puntos y parecen miles, se remueven.
            parts = txt.split(".")
            if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
                txt = txt.replace(".", "")
        try:
            return float(txt)
        except Exception:
            return 0.0
    return ser.map(parse_one).astype(float)


def period_from_any(value, fallback_name: str = "") -> Optional[str]:
    """Retorna periodo en formato MM.AAAA."""
    text = ""
    if value is not None and not pd.isna(value):
        text = str(value)
    if not text and fallback_name:
        text = fallback_name
    text = text.strip()

    # 01.2026 / 01-2026 / 01_2026
    m = re.search(r"(?<!\d)(0?[1-9]|1[0-2])[\.\-_ /](20\d{2})(?!\d)", text)
    if m:
        return f"{int(m.group(1)):02d}.{m.group(2)}"

    # 202601 o 012026. Se evalúa antes de pd.to_datetime para no volver lento el procesamiento fila a fila.
    s = re.sub(r"\.0$", "", text)
    if re.fullmatch(r"20\d{2}(0[1-9]|1[0-2])", s):
        return f"{s[4:6]}.{s[0:4]}"
    if re.fullmatch(r"(0[1-9]|1[0-2])20\d{2}", s):
        return f"{s[0:2]}.{s[2:6]}"
    m = re.search(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(?!\d)", text)
    if m:
        return f"{m.group(2)}.{m.group(1)}"
    m = re.search(r"(?<!\d)(0[1-9]|1[0-2])(20\d{2})(?!\d)", text)
    if m:
        return f"{m.group(1)}.{m.group(2)}"

    # Fecha dd.mm.yyyy o yyyy-mm-dd
    try:
        dt = pd.to_datetime(text, dayfirst=True, errors="coerce")
        if pd.notna(dt):
            return f"{int(dt.month):02d}.{int(dt.year)}"
    except Exception:
        pass
    return None


def period_sort_key(p: str) -> int:
    if not p or not isinstance(p, str) or "." not in p:
        return 0
    mm, yy = p.split(".")[:2]
    return int(yy) * 100 + int(mm)



def fill_period_series(series: pd.Series, fallback: Optional[str]) -> pd.Series:
    if fallback:
        return series.fillna(fallback)
    return series

def shift_period(periodo: str, months: int) -> Optional[str]:
    if not periodo or "." not in str(periodo):
        return None
    mm, yy = map(int, periodo.split(".")[:2])
    total = yy * 12 + (mm - 1) + months
    new_y = total // 12
    new_m = total % 12 + 1
    return f"{new_m:02d}.{new_y}"


def calendar_suggestion(periodo: str) -> Dict[str, int]:
    mm, yy = map(int, periodo.split(".")[:2])
    total = calendar.monthrange(yy, mm)[1]
    sundays = sum(1 for d in range(1, total + 1) if date(yy, mm, d).weekday() == 6)
    festivos = 0
    if holidays is not None:
        try:
            co_holidays = holidays.country_holidays("CO", years=[yy])
            festivos = sum(1 for d in co_holidays if d.year == yy and d.month == mm and d.weekday() != 6)
        except Exception:
            festivos = 0
    return {"dias_mes": total, "domingos": sundays, "festivos": festivos}


def driver_days(concepto: str, dias_mes: float, domingos: float, festivos: float) -> float:
    concepto = str(concepto).upper()
    if concepto in ["YM01", "Y221", "Y310", "Y315"]:
        return max(float(domingos or 0) + float(festivos or 0), 1.0)
    return max(float(dias_mes or 0), 1.0)


def read_uploaded_bytes(uploaded_file) -> bytes:
    if uploaded_file is None:
        return b""
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    with open(uploaded_file, "rb") as f:
        return f.read()


def uploaded_name(uploaded_file) -> str:
    return getattr(uploaded_file, "name", str(uploaded_file))



def read_xlsx_fast(raw: bytes, usecols=None, sheet_name=None, header=0, nrows=None) -> pd.DataFrame:
    """Lector rápido para .xlsx cuando solo se necesitan algunas columnas."""
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    ws = wb[sheet_name] if sheet_name and sheet_name in wb.sheetnames else wb[wb.sheetnames[0]]
    header_row_num = (header or 0) + 1
    header_values = [cell for cell in next(ws.iter_rows(min_row=header_row_num, max_row=header_row_num, values_only=True))]
    headers = [str(h).strip() if h is not None else f"Unnamed_{i}" for i, h in enumerate(header_values)]
    if nrows == 0:
        return pd.DataFrame(columns=headers)
    if usecols is not None:
        wanted = {str(c).strip() for c in usecols}
        idxs = [i for i, h in enumerate(headers) if str(h).strip() in wanted]
    else:
        idxs = list(range(len(headers)))
    selected_headers = [headers[i] for i in idxs]
    data = []
    max_row = None if nrows is None else header_row_num + int(nrows)
    for row in ws.iter_rows(min_row=header_row_num + 1, max_row=max_row, values_only=True):
        data.append([row[i] if i < len(row) else None for i in idxs])
    wb.close()
    return pd.DataFrame(data, columns=selected_headers)

def read_sap_utf16_report(raw: bytes) -> pd.DataFrame:
    txt = raw.decode("utf-16", errors="replace")
    lines = txt.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:200]):
        if ("Nº pers." in line or "N° pers." in line or "No pers" in line) and "CC-n" in line:
            header_idx = i
            break
    if header_idx is None:
        # Para otros planos tipo interface sin encabezado.
        header_idx = 0
    content = "\n".join([l for l in lines[header_idx:] if l.strip() != ""])
    df = pd.read_csv(io.StringIO(content), sep="\t", dtype=str, low_memory=False)
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def make_unique_columns(cols: List[str]) -> List[str]:
    counts = {}
    out = []
    for c in cols:
        base = str(c).strip()
        if base in counts:
            counts[base] += 1
            out.append(f"{base}.{counts[base]}")
        else:
            counts[base] = 0
            out.append(base)
    return out


def read_sap_pipe_report(raw: bytes) -> pd.DataFrame:
    """Lee reportes SAP TXT con líneas separadas por |, como Master Data con varios renglones por empleado."""
    chosen_lines = None
    header_idx = None
    for enc in ["latin1", "cp1252", "utf-8-sig", "utf-16"]:
        try:
            txt = raw.decode(enc, errors="replace")
        except Exception:
            continue
        lines = txt.splitlines()
        for i, line in enumerate(lines[:500]):
            line_norm = quitar_tildes(line)
            if line.strip().startswith("|") and ("Nº pers." in line or "N° pers." in line or "No pers" in line or "N pers" in line_norm) and ("CC-nomina" in line_norm or "CC-n" in line_norm):
                chosen_lines = lines
                header_idx = i
                break
        if chosen_lines is not None:
            break
    if chosen_lines is None or header_idx is None:
        raise ValueError("No encontré encabezado SAP tipo |Nº pers.|...|CC-nómina| en el TXT.")
    useful = []
    for line in chosen_lines[header_idx:]:
        s = line.strip()
        if not s.startswith("|"):
            continue
        if set(s) <= set("|-"):
            continue
        useful.append(line)
    df = pd.read_csv(io.StringIO("\n".join(useful)), sep="|", dtype=str, engine="python")
    df = df.dropna(axis=1, how="all")
    df.columns = make_unique_columns([str(c).strip() for c in df.columns])
    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    sap_col = find_col(df, ["Nº pers.", "N° pers.", "SAP", "Número de personal"], required=False)
    if sap_col:
        df = df[df[sap_col].astype(str).str.strip().str.fullmatch(r"\d+")]
    return df

def parse_sap_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str.strip(), format="%d.%m.%Y", errors="coerce")


def read_table(uploaded_file, sheet_name=None, header=0, nrows=None, no_header=False, usecols=None) -> pd.DataFrame:
    name = uploaded_name(uploaded_file)
    ext = name.lower().split(".")[-1]
    raw = read_uploaded_bytes(uploaded_file)
    hdr = None if no_header else header

    calamine_available = False
    try:
        import python_calamine  # noqa: F401
        calamine_available = True
    except Exception:
        calamine_available = False
    if (not calamine_available) and ext in ["xlsx", "xlsm"] and header == 0 and not no_header and (usecols is not None or nrows == 0):
        try:
            return read_xlsx_fast(raw, usecols=usecols, sheet_name=sheet_name, header=header, nrows=nrows)
        except Exception:
            pass

    if ext in ["csv", "txt"]:
        # Reporte SAP TXT tipo |col1|col2|...| (Master Data).
        for enc in ["utf-8-sig", "latin1", "cp1252", "utf-16"]:
            try:
                sample_txt = raw[:50000].decode(enc, errors="ignore")
                if "|" in sample_txt and ("Nº pers." in sample_txt or "N° pers." in sample_txt or "CC-n" in sample_txt):
                    df = read_sap_pipe_report(raw)
                    if usecols is not None:
                        keep = [c for c in usecols if c in df.columns]
                        if keep:
                            df = df[keep]
                    return df.head(nrows) if nrows else df
            except Exception:
                continue
        for enc in ["utf-8-sig", "latin1", "cp1252", "utf-16"]:
            try:
                decoded = raw.decode(enc, errors="ignore")
                sep = ";" if decoded.count(";") > decoded.count("\t") else "\t"
                return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, header=hdr, nrows=nrows, usecols=usecols)
            except Exception:
                continue
        return pd.read_csv(io.BytesIO(raw), header=hdr, nrows=nrows, usecols=usecols)

    if ext == "xls" and (raw[:2] == b"\xff\xfe" or b"\t" in raw[:1000]):
        df = read_sap_utf16_report(raw)
        if usecols is not None:
            keep = [c for c in usecols if c in df.columns]
            if keep:
                df = df[keep]
        if no_header:
            # Si el usuario carga un .xls de interface sin encabezados, intentamos re-leer directo como tabla.
            try:
                txt = raw.decode("utf-16", errors="replace")
                return pd.read_csv(io.StringIO(txt), sep="\t", header=None, nrows=nrows)
            except Exception:
                pass
        return df

    engine = None
    # En Streamlit Cloud recomendamos python-calamine para leer Excel grande mucho más rápido.
    try:
        import python_calamine  # noqa: F401
        if ext in ["xlsx", "xlsm", "xlsb", "xls"]:
            engine = "calamine"
    except Exception:
        engine = None
    if engine is None:
        if ext in ["xlsx", "xlsm"]:
            engine = "openpyxl"
        elif ext == "xlsb":
            engine = "pyxlsb"
        elif ext == "xls":
            engine = "xlrd"
    bio = io.BytesIO(raw)
    excel_usecols = usecols
    if isinstance(usecols, list):
        wanted = {str(c).strip() for c in usecols}
        excel_usecols = lambda c: str(c).strip() in wanted
    if sheet_name is None:
        return pd.read_excel(bio, engine=engine, header=hdr, nrows=nrows, usecols=excel_usecols)
    return pd.read_excel(bio, engine=engine, sheet_name=sheet_name, header=hdr, nrows=nrows, usecols=excel_usecols)


def available_sheets(uploaded_file) -> List[str]:
    name = uploaded_name(uploaded_file)
    ext = name.lower().split(".")[-1]
    engine = "pyxlsb" if ext == "xlsb" else ("openpyxl" if ext in ["xlsx", "xlsm"] else None)
    try:
        raw = read_uploaded_bytes(uploaded_file)
        xl = pd.ExcelFile(io.BytesIO(raw), engine=engine)
        return xl.sheet_names
    except Exception:
        return []


def first_existing_sheet(uploaded_file, preferred: List[str]) -> Optional[str]:
    sheets = available_sheets(uploaded_file)
    if not sheets:
        return None
    normalized = {norm_text(s): s for s in sheets}
    for p in preferred:
        if norm_text(p) in normalized:
            return normalized[norm_text(p)]
    for s in sheets:
        ns = norm_text(s)
        for p in preferred:
            if norm_text(p) in ns:
                return s
    return sheets[0]


def classify_area(ceco=None, tipo=None, div_personal=None, area_nomina=None, centro=None) -> str:
    c = clean_ceco(ceco)
    txt = norm_text(" ".join([str(x) for x in [tipo, div_personal, area_nomina, centro] if x is not None]))
    if "BODEGA CANASTO" in txt or " BDC" in txt or txt.startswith("BDC"):
        return "BDC"
    if c.startswith("102") or "CEDI" in txt or "LOGIST" in txt:
        return "CEDI"
    if c.startswith("103") or "OFICINA" in txt or "ADMINISTR" in txt or "SOPORTE" in txt:
        return "Oficina soporte"
    if c.startswith("101") or "TIENDA" in txt or "ARA STORES" in txt:
        return "Tiendas"
    return "Sin clasificar"


def build_homologacion(file_detalle) -> Tuple[Dict, pd.DataFrame, List[str]]:
    alerts = []
    empty = {
        "cargo_by_funcion": {},
        "cargo_by_text": {},
        "concept_type": TIPO_HORA_DEFAULT.copy(),
        "homologacion_df": pd.DataFrame(),
    }
    if file_detalle is None:
        alerts.append("No se cargó Detalle Horas / Homologación. Se usarán reglas base, pero pueden quedar cargos sin homologar.")
        return empty, pd.DataFrame(), alerts
    try:
        sh = first_existing_sheet(file_detalle, ["Homologación", "Homologacion"])
        df = read_table(file_detalle, sheet_name=sh, nrows=50000)
        df.columns = [str(c).strip() for c in df.columns]
        concept_col = find_col(df, ["CC-n.", "CC-n", "Concepto"], required=False)
        hora_col = find_col(df, ["Hora", "Tipo de hora", "Tipo hora"], required=False)
        concept_type = TIPO_HORA_DEFAULT.copy()
        if concept_col and hora_col:
            aux = df[[concept_col, hora_col]].dropna().drop_duplicates()
            for _, r in aux.iterrows():
                c = clean_code(r[concept_col]).upper()
                h = str(r[hora_col]).strip()
                if c in CONCEPTOS and h:
                    concept_type[c] = h
        concept_type["Y350"] = "Compensatorio"

        func_col = find_col(df, ["Función", "Funcion"], required=False)
        func_text_col = find_col(df, ["Función_4", "Funcion_4", "Denominación función", "Denominacion funcion"], required=False)
        cargo_col = find_col(df, ["Cargo"], required=False)
        cargo_by_funcion, cargo_by_text = {}, {}
        if cargo_col:
            for _, r in df.iterrows():
                cargo = str(r.get(cargo_col, "")).strip()
                if not cargo or cargo.lower() == "nan":
                    continue
                if func_col:
                    k = clean_code(r.get(func_col, ""))
                    if k:
                        cargo_by_funcion[k] = cargo
                if func_text_col:
                    kt = norm_text(r.get(func_text_col, ""))
                    if kt:
                        cargo_by_text[kt] = cargo
                # También mapeamos el propio cargo si aparece como texto.
                cargo_by_text[norm_text(cargo)] = cargo
        if not cargo_by_funcion and not cargo_by_text:
            alerts.append("No logré construir homologación de cargos desde Detalle Horas. Revisa hoja Homologación.")

        hom = {
            "cargo_by_funcion": cargo_by_funcion,
            "cargo_by_text": cargo_by_text,
            "concept_type": concept_type,
            "homologacion_df": df,
        }
        return hom, df, alerts
    except Exception as e:
        alerts.append(f"Error leyendo Detalle Horas / Homologación: {e}")
        return empty, pd.DataFrame(), alerts


def homologar_cargo(funcion=None, cargo_texto=None, hom=None) -> Tuple[str, bool]:
    hom = hom or {}
    k_func = clean_code(funcion)
    if k_func and k_func in hom.get("cargo_by_funcion", {}):
        return hom["cargo_by_funcion"][k_func], True
    t = norm_text(cargo_texto)
    if t and t in hom.get("cargo_by_text", {}):
        return hom["cargo_by_text"][t], True
    # Fallback por patrones. Esto solo evita explosión de cargos cuando la homologación no cruza perfecto.
    if not t or t == "NAN":
        return "Sin cargo", False
    if "APRENDIZ" in t:
        return "Aprendiz", False
    if "PART" in t and "TIME" in t:
        return "Part time", False
    if "JEFE" in t and "TIENDA" in t:
        return "Jefe Tienda", False
    if ("SUPERVISOR JR" in t or "SUPERVISOR JUNIOR" in t or "SUP JR" in t) and "TIENDA" in t:
        return "Supervisor Jr", False
    if "SUPERVISOR" in t and "TIENDA" in t:
        return "Supervisor Tienda", False
    if "OPERADOR" in t and "TIENDA" in t:
        return "Operador Tienda", False
    if "MONTACARGA" in t and "BDC" in t:
        return "Op . Montacarga BDC", False
    if "MONTACARGA" in t:
        return "Montacarga Cedi", False
    if "CEDI" in t or "LOGIST" in t or "BODEGA" in t:
        return "Op. Cedi", False
    if "ASISTENTE" in t or "ANALISTA" in t or "ADMIN" in t or "SOPORTE" in t:
        return "Oficina Soporte", False
    return str(cargo_texto).strip(), False


def tipo_hora(concepto: str, hom=None) -> str:
    c = clean_code(concepto).upper()
    return (hom or {}).get("concept_type", TIPO_HORA_DEFAULT).get(c, TIPO_HORA_DEFAULT.get(c, "Sin tipo hora"))


def is_manager_excluido(row: pd.Series) -> bool:
    texto = norm_text(" ".join([str(v) for v in row.values if pd.notna(v)]))
    # No excluir Non Manager.
    if "NON MANAGER" in texto:
        return False
    patrones = ["MANAGER I", "MANAGER II", "MANAGER III", "MANAGER IV", "MANAGER I-IV", "MANAGER I - IV"]
    return any(p in texto for p in patrones)


def standard_alert_df(alerts: List[str]) -> pd.DataFrame:
    if not alerts:
        return pd.DataFrame(columns=["tipo", "alerta"])
    rows = []
    for a in alerts:
        if isinstance(a, dict):
            rows.append(a)
        else:
            rows.append({"tipo": "Validación", "alerta": str(a)})
    return pd.DataFrame(rows)


def money_fmt(v) -> str:
    try:
        return ("${:,.0f}".format(float(v))).replace(",", ".")
    except Exception:
        return "$0"


def qty_fmt(v) -> str:
    try:
        s = "{:,.2f}".format(float(v))
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "0,00"


FRIENDLY_COLUMNS = {
    "periodo_novedad": "Mes novedad",
    "periodo_pago": "Mes pago",
    "concepto": "Concepto",
    "tipo_hora": "Tipo hora",
    "cargo_homologado": "Cargo homologado",
    "area_negocio": "Área negocio",
    "ceco": "CECO",
    "hc": "HC",
    "valor_pagado": "Valor pagado",
    "valor_provisionado": "Valor provisión",
    "valor_proyectado": "Valor proyección",
    "cantidad_pagada": "Cantidad pagada",
    "cantidad_provisionada": "Cantidad provisión",
    "cantidad_proyectada": "Cantidad proyección",
    "dif_valor_pagado_vs_provision": "Dif. pagado vs provisión",
    "dif_valor_pagado_vs_proyeccion": "Dif. pagado vs proyección",
    "dif_cant_pagada_vs_provision": "Dif. cant. pagada vs provisión",
    "dif_cant_pagada_vs_proyeccion": "Dif. cant. pagada vs proyección",
    "pct_desv_provision": "% desv. provisión",
    "pct_desv_proyeccion": "% desv. proyección",
    "valor_estimado": "Valor estimado",
    "cantidad_estimada": "Cantidad estimada",
    "valor_hora_ref": "Valor hora ref.",
    "salario": "Salario",
    "salario_total": "Salario total",
    "metodo_homologacion": "Método homologación",
    "cargo_original": "Cargo/función original",
    "posicion_original": "Posición original",
    "funcion_nombre": "Función período",
}

def _is_money_col(c: str) -> bool:
    cl = norm_text(c)
    return any(x in cl for x in ["VALOR", "IMPORTE", "PROVISION", "PROYECCION", "PAGADO", "DIF", "COSTO", "SALARIO", "CUENTA"]) and not any(x in cl for x in ["CANT", "CANTIDAD", "PCT", "PORC"] )

def _is_qty_col(c: str) -> bool:
    cl = norm_text(c)
    return any(x in cl for x in ["CANTIDAD", "CANT", "HORAS", "Q", "HC"]) and not _is_money_col(c)

def _is_pct_col(c: str) -> bool:
    cl = norm_text(c)
    return "PCT" in cl or "DESV" in cl or "PORC" in cl or cl.startswith("%")

def pretty_df(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia lista para visualización: pesos con miles, cantidades con 2 decimales y encabezados amigables."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            if _is_pct_col(c):
                out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{float(x)*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))
            elif _is_money_col(c):
                out[c] = out[c].map(money_fmt)
            elif _is_qty_col(c):
                out[c] = out[c].map(qty_fmt)
    out = out.rename(columns={c: FRIENDLY_COLUMNS.get(c, c) for c in out.columns})
    return out

def display_df(df: pd.DataFrame, height: Optional[int] = None):
    # Streamlit no acepta height=None en versiones recientes.
    # Si no se envía altura, dejamos que Streamlit use la altura automática.
    data = pretty_df(df)
    if height is None:
        st.dataframe(data, use_container_width=True)
    else:
        try:
            h = int(height)
            if h <= 0:
                st.dataframe(data, use_container_width=True)
            else:
                st.dataframe(data, use_container_width=True, height=h)
        except Exception:
            st.dataframe(data, use_container_width=True)

def dataframe_config(df: pd.DataFrame) -> Dict:
    # Se mantiene por compatibilidad, pero las tablas principales usan pretty_df/display_df
    return {}

# =============================================================
# LECTURA / NORMALIZACIÓN DE INSUMOS
# =============================================================
def process_ccnomina(files, hom) -> Tuple[pd.DataFrame, List[str]]:
    alerts = []
    rows = []
    if not files:
        return pd.DataFrame(), ["No se cargaron CC-nóminas base plana."]
    for f in files:
        name = uploaded_name(f)
        try:
            header_df = read_table(f, nrows=0)
            header_df.columns = [str(c).strip() for c in header_df.columns]
            col_period = find_col(header_df, ["Per.para", "Periodo", "Periodo pago"], required=False)
            col_concept = find_col(header_df, ["CC-n.", "CC-n", "CC nomina", "Concepto"], required=True)
            col_text = find_col(header_df, ["Texto expl.CC-nómina", "Txt.expl.", "Texto concepto"], required=False)
            col_qty = find_col(header_df, ["Cantidad", "Cant"], required=True)
            col_val = find_col(header_df, ["Importe", "Valor"], required=True)
            col_ceco = find_col(header_df, ["Ce.coste", "CECO", "Centro de coste"], required=False)
            col_func = find_col(header_df, ["Función", "Funcion"], required=False)
            col_cargo = find_col(header_df, ["Denominación función", "Denominacion funcion", "Cargo", "Función.1"], required=False)
            col_area_nom = find_col(header_df, ["Texto área nómina", "Área de nómina", "Area de nomina"], required=False)
            col_div = find_col(header_df, ["Texto división de personal", "División de personal"], required=False)
            col_centro = find_col(header_df, ["Denominación", "Centro de coste"], required=False)
            needed_cols = list(dict.fromkeys([c for c in [col_period, col_concept, col_text, col_qty, col_val, col_ceco, col_func, col_cargo, col_area_nom, col_div, col_centro] if c]))
            df = read_table(f, usecols=needed_cols)
            df.columns = [str(c).strip() for c in df.columns]

            out = pd.DataFrame()
            out["source_file"] = name
            out["periodo_pago"] = df[col_period].map(lambda x: period_from_any(x, name)) if col_period else period_from_any(None, name)
            out["periodo_pago"] = fill_period_series(out["periodo_pago"], period_from_any(None, name))
            out["periodo_novedad"] = out["periodo_pago"].map(lambda p: shift_period(p, -1))
            out["concepto"] = df[col_concept].map(lambda x: clean_code(x).upper())
            out = out[out["concepto"].isin(CONCEPTOS)].copy()
            if out.empty:
                alerts.append(f"{name}: no encontré conceptos objetivo {CONCEPTOS}.")
                continue
            idx = out.index
            out["texto_concepto"] = df.loc[idx, col_text].astype(str).values if col_text else out["concepto"]
            out["cantidad_pagada"] = to_num(df.loc[idx, col_qty]).values
            out["valor_pagado"] = to_num(df.loc[idx, col_val]).values
            out["ceco"] = df.loc[idx, col_ceco].map(clean_ceco).values if col_ceco else ""
            out["funcion"] = df.loc[idx, col_func].map(clean_code).values if col_func else ""
            out["cargo_original"] = df.loc[idx, col_cargo].astype(str).values if col_cargo else ""
            out["area_nomina"] = df.loc[idx, col_area_nom].astype(str).values if col_area_nom else ""
            div_vals = df.loc[idx, col_div].astype(str).values if col_div else [""] * len(out)
            centro_vals = df.loc[idx, col_centro].astype(str).values if col_centro else [""] * len(out)
            out["tipo_hora"] = out["concepto"].map(lambda c: tipo_hora(c, hom))
            homol = [homologar_cargo(fun, car, hom) for fun, car in zip(out["funcion"], out["cargo_original"])]
            out["cargo_homologado"] = [x[0] for x in homol]
            out["cargo_homologado_ok"] = [x[1] for x in homol]
            out["area_negocio"] = [classify_area(c, None, d, a, cen) for c, d, a, cen in zip(out["ceco"], div_vals, out["area_nomina"], centro_vals)]
            out["fuente"] = "CC-nómina"
            rows.append(out)
        except Exception as e:
            alerts.append(f"Error procesando CC-nómina {name}: {e}")
    if not rows:
        return pd.DataFrame(), alerts
    res = pd.concat(rows, ignore_index=True)
    not_ok = res.loc[~res["cargo_homologado_ok"], "cargo_original"].dropna().astype(str).unique()
    if len(not_ok):
        alerts.append(f"CC-nómina: {len(not_ok)} cargos quedaron por regla/patrón o sin homologación exacta. Revisa hoja Alertas.")
    return res, alerts


def process_compensatorios(files, hom) -> Tuple[pd.DataFrame, List[str]]:
    alerts = []
    if not files:
        return pd.DataFrame(), []
    cc_rows = []
    for f in files:
        name = uploaded_name(f)
        try:
            df = read_table(f)
            df.columns = [str(c).strip() for c in df.columns]
            col_period = find_col(df, ["Per.para", "Periodo", "Periodo pago"], required=False)
            col_concept = find_col(df, ["CC-n.", "CC-n", "Concepto"], required=False)
            col_qty = find_col(df, ["Cantidad", "Cant"], required=True)
            col_val = find_col(df, ["Importe", "Valor"], required=True)
            col_ceco = find_col(df, ["Ce.coste", "CECO", "Centro de coste"], required=False)
            col_func = find_col(df, ["Función", "Funcion"], required=False)
            col_cargo = find_col(df, ["Denominación función", "Denominacion funcion", "Cargo"], required=False)
            col_area_nom = find_col(df, ["Texto área nómina", "Área de nómina", "Area de nomina"], required=False)
            col_div = find_col(df, ["Texto división de personal", "División de personal"], required=False)
            col_centro = find_col(df, ["Denominación", "Centro de coste"], required=False)
            out = pd.DataFrame()
            out["source_file"] = name
            out["periodo_pago"] = df[col_period].map(lambda x: period_from_any(x, name)) if col_period else period_from_any(None, name)
            out["periodo_pago"] = fill_period_series(out["periodo_pago"], period_from_any(None, name))
            out["periodo_novedad"] = out["periodo_pago"].map(lambda p: shift_period(p, -1))
            out["concepto"] = df[col_concept].map(lambda x: clean_code(x).upper()) if col_concept else "Y350"
            out["concepto"] = out["concepto"].replace("", "Y350").fillna("Y350")
            out = out[out["concepto"].eq("Y350")].copy()
            idx = out.index
            out["texto_concepto"] = "Compensatorio"
            out["cantidad_pagada"] = to_num(df.loc[idx, col_qty]).values
            out["valor_pagado"] = to_num(df.loc[idx, col_val]).values
            out["ceco"] = df.loc[idx, col_ceco].map(clean_ceco).values if col_ceco else ""
            out["funcion"] = df.loc[idx, col_func].map(clean_code).values if col_func else ""
            out["cargo_original"] = df.loc[idx, col_cargo].astype(str).values if col_cargo else ""
            out["area_nomina"] = df.loc[idx, col_area_nom].astype(str).values if col_area_nom else ""
            div_vals = df.loc[idx, col_div].astype(str).values if col_div else [""] * len(out)
            centro_vals = df.loc[idx, col_centro].astype(str).values if col_centro else [""] * len(out)
            out["tipo_hora"] = "Compensatorio"
            homol = [homologar_cargo(fun, car, hom) for fun, car in zip(out["funcion"], out["cargo_original"])]
            out["cargo_homologado"] = [x[0] for x in homol]
            out["cargo_homologado_ok"] = [x[1] for x in homol]
            out["area_negocio"] = [classify_area(c, None, d, a, cen) for c, d, a, cen in zip(out["ceco"], div_vals, out["area_nomina"], centro_vals)]
            out["fuente"] = "Compensatorios"
            cc_rows.append(out)
        except Exception as e:
            alerts.append(f"Error procesando compensatorios {name}: {e}")
    return (pd.concat(cc_rows, ignore_index=True) if cc_rows else pd.DataFrame()), alerts


def process_provision(files, hom, md_periodo=None) -> Tuple[pd.DataFrame, List[str]]:
    alerts, rows = [], []
    if not files:
        return pd.DataFrame(), ["No se cargó provisión."]

    # Lookup por período + posición/texto de cargo desde Headcount/MD del período.
    # Sirve para bases sin SAP: si el reporte trae nombre de posición, primero buscamos
    # esa posición en el Headcount del mes para recuperar la función real y luego homologar.
    pos_lookup = pd.DataFrame()
    if md_periodo is not None and not md_periodo.empty and "periodo_novedad" in md_periodo.columns:
        tmp_lu = md_periodo.copy()
        if "posicion_original" in tmp_lu.columns:
            tmp_lu["posicion_key"] = tmp_lu["posicion_original"].map(norm_text)
        else:
            tmp_lu["posicion_key"] = ""
        if "cargo_original" in tmp_lu.columns:
            tmp_lu["cargo_key"] = tmp_lu["cargo_original"].map(norm_text)
        else:
            tmp_lu["cargo_key"] = ""
        keep = [c for c in ["periodo_novedad", "posicion_key", "cargo_key", "funcion", "funcion_nombre", "cargo_original", "cargo_homologado", "area_negocio", "ceco", "area_nomina"] if c in tmp_lu.columns]
        pos_lookup = tmp_lu[keep].copy()
        pos_lookup = pos_lookup.drop_duplicates(["periodo_novedad", "posicion_key"], keep="last")

    for f in files:
        name = uploaded_name(f)
        try:
            sh = first_existing_sheet(f, ["Horas_Provisión", "Horas_Provision", "Provisión", "Provision"])
            header_df = read_table(f, sheet_name=sh, nrows=0)
            header_df.columns = [str(c).strip() for c in header_df.columns]
            col_source = find_col(header_df, ["Source.Name", "MES", "Archivo", "Periodo"], required=False)
            col_concept = find_col(header_df, ["Valores", "Concepto", "CC-n."], required=True)
            col_qty = find_col(header_df, ["Total", "Cantidad"], required=True)
            col_val = find_col(header_df, ["PROVISIÓN", "Provision", "Valor provisionado"], required=True)
            col_ceco = find_col(header_df, ["CECO", "Ce.coste"], required=False)
            col_tipo = find_col(header_df, ["TIPO", "Tipo"], required=False)
            col_cargo = find_col(header_df, ["CARGO", "Cargo", "Funcion"], required=False)
            needed_cols = list(dict.fromkeys([c for c in [col_source, col_concept, col_qty, col_val, col_ceco, col_tipo, col_cargo] if c]))
            df = read_table(f, sheet_name=sh, usecols=needed_cols)
            df.columns = [str(c).strip() for c in df.columns]
            out = pd.DataFrame()
            out["source_file"] = name
            src = df[col_source] if col_source else pd.Series([name] * len(df))
            out["periodo_novedad"] = fill_period_series(src.map(lambda x: period_from_any(x, name)), period_from_any(None, name))
            out["concepto"] = df[col_concept].map(lambda x: clean_code(x).upper())
            out = out[out["concepto"].isin(CONCEPTOS)].copy()
            idx = out.index
            out["tipo_hora"] = out["concepto"].map(lambda c: tipo_hora(c, hom))
            out["cantidad_provisionada"] = to_num(df.loc[idx, col_qty]).values
            out["valor_provisionado"] = to_num(df.loc[idx, col_val]).values
            out["ceco"] = df.loc[idx, col_ceco].map(clean_ceco).values if col_ceco else ""
            tipo_vals = df.loc[idx, col_tipo].astype(str).values if col_tipo else [""] * len(out)
            out["cargo_original"] = df.loc[idx, col_cargo].astype(str).values if col_cargo else ""
            out["cargo_key"] = out["cargo_original"].map(norm_text)
            out["metodo_homologacion"] = "Cargo/función provisión"
            if not pos_lookup.empty:
                # V8.1.1: evitar columnas duplicadas (cargo_key) al renombrar posicion_key.
                # Se hacen 2 intentos ordenados:
                #   1) posición/cargo del reporte contra posición del Headcount del mismo período
                #   2) posición/cargo del reporte contra función/cargo del Headcount del mismo período
                lu_pos = pos_lookup.drop(columns=["cargo_key"], errors="ignore").rename(columns={"posicion_key": "cargo_key"})
                lu_pos = lu_pos.loc[:, ~lu_pos.columns.duplicated()].copy()
                out = out.loc[:, ~out.columns.duplicated()].copy()
                out = out.merge(lu_pos, on=["periodo_novedad", "cargo_key"], how="left", suffixes=("", "_hc"))

                tiene_hc = out.get("cargo_homologado", pd.Series([np.nan]*len(out))).notna()

                # Fallback: si no cruzó por posición, intenta por cargo/función del Headcount.
                if (~tiene_hc).any() and "cargo_key" in pos_lookup.columns:
                    lu_fun = pos_lookup.drop(columns=["posicion_key"], errors="ignore").copy()
                    lu_fun = lu_fun.loc[:, ~lu_fun.columns.duplicated()].copy()
                    lu_fun = lu_fun.drop_duplicates(["periodo_novedad", "cargo_key"], keep="last")
                    missing_idx = out.index[~tiene_hc]
                    base_missing = out.loc[missing_idx].drop(columns=[c for c in lu_fun.columns if c not in ["periodo_novedad", "cargo_key"]], errors="ignore")
                    base_missing = base_missing.loc[:, ~base_missing.columns.duplicated()].copy()
                    merged_missing = base_missing.merge(lu_fun, on=["periodo_novedad", "cargo_key"], how="left", suffixes=("", "_hc2"))
                    for col in ["funcion", "funcion_nombre", "cargo_original", "cargo_homologado", "area_negocio", "ceco", "area_nomina"]:
                        if col in merged_missing.columns:
                            out.loc[missing_idx, col] = merged_missing[col].values
                    tiene_hc = out.get("cargo_homologado", pd.Series([np.nan]*len(out))).notna()

                out["cargo_original_hc"] = out.get("cargo_original_hc", out.get("cargo_original", ""))
                out["cargo_para_homologar"] = np.where(tiene_hc, out.get("cargo_original_hc", out["cargo_original"]), out["cargo_original"])
                out["funcion_para_homologar"] = np.where(tiene_hc, out.get("funcion", ""), "")
                ceco_hc_col = "ceco_hc" if "ceco_hc" in out.columns else "ceco"
                area_hc_col = "area_negocio_hc" if "area_negocio_hc" in out.columns else ("area_negocio" if "area_negocio" in out.columns else None)
                out["ceco"] = np.where(tiene_hc & out[ceco_hc_col].astype(str).str.strip().ne(""), out[ceco_hc_col], out["ceco"])
                if area_hc_col:
                    out["area_negocio_pre"] = np.where(tiene_hc & out[area_hc_col].astype(str).str.strip().ne(""), out[area_hc_col], "")
                else:
                    out["area_negocio_pre"] = ""
                out["metodo_homologacion"] = np.where(tiene_hc, "Posición/función provisión + Headcount período → Función", "Cargo/función provisión")
            else:
                out["cargo_para_homologar"] = out["cargo_original"]
                out["funcion_para_homologar"] = ""
                out["area_negocio_pre"] = ""
            homol = [homologar_cargo(fun, car, hom) for fun, car in zip(out["funcion_para_homologar"], out["cargo_para_homologar"])]
            out["cargo_homologado"] = [x[0] for x in homol]
            out["cargo_homologado_ok"] = [x[1] for x in homol]
            out["area_negocio"] = np.where(out["area_negocio_pre"].astype(str).str.strip().ne(""), out["area_negocio_pre"], [classify_area(c, t, None, None, None) for c, t in zip(out["ceco"], tipo_vals)])
            drop_cols = ["cargo_key", "cargo_para_homologar", "funcion_para_homologar", "area_negocio_pre"]
            out = out.drop(columns=[c for c in drop_cols if c in out.columns])
            rows.append(out)
        except Exception as e:
            alerts.append(f"Error procesando provisión {name}: {e}")
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), alerts


def process_proyeccion(files, hom, md_periodo=None) -> Tuple[pd.DataFrame, List[str]]:
    """Procesa la proyección.

    Reglas de homologación de cargo:
    1) Si la proyección trae SAP válido y existe ese SAP en el Headcount/MD del mismo periodo,
       toma la función/cargo/CECO/área de ese periodo.
    2) Si SAP viene Error, vacío o no cruza, toma la columna Función/Cargo propia de la proyección.
    3) Si tampoco homologa, deja alerta para revisión.
    """
    alerts, rows = [], []
    if not files:
        return pd.DataFrame(), ["No se cargó proyección."]

    md_lookup = pd.DataFrame()
    pos_lookup = pd.DataFrame()
    if md_periodo is not None and not md_periodo.empty:
        cols = [c for c in ["periodo_novedad", "sap", "funcion", "funcion_nombre", "posicion_original", "cargo_original", "cargo_homologado", "area_negocio", "ceco", "area_nomina"] if c in md_periodo.columns]
        md_lookup = md_periodo[cols].copy()
        if "sap" in md_lookup.columns:
            md_lookup["sap"] = md_lookup["sap"].map(clean_code)
        md_lookup = md_lookup.drop_duplicates(["periodo_novedad", "sap"], keep="last") if {"periodo_novedad", "sap"}.issubset(md_lookup.columns) else pd.DataFrame()

        # Lookup adicional por texto de posición/cargo del reporte contra Headcount del mismo período.
        pos_lookup = md_periodo.copy()
        pos_lookup["posicion_key"] = pos_lookup["posicion_original"].map(norm_text) if "posicion_original" in pos_lookup.columns else ""
        keep_pos = [c for c in ["periodo_novedad", "posicion_key", "funcion", "funcion_nombre", "cargo_original", "cargo_homologado", "area_negocio", "ceco", "area_nomina"] if c in pos_lookup.columns]
        pos_lookup = pos_lookup[keep_pos].copy()
        pos_lookup = pos_lookup[pos_lookup["posicion_key"].astype(str).str.strip().ne("")].drop_duplicates(["periodo_novedad", "posicion_key"], keep="last") if {"periodo_novedad", "posicion_key"}.issubset(pos_lookup.columns) else pd.DataFrame()

    for f in files:
        name = uploaded_name(f)
        try:
            sh = first_existing_sheet(f, ["Horas_Proyección", "Horas_Proyeccion", "Proyección", "Proyeccion"])
            df = read_table(f, sheet_name=sh)
            df.columns = [str(c).strip() for c in df.columns]
            col_period = find_col(df, ["MES", "Periodo", "Source.Name"], required=True)
            col_sap = find_col(df, ["SAP", "Nº pers.", "N° pers.", "Número de personal"], required=False)
            col_ceco = find_col(df, ["Ce.coste", "CECO"], required=False)
            col_cargo = find_col(df, ["Funcion", "Función", "Cargo"], required=False)
            col_tipo = find_col(df, ["Tipo"], required=False)
            col_area_nom = find_col(df, ["Área de nómina", "Area de nomina"], required=False)
            base = pd.DataFrame({
                "source_file": name,
                "periodo_novedad": fill_period_series(df[col_period].map(lambda x: period_from_any(x, name)), period_from_any(None, name)),
                "sap": df[col_sap].map(clean_code) if col_sap else "",
                "ceco_origen": df[col_ceco].map(clean_ceco) if col_ceco else "",
                "cargo_original_base": df[col_cargo].astype(str) if col_cargo else "",
                "tipo_origen": df[col_tipo].astype(str) if col_tipo else "",
                "area_nomina_origen": df[col_area_nom].astype(str) if col_area_nom else "",
            })
            # SAP inválido: Error, vacío, nan o no numérico. En esos casos NO se fuerza cruce por SAP.
            base["sap_valido"] = base["sap"].astype(str).str.fullmatch(r"\d+").fillna(False)

            if not md_lookup.empty:
                base = base.merge(md_lookup, on=["periodo_novedad", "sap"], how="left", suffixes=("", "_md"))
                base["tiene_md_periodo"] = base["cargo_homologado"].notna() & base["sap_valido"]
            else:
                base["funcion"] = ""
                base["cargo_original"] = ""
                base["cargo_homologado"] = np.nan
                base["area_negocio"] = np.nan
                base["ceco"] = np.nan
                base["area_nomina"] = np.nan
                base["tiene_md_periodo"] = False

            # Fallback 1 cuando SAP es Error/vacío/no cruza: validar si el texto de la proyección
            # existe como posición en el Headcount del mismo período. Si existe, se toma la función
            # de ese Headcount y solo después se homologa.
            base["posicion_key"] = base["cargo_original_base"].map(norm_text)
            if not pos_lookup.empty:
                base = base.merge(pos_lookup.rename(columns={
                    "funcion": "funcion_pos",
                    "funcion_nombre": "funcion_nombre_pos",
                    "cargo_original": "cargo_original_pos",
                    "cargo_homologado": "cargo_homologado_pos",
                    "area_negocio": "area_negocio_pos",
                    "ceco": "ceco_pos",
                    "area_nomina": "area_nomina_pos",
                }), on=["periodo_novedad", "posicion_key"], how="left")
                base["tiene_pos_periodo"] = base["cargo_homologado_pos"].notna() & (~base["tiene_md_periodo"])
            else:
                base["tiene_pos_periodo"] = False
                for c in ["funcion_pos", "cargo_original_pos", "cargo_homologado_pos", "area_negocio_pos", "ceco_pos", "area_nomina_pos"]:
                    base[c] = np.nan

            base["cargo_para_homologar"] = np.select(
                [base["tiene_md_periodo"], base["tiene_pos_periodo"]],
                [base.get("cargo_original", ""), base.get("cargo_original_pos", "")],
                default=base["cargo_original_base"]
            )
            base["funcion_para_homologar"] = np.select(
                [base["tiene_md_periodo"], base["tiene_pos_periodo"]],
                [base.get("funcion", ""), base.get("funcion_pos", "")],
                default=""
            )
            fallback_hom = [homologar_cargo(fun, car, hom) for fun, car in zip(base["funcion_para_homologar"], base["cargo_para_homologar"])]
            base["cargo_homologado_calc"] = [x[0] for x in fallback_hom]
            base["cargo_homologado_ok_calc"] = [x[1] for x in fallback_hom]
            base["cargo_homologado_final"] = np.where(base["tiene_md_periodo"], base["cargo_homologado"], base["cargo_homologado_calc"])
            base["cargo_homologado_ok_final"] = np.where(base["tiene_md_periodo"], True, base["cargo_homologado_ok_calc"])
            base["ceco_final"] = np.select(
                [base["tiene_md_periodo"], base["tiene_pos_periodo"]],
                [base["ceco"].fillna(""), base.get("ceco_pos", "").fillna("")],
                default=base["ceco_origen"]
            )
            base["area_nomina_final"] = np.select(
                [base["tiene_md_periodo"], base["tiene_pos_periodo"]],
                [base.get("area_nomina", "").fillna(""), base.get("area_nomina_pos", "").fillna("")],
                default=base["area_nomina_origen"]
            )
            base["area_negocio_final"] = np.select(
                [base["tiene_md_periodo"], base["tiene_pos_periodo"]],
                [base["area_negocio"].fillna("Sin clasificar"), base.get("area_negocio_pos", "").fillna("Sin clasificar")],
                default=[classify_area(c, t, None, a, None) for c, t, a in zip(base["ceco_origen"], base["tipo_origen"], base["area_nomina_origen"])]
            )
            base["metodo_homologacion"] = np.select(
                [base["tiene_md_periodo"], base["tiene_pos_periodo"], base["sap_valido"]],
                ["SAP + Headcount/MD del período", "Posición proyección + Headcount período → Función", "Función/Cargo proyección (SAP no cruzó)"],
                default="Función/Cargo proyección (SAP inválido/Error)"
            )

            for concepto in CONCEPTOS:
                q_col = f"{concepto}_Q"
                v_col = f"{concepto}_$"
                if q_col not in df.columns and v_col not in df.columns:
                    continue
                tmp = base.copy()
                tmp["concepto"] = concepto
                tmp["tipo_hora"] = tipo_hora(concepto, hom)
                tmp["cantidad_proyectada"] = to_num(df[q_col]) if q_col in df.columns else 0.0
                tmp["valor_proyectado"] = to_num(df[v_col]) if v_col in df.columns else 0.0
                tmp = tmp[(tmp["cantidad_proyectada"].abs() > 0) | (tmp["valor_proyectado"].abs() > 0)].copy()
                if tmp.empty:
                    continue
                out = pd.DataFrame()
                out["source_file"] = tmp["source_file"]
                out["periodo_novedad"] = tmp["periodo_novedad"]
                out["sap"] = tmp["sap"]
                out["concepto"] = tmp["concepto"]
                out["tipo_hora"] = tmp["tipo_hora"]
                out["cantidad_proyectada"] = tmp["cantidad_proyectada"]
                out["valor_proyectado"] = tmp["valor_proyectado"]
                out["ceco"] = tmp["ceco_final"].map(clean_ceco)
                out["cargo_original"] = tmp["cargo_para_homologar"].astype(str)
                out["cargo_homologado"] = tmp["cargo_homologado_final"].astype(str)
                out["cargo_homologado_ok"] = tmp["cargo_homologado_ok_final"].astype(bool)
                out["area_negocio"] = tmp["area_negocio_final"].astype(str)
                out["metodo_homologacion"] = tmp["metodo_homologacion"].astype(str)
                rows.append(out)

            sap_error = int((~base["sap_valido"]).sum()) if col_sap else 0
            if sap_error:
                alerts.append(f"{name}: {sap_error:,.0f} registros de proyección tenían SAP inválido/Error; se homologaron por Función/Cargo de la proyección.")
        except Exception as e:
            alerts.append(f"Error procesando proyección {name}: {e}")
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), alerts


def process_headcount(files, hom) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    alerts, rows, excl = [], [], []
    if not files:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ["No se cargó Headcount."]
    for f in files:
        name = uploaded_name(f)
        try:
            df = read_table(f)
            df.columns = [str(c).strip() for c in df.columns]
            col_sap = find_col(df, ["Nº pers.", "N° pers.", "SAP", "Número de personal"], required=True)
            col_status = find_col(df, ["Status ocupación", "Status ocupacion", "Estado"], required=False)
            col_ceco = find_col(df, ["Ce.coste", "CECO"], required=False)
            # En Headcount SAP suele traer código y texto duplicados:
            # Posición / Posición.1 y Función / Función.1.
            # Para homologar correctamente no usamos el texto de posición como cargo final;
            # lo usamos como llave para encontrar la función del período.
            col_func = find_col(df, ["Función", "Funcion"], required=False)
            col_func_name = find_col(df, ["Función.1", "Funcion.1", "Denominación función", "Denominacion funcion"], required=False)
            col_pos_name = find_col(df, ["Posición.1", "Posicion.1", "Denominación posición", "Denominacion posicion", "Posición", "Posicion"], required=False)
            col_cargo = col_func_name or col_pos_name
            col_area_nom = find_col(df, ["Área de nómina", "Area de nomina"], required=False)
            col_div = find_col(df, ["División de personal", "Division de personal"], required=False)
            col_area_pers = find_col(df, ["Área de personal", "Area de personal"], required=False)
            work = df.copy()
            if col_status:
                work = work[work[col_status].astype(str).str.upper().str.contains("ACTIVO", na=False)].copy()
            period = period_from_any(None, name)
            # Excluir Manager I, II, III, IV, sin afectar Non Manager.
            check_cols = [c for c in [col_area_pers, col_area_nom, col_cargo] if c]
            if check_cols:
                mask_excl = work[check_cols].apply(is_manager_excluido, axis=1)
                if mask_excl.any():
                    ex = work.loc[mask_excl].copy()
                    ex["periodo_novedad"] = period
                    ex["motivo_exclusion"] = "Manager I-IV / no aplica tiempo suplementario"
                    excl.append(ex)
                    work = work.loc[~mask_excl].copy()
            out = pd.DataFrame()
            out["periodo_novedad"] = period
            out["sap"] = work[col_sap].map(clean_code)
            out["ceco"] = work[col_ceco].map(clean_ceco) if col_ceco else ""
            out["funcion"] = work[col_func].map(clean_code) if col_func else ""
            out["funcion_nombre"] = work[col_func_name].astype(str) if col_func_name else ""
            out["posicion_original"] = work[col_pos_name].astype(str) if col_pos_name else ""
            # Cargo final para homologar = función del período. La posición queda solo como llave/auditoría.
            out["cargo_original"] = np.where(out["funcion_nombre"].astype(str).str.strip().ne(""), out["funcion_nombre"], out["posicion_original"])
            out["area_nomina"] = work[col_area_nom].astype(str) if col_area_nom else ""
            div_vals = work[col_div].astype(str).values if col_div else [""] * len(out)
            homol = [homologar_cargo(fun, car, hom) for fun, car in zip(out["funcion"], out["cargo_original"])]
            out["cargo_homologado"] = [x[0] for x in homol]
            out["cargo_homologado_ok"] = [x[1] for x in homol]
            out["area_negocio"] = [classify_area(c, None, d, a, None) for c, d, a in zip(out["ceco"], div_vals, out["area_nomina"])]
            rows.append(out)
        except Exception as e:
            alerts.append(f"Error procesando Headcount {name}: {e}")
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), alerts
    detail = pd.concat(rows, ignore_index=True)
    hc = detail.groupby(KEY_HC, dropna=False).agg(hc=("sap", "nunique")).reset_index()
    excl_df = pd.concat(excl, ignore_index=True) if excl else pd.DataFrame()
    if not excl_df.empty:
        alerts.append(f"Headcount: se excluyeron {len(excl_df):,.0f} registros Manager I-IV/no aplican horas.")
    return hc, excl_df, detail, alerts



def build_maestro_posicion_funcion(hc_detail: pd.DataFrame, hom=None) -> pd.DataFrame:
    """Construye el maestro base de homologación desde TODOS los Headcount cargados.

    Idea de negocio V9:
    1) El Headcount trae posición y función del período.
    2) Antes de comparar, convertimos cualquier texto de posición/cargo de las fuentes
       a la función correspondiente usando este maestro.
    3) Solo después homologamos la función a los agrupados ejecutivos: Jefe Tienda,
       Operador Tienda, Op. Cedi, etc.

    Esto evita que el comparativo se fragmente por variaciones de nombre de posición
    como "Operador tienda", "Operador de tienda", "Operador Tienda Encargado".
    """
    cols_out = [
        "periodo_novedad", "posicion_original", "posicion_key",
        "funcion", "funcion_nombre", "funcion_key", "cargo_homologado",
        "area_negocio", "ceco", "area_nomina",
    ]
    if hc_detail is None or hc_detail.empty:
        return pd.DataFrame(columns=cols_out)
    d = hc_detail.copy()
    for c in ["periodo_novedad", "posicion_original", "funcion", "funcion_nombre", "cargo_original", "area_negocio", "ceco", "area_nomina"]:
        if c not in d.columns:
            d[c] = ""
    # Función texto: prioriza texto de función; si no existe, usa cargo_original; si no existe, posición.
    d["funcion_nombre"] = d["funcion_nombre"].astype(str)
    d["posicion_original"] = d["posicion_original"].astype(str)
    d["cargo_original"] = d["cargo_original"].astype(str)
    d["funcion_nombre"] = np.where(
        d["funcion_nombre"].str.strip().ne("") & d["funcion_nombre"].str.upper().ne("NAN"),
        d["funcion_nombre"],
        np.where(d["cargo_original"].str.strip().ne(""), d["cargo_original"], d["posicion_original"])
    )
    d["posicion_key"] = d["posicion_original"].map(norm_text)
    d["funcion_key"] = d["funcion_nombre"].map(norm_text)
    # Recalcula cargo homologado estrictamente desde función.
    homol = [homologar_cargo(fun, fn, hom) for fun, fn in zip(d["funcion"], d["funcion_nombre"])]
    d["cargo_homologado"] = [x[0] for x in homol]
    # Filtra llaves vacías.
    d = d[(d["posicion_key"].astype(str).str.strip().ne("")) | (d["funcion_key"].astype(str).str.strip().ne(""))].copy()
    if d.empty:
        return pd.DataFrame(columns=cols_out)
    # Preferir registros con función texto/código y con área clasificada.
    d["_score"] = 0
    d["_score"] += d["funcion_nombre"].astype(str).str.strip().ne("").astype(int) * 4
    d["_score"] += d["funcion"].astype(str).str.strip().ne("").astype(int) * 2
    d["_score"] += d["area_negocio"].astype(str).str.upper().ne("SIN CLASIFICAR").astype(int)
    d = d.sort_values(["periodo_novedad", "_score"], ascending=[True, False])
    keep_cols = [c for c in cols_out if c in d.columns]
    # Maestro por posición del período.
    maestro = d[keep_cols].drop_duplicates(["periodo_novedad", "posicion_key"], keep="first")
    # Agrega filas alternas por función_key para que si la fuente ya trae función, también cruce.
    alt = d[keep_cols].copy()
    alt["posicion_key"] = alt["funcion_key"]
    maestro = pd.concat([maestro, alt], ignore_index=True)
    maestro = maestro.drop_duplicates(["periodo_novedad", "posicion_key"], keep="first")
    return maestro.reset_index(drop=True)


def apply_maestro_posicion_funcion(df: pd.DataFrame, maestro: pd.DataFrame, hom=None, fuente: str = "") -> pd.DataFrame:
    """Convierte cargo/posición original de una fuente a función del Headcount del período.

    Prioridad:
    1) periodo_novedad + nombre normalizado de posición/cargo contra maestro HC.
    2) nombre normalizado contra maestro global HC (cuando no coincide el período exacto).
    3) si no cruza, conserva función/cargo original y deja método para revisión.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in ["periodo_novedad", "cargo_original", "funcion", "funcion_nombre", "cargo_homologado", "area_negocio", "ceco", "area_nomina"]:
        if c not in out.columns:
            out[c] = ""
    out["cargo_reporte_original"] = out["cargo_original"].astype(str)
    out["cargo_key_src"] = out["cargo_original"].map(norm_text)
    # Si viene una función_nombre explícita, también puede ser llave.
    out["funcion_key_src"] = out.get("funcion_nombre", pd.Series([""] * len(out))).map(norm_text)

    if maestro is not None and not maestro.empty:
        m = maestro.copy()
        for c in ["periodo_novedad", "posicion_key", "funcion", "funcion_nombre", "cargo_homologado", "area_negocio", "ceco", "area_nomina"]:
            if c not in m.columns:
                m[c] = ""
        m = m.drop_duplicates(["periodo_novedad", "posicion_key"], keep="first")
        m2 = m[["periodo_novedad", "posicion_key", "funcion", "funcion_nombre", "cargo_homologado", "area_negocio", "ceco", "area_nomina"]].copy()
        m2 = m2.rename(columns={
            "funcion": "funcion_hc",
            "funcion_nombre": "funcion_nombre_hc",
            "cargo_homologado": "cargo_homologado_hc",
            "area_negocio": "area_negocio_hc",
            "ceco": "ceco_hc",
            "area_nomina": "area_nomina_hc",
        })
        out = out.merge(m2, left_on=["periodo_novedad", "cargo_key_src"], right_on=["periodo_novedad", "posicion_key"], how="left")
        # Fallback global: si la posición existe en otro Headcount cargado, usarla.
        missing = out["funcion_nombre_hc"].isna() | out["funcion_nombre_hc"].astype(str).str.strip().eq("")
        if missing.any():
            mg = m.sort_values("periodo_novedad").drop_duplicates(["posicion_key"], keep="last")
            mg = mg[["posicion_key", "funcion", "funcion_nombre", "cargo_homologado", "area_negocio", "ceco", "area_nomina"]].rename(columns={
                "funcion": "funcion_hc_g",
                "funcion_nombre": "funcion_nombre_hc_g",
                "cargo_homologado": "cargo_homologado_hc_g",
                "area_negocio": "area_negocio_hc_g",
                "ceco": "ceco_hc_g",
                "area_nomina": "area_nomina_hc_g",
            })
            tmp = out.loc[missing].drop(columns=[c for c in mg.columns if c != "posicion_key"], errors="ignore")
            tmp = tmp.merge(mg, left_on="cargo_key_src", right_on="posicion_key", how="left")
            for base_col in ["funcion", "funcion_nombre", "cargo_homologado", "area_negocio", "ceco", "area_nomina"]:
                gcol = f"{base_col}_hc_g"
                if gcol in tmp.columns:
                    out.loc[missing, f"{base_col}_hc_g"] = tmp[gcol].values
        else:
            for base_col in ["funcion", "funcion_nombre", "cargo_homologado", "area_negocio", "ceco", "area_nomina"]:
                out[f"{base_col}_hc_g"] = np.nan

        # Aplica función encontrada por período o global.
        found_period = out["funcion_nombre_hc"].notna() & out["funcion_nombre_hc"].astype(str).str.strip().ne("")
        found_global = (~found_period) & out.get("funcion_nombre_hc_g", pd.Series([np.nan]*len(out))).notna() & out.get("funcion_nombre_hc_g", pd.Series([""]*len(out))).astype(str).str.strip().ne("")
        out["funcion"] = np.select([found_period, found_global], [out.get("funcion_hc", ""), out.get("funcion_hc_g", "")], default=out["funcion"])
        out["funcion_nombre"] = np.select([found_period, found_global], [out.get("funcion_nombre_hc", ""), out.get("funcion_nombre_hc_g", "")], default=np.where(out["funcion_nombre"].astype(str).str.strip().ne(""), out["funcion_nombre"], out["cargo_original"]))
        # Cargo original para el comparativo pasa a ser la función ya normalizada.
        out["cargo_original"] = out["funcion_nombre"]
        homol = [homologar_cargo(fun, fn, hom) for fun, fn in zip(out["funcion"], out["funcion_nombre"])]
        out["cargo_homologado"] = [x[0] for x in homol]
        out["cargo_homologado_ok"] = [x[1] for x in homol]
        # Área: prioriza la fuente; si está sin clasificar, usa HC.
        area_src_bad = out["area_negocio"].astype(str).str.strip().eq("") | out["area_negocio"].astype(str).str.upper().eq("SIN CLASIFICAR")
        out["area_negocio"] = np.where(area_src_bad & found_period, out.get("area_negocio_hc", out["area_negocio"]), out["area_negocio"])
        out["area_negocio"] = np.where(area_src_bad & found_global, out.get("area_negocio_hc_g", out["area_negocio"]), out["area_negocio"])
        out["metodo_homologacion"] = np.select(
            [found_period, found_global],
            ["Posición/Cargo fuente → Headcount del período → Función", "Posición/Cargo fuente → Headcount consolidado → Función"],
            default="Sin cruce en Headcount; función/cargo original"
        )
        drop_aux = [c for c in out.columns if c.endswith("_hc") or c.endswith("_hc_g") or c in ["posicion_key", "posicion_key_x", "posicion_key_y", "cargo_key_src", "funcion_key_src"]]
        out = out.drop(columns=drop_aux, errors="ignore")
    else:
        out["funcion_nombre"] = np.where(out["funcion_nombre"].astype(str).str.strip().ne(""), out["funcion_nombre"], out["cargo_original"])
        out["cargo_original"] = out["funcion_nombre"]
        homol = [homologar_cargo(fun, fn, hom) for fun, fn in zip(out["funcion"], out["funcion_nombre"])]
        out["cargo_homologado"] = [x[0] for x in homol]
        out["cargo_homologado_ok"] = [x[1] for x in homol]
        out["metodo_homologacion"] = "Sin maestro Headcount; función/cargo original"
    return out


def force_key_types(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Evita ValueError de pandas al hacer merge por llaves con dtype distinto."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for k in keys:
        if k not in out.columns:
            out[k] = ""
        # Todo key se compara como texto normalizado/controlado.
        out[k] = out[k].fillna("").astype(str)
        if k == "ceco":
            out[k] = out[k].map(clean_ceco)
        elif k == "concepto":
            out[k] = out[k].map(lambda x: clean_code(x).upper())
        elif k == "periodo_novedad":
            out[k] = out[k].map(lambda x: period_from_any(x, None) or str(x))
        else:
            out[k] = out[k].str.strip()
    return out

def aggregate_sources(pagado, provision, proyeccion, hc) -> pd.DataFrame:
    # V9: antes de agrupar/mergear, fuerza llaves a texto para evitar errores por dtype distinto
    # (por ejemplo CECO numérico en una fuente y texto en otra).
    pagado = force_key_types(pagado, KEY_COMPARATIVO) if pagado is not None else pagado
    provision = force_key_types(provision, KEY_COMPARATIVO) if provision is not None else provision
    proyeccion = force_key_types(proyeccion, KEY_COMPARATIVO) if proyeccion is not None else proyeccion
    hc = force_key_types(hc, KEY_HC) if hc is not None else hc

    frames = []
    if pagado is not None and not pagado.empty:
        p = pagado.groupby(KEY_COMPARATIVO, dropna=False).agg(
            cantidad_pagada=("cantidad_pagada", "sum"),
            valor_pagado=("valor_pagado", "sum"),
        ).reset_index()
        p = force_key_types(p, KEY_COMPARATIVO)
        frames.append(p)
    comp = frames[0] if frames else pd.DataFrame(columns=KEY_COMPARATIVO)
    comp = force_key_types(comp, KEY_COMPARATIVO)

    if provision is not None and not provision.empty:
        pr = provision.groupby(KEY_COMPARATIVO, dropna=False).agg(
            cantidad_provisionada=("cantidad_provisionada", "sum"),
            valor_provisionado=("valor_provisionado", "sum"),
        ).reset_index()
        pr = force_key_types(pr, KEY_COMPARATIVO)
        comp = comp.merge(pr, on=KEY_COMPARATIVO, how="outer")
    if proyeccion is not None and not proyeccion.empty:
        py = proyeccion.groupby(KEY_COMPARATIVO, dropna=False).agg(
            cantidad_proyectada=("cantidad_proyectada", "sum"),
            valor_proyectado=("valor_proyectado", "sum"),
        ).reset_index()
        py = force_key_types(py, KEY_COMPARATIVO)
        comp = comp.merge(py, on=KEY_COMPARATIVO, how="outer")
    if hc is not None and not hc.empty:
        hc = force_key_types(hc, KEY_HC)
        comp = force_key_types(comp, KEY_COMPARATIVO)
        comp = comp.merge(hc, on=KEY_HC, how="left")

    for c in ["cantidad_pagada", "valor_pagado", "cantidad_provisionada", "valor_provisionado", "cantidad_proyectada", "valor_proyectado", "hc"]:
        if c not in comp.columns:
            comp[c] = 0.0
        comp[c] = to_num(comp[c])
    comp["dif_valor_pagado_vs_provision"] = comp["valor_pagado"] - comp["valor_provisionado"]
    comp["dif_valor_pagado_vs_proyeccion"] = comp["valor_pagado"] - comp["valor_proyectado"]
    comp["dif_cant_pagada_vs_provision"] = comp["cantidad_pagada"] - comp["cantidad_provisionada"]
    comp["dif_cant_pagada_vs_proyeccion"] = comp["cantidad_pagada"] - comp["cantidad_proyectada"]
    comp["pct_desv_provision"] = np.where(comp["valor_provisionado"].abs() > 0, comp["dif_valor_pagado_vs_provision"] / comp["valor_provisionado"], np.nan)
    comp["pct_desv_proyeccion"] = np.where(comp["valor_proyectado"].abs() > 0, comp["dif_valor_pagado_vs_proyeccion"] / comp["valor_proyectado"], np.nan)
    if comp.empty:
        return comp
    comp["orden_periodo"] = comp["periodo_novedad"].map(period_sort_key)
    comp = comp.sort_values(["orden_periodo", "area_negocio", "cargo_homologado", "concepto"], na_position="last").drop(columns=["orden_periodo"], errors="ignore")
    return comp


# =============================================================
# PREDICCIÓN
# =============================================================
def process_md_actual(file_md, hom, jornada_default: float = 220.0) -> Tuple[pd.DataFrame, List[str]]:
    """Procesa el MD actual para predicción.

    Soporta dos formatos:
    1) TXT SAP completo con varios renglones por persona/concepto.
       Aplica la misma lógica del consolidador de salario vigente:
       - Si la persona tiene Hasta = 31.12.9999, se usan SOLO esos registros vigentes.
       - Si no tiene vigencia abierta, se usa la fecha máxima de Hasta.
       - Dentro de esa vigencia, para cada SAP + concepto, toma el registro más reciente
         ordenando por Modif.el desc y luego Importe desc.
       - Salario total = conceptos base + bonos configurados en CONCEPTOS_SALARIO_MD.
    2) Excel ya consolidado con columna Salario total.
    """
    alerts = []
    if file_md is None:
        return pd.DataFrame(), ["No se cargó MD actual. No se podrá calcular costo por salario."]
    try:
        df = read_table(file_md)
        df.columns = [str(c).strip() for c in df.columns]

        col_sap = find_col(df, ["Nº pers.", "N° pers.", "SAP", "Número de personal"], required=True)
        col_status = find_col(df, ["Status ocupación", "Status ocupacion", "Estado"], required=False)
        col_ceco = find_col(df, ["Ce.coste", "CECO"], required=False)
        col_func = find_col(df, ["Función", "Funcion"], required=False)
        col_cargo = find_col(df, ["Función.1", "Funcion.1", "Posición.1", "Posicion.1", "Cargo", "Denominación función", "Denominacion funcion"], required=False)
        col_area_nom = find_col(df, ["Área de nómina", "Area de nomina"], required=False)
        col_area_personal = find_col(df, ["Área de personal", "Area de personal"], required=False)
        col_div = find_col(df, ["División de personal", "Division de personal"], required=False)
        col_sal_total = find_col(df, ["Salario total", "Total Salario", "Salario Total", "Salario_Total", "salario_total"], required=False)
        col_sal = find_col(df, ["Importe", "Salario", "Sueldo", "Sueldo Básico", "Sueldo Basico"], required=False)
        col_concepto = find_col(df, ["CC-nómina", "CC-nomina", "Concepto", "CC-n."], required=False)
        col_desde = find_col(df, ["Desde"], required=False)
        col_hasta = find_col(df, ["Hasta"], required=False)
        col_modif = find_col(df, ["Modif.el", "Modif el", "Modificado el", "Fecha modificación", "Fecha modificacion"], required=False)
        col_jornada = find_col(df, ["Jornada mensual", "Horas mes", "H mes", "H mensual", "Jornada", "H sem."], required=False)

        work = df.copy()
        work["_sap"] = work[col_sap].map(clean_code)
        work = work[work["_sap"].astype(str).str.len() > 0].copy()

        if col_status:
            work = work[work[col_status].astype(str).str.upper().str.contains("ACTIVO", na=False)].copy()

        if col_desde and col_desde in work.columns:
            work["_desde_dt"] = parse_sap_date_series(work[col_desde])
        else:
            work["_desde_dt"] = pd.NaT
        if col_hasta and col_hasta in work.columns:
            work["_hasta_dt"] = parse_sap_date_series(work[col_hasta])
            work["_hasta_txt"] = work[col_hasta].astype(str).str.strip()
            work["_is_open"] = work["_hasta_txt"].eq("31.12.9999")
        else:
            work["_hasta_dt"] = pd.NaT
            work["_is_open"] = False
        if col_modif and col_modif in work.columns:
            work["_modif_dt"] = parse_sap_date_series(work[col_modif])
        else:
            work["_modif_dt"] = pd.NaT

        # Vigencia objetivo por persona, NO global:
        # Si una persona tiene 31.12.9999, se queda con esa vigencia.
        # Si no, toma su fecha máxima de Hasta.
        if col_hasta and col_hasta in work.columns:
            has_open = work.groupby("_sap")["_is_open"].transform("any")
            max_hasta = work.groupby("_sap")["_hasta_dt"].transform("max")
            keep_vig = (has_open & work["_is_open"]) | ((~has_open) & work["_hasta_dt"].eq(max_hasta))
            work_vig = work[keep_vig].copy()
            activos_abiertos = int(work_vig["_is_open"].any())
            alerts.append("MD actual: apliqué vigencia por persona. Si existe Hasta = 31.12.9999 uso esa vigencia; si no existe, uso la fecha máxima de Hasta.")
        else:
            work_vig = work.copy()
            alerts.append("MD actual: no encontré columna Hasta; no pude aplicar filtro de vigencia por persona.")

        # Excluye Manager I, II, III, IV para headcount/predicción.
        check_cols = [c for c in [col_area_personal, col_area_nom, col_cargo] if c and c in work_vig.columns]
        if check_cols:
            before = len(work_vig)
            work_vig = work_vig.loc[~work_vig[check_cols].apply(is_manager_excluido, axis=1)].copy()
            excl = before - len(work_vig)
            if excl:
                alerts.append(f"MD actual: se excluyeron {excl:,.0f} renglones Manager I-IV del universo de predicción.")

        # Metadata: una fila por SAP. También toma la más reciente por Modif.el.
        meta_sort = ["_hasta_dt", "_desde_dt", "_modif_dt"]
        meta = work_vig.sort_values(meta_sort).drop_duplicates("_sap", keep="last").copy()

        out = pd.DataFrame()
        out["sap"] = meta["_sap"].map(clean_code)
        out["ceco"] = meta[col_ceco].map(clean_ceco) if col_ceco else ""
        out["funcion"] = meta[col_func].map(clean_code) if col_func else ""
        out["cargo_original"] = meta[col_cargo].astype(str) if col_cargo else ""
        out["area_nomina"] = meta[col_area_nom].astype(str) if col_area_nom else ""
        div_vals = meta[col_div].astype(str).values if col_div else [""] * len(out)
        homol = [homologar_cargo(fun, car, hom) for fun, car in zip(out["funcion"], out["cargo_original"])]
        out["cargo_homologado"] = [x[0] for x in homol]
        out["cargo_homologado_ok"] = [x[1] for x in homol]
        out["area_negocio"] = [classify_area(c, None, d, a, None) for c, d, a in zip(out["ceco"], div_vals, out["area_nomina"])]

        # Salario Total:
        # 1) Si ya viene una columna Salario Total, se usa directo.
        # 2) Si viene SAP plano con CC-nómina + Importe, se suma el último registro vigente
        #    por SAP + concepto, ordenado por Modif.el desc y luego Importe desc.
        if col_sal_total and col_sal_total in meta.columns:
            sal_df = meta[["_sap", col_sal_total]].copy()
            sal_df["sap"] = sal_df["_sap"].map(clean_code)
            sal_df["salario"] = to_num(sal_df[col_sal_total]).values
            out = out.merge(sal_df[["sap", "salario"]].drop_duplicates("sap"), on="sap", how="left")
            alerts.append("MD actual: usé la columna 'Salario total' del archivo cargado.")
        elif col_concepto and col_sal and col_concepto in work_vig.columns and col_sal in work_vig.columns:
            sal_work = work_vig.copy()
            sal_work["_concepto_sal"] = sal_work[col_concepto].astype(str).str.strip().str.upper()
            sal_work["_importe_sal"] = to_num(sal_work[col_sal]).values
            sal_work = sal_work[sal_work["_concepto_sal"].isin(CONCEPTOS_SALARIO_MD)].copy()
            if sal_work.empty:
                out["salario"] = 0.0
                alerts.append(f"MD actual: no encontré conceptos salariales {CONCEPTOS_SALARIO_MD}. Valor hora quedará en 0.")
            else:
                # Esta es la regla crítica: último por SAP + concepto dentro de la vigencia objetivo.
                sal_work = sal_work.sort_values(
                    ["_sap", "_concepto_sal", "_modif_dt", "_importe_sal"],
                    ascending=[True, True, False, False],
                    na_position="last",
                )
                ultimo_concepto = sal_work.drop_duplicates(["_sap", "_concepto_sal"], keep="first").copy()
                sal_total = ultimo_concepto.groupby("_sap", dropna=False).agg(salario=("_importe_sal", "sum")).reset_index().rename(columns={"_sap": "sap"})
                out = out.merge(sal_total, on="sap", how="left")
                alerts.append("MD actual: calculé Salario total con el último registro vigente por SAP + concepto, ordenado por Modif.el desc e Importe desc.")
        elif col_sal and col_sal in meta.columns:
            out["salario"] = to_num(meta[col_sal]).values
            alerts.append("MD actual: no encontré 'Salario total'; usé la columna de salario/importe disponible en la fila única.")
        else:
            out["salario"] = 0.0
            alerts.append("MD actual: no encontré salario ni conceptos salariales. Valor hora quedará en 0.")

        out["salario"] = to_num(out["salario"]).values

        # Jornada vigente: si viene H sem., se multiplica por 5 para llevarla a horas mes (44 -> 220).
        if col_jornada and col_jornada in meta.columns:
            jornada = to_num(meta[col_jornada]).values
            if "H SEM" in norm_text(col_jornada):
                jornada = jornada * 5
            jornada = np.where(jornada > 0, jornada, jornada_default)
        else:
            jornada = np.repeat(jornada_default, len(out))
            alerts.append(f"MD actual: no encontré jornada. Se usará {jornada_default:,.0f} horas por defecto.")

        out["jornada"] = jornada
        out["valor_hora"] = np.where(out["jornada"] > 0, out["salario"] / out["jornada"], 0.0)

        sin_sal = int((out["salario"].fillna(0) <= 0).sum())
        if sin_sal:
            alerts.append(f"MD actual: {sin_sal:,.0f} empleados quedaron sin salario total o con salario 0.")
        return out, alerts
    except Exception as e:
        return pd.DataFrame(), [f"Error procesando MD actual: {e}"]

def process_interfaces(files, md_actual, hom) -> Tuple[pd.DataFrame, List[str]]:
    alerts, rows = [], []
    if not files:
        return pd.DataFrame(), ["No se cargaron interfaces. La predicción se hará solo con histórico/proyección si existen."]
    md_cols = ["sap", "ceco", "cargo_homologado", "area_negocio", "area_nomina", "valor_hora", "salario"]
    md_small = md_actual[md_cols].drop_duplicates("sap") if md_actual is not None and not md_actual.empty else pd.DataFrame(columns=md_cols)
    for f in files:
        name = uploaded_name(f)
        try:
            df = read_table(f, no_header=True)
            df = df.dropna(how="all")
            # Si parece tener encabezado, intentamos detectarlo.
            if df.shape[1] < 4:
                df = read_table(f)
            if df.shape[1] >= 4:
                tmp = df.iloc[:, :4].copy()
                tmp.columns = ["sap", "fecha_pago", "concepto_interface", "cantidad_interface"]
            else:
                raise ValueError("La interfaz debe traer mínimo 4 columnas: SAP, fecha pago, concepto, cantidad.")
            out = pd.DataFrame()
            out["source_file"] = name
            out["sap"] = tmp["sap"].map(clean_code)
            out["periodo_pago"] = fill_period_series(tmp["fecha_pago"].map(lambda x: period_from_any(x, name)), period_from_any(None, name))
            out["periodo_novedad"] = out["periodo_pago"].map(lambda p: shift_period(p, -1))
            out["concepto_interface"] = tmp["concepto_interface"].map(lambda x: clean_code(x).upper())
            out["concepto"] = out["concepto_interface"].map(INTERFAZ_A_CCNOMINA).fillna(out["concepto_interface"])
            out = out[out["concepto"].isin(CONCEPTOS)].copy()
            idx = out.index
            out["cantidad_interface"] = to_num(tmp.loc[idx, "cantidad_interface"]).values
            out["tipo_hora"] = out["concepto"].map(lambda c: tipo_hora(c, hom))
            out = out.merge(md_small, on="sap", how="left")
            out["ceco"] = out["ceco"].fillna("")
            out["cargo_homologado"] = out["cargo_homologado"].fillna("Sin MD actual")
            out["area_negocio"] = out["area_negocio"].fillna("Sin MD actual")
            rows.append(out)
            sin_md = out["valor_hora"].isna().sum() if "valor_hora" in out.columns else len(out)
            if sin_md:
                alerts.append(f"{name}: {sin_md:,.0f} registros de interface no cruzaron con MD actual.")
        except Exception as e:
            alerts.append(f"Error procesando interface {name}: {e}")
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), alerts


def prepare_calendar_by_period(periods: List[str]) -> pd.DataFrame:
    rows = []
    for p in sorted([x for x in set(periods) if isinstance(x, str) and "." in x], key=period_sort_key):
        info = calendar_suggestion(p)
        rows.append({"periodo_novedad": p, **info})
    return pd.DataFrame(rows)


def prediccion_mes(
    mes_pred: str,
    comparativo: pd.DataFrame,
    interfaces: pd.DataFrame,
    hc_actual: pd.DataFrame,
    md_actual: pd.DataFrame,
    proyeccion: pd.DataFrame,
    factores: pd.DataFrame,
    cuentas: pd.DataFrame,
    pesos: Dict[str, float],
    dias_pred: int,
    domingos_pred: int,
    festivos_pred: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    alerts = []
    if comparativo is None or comparativo.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ["No hay comparativo histórico procesado para generar predicción."]

    # Headcount actual: si existe HC del mes seleccionado en históricos, se usa; si no, se calcula desde MD actual.
    if hc_actual is not None and not hc_actual.empty:
        hc_actual = hc_actual.rename(columns={"hc": "hc_actual"})
        if "periodo_novedad" in hc_actual.columns:
            hc_actual = hc_actual[hc_actual["periodo_novedad"].eq(mes_pred)].drop(columns=["periodo_novedad"], errors="ignore")
        if not hc_actual.empty:
            hc_actual = hc_actual.groupby(["cargo_homologado", "area_negocio"], dropna=False).agg(hc_actual=("hc_actual", "sum")).reset_index()
    if hc_actual is None or hc_actual.empty:
        if md_actual is not None and not md_actual.empty:
            hc_actual = md_actual.groupby(["cargo_homologado", "area_negocio"], dropna=False).agg(hc_actual=("sap", "nunique")).reset_index()
            alerts.append("No encontré HC procesado para el mes seleccionado; usé el MD actual como HC base de predicción.")
        else:
            hc_actual = pd.DataFrame(columns=["cargo_homologado", "area_negocio", "hc_actual"])
            alerts.append("No hay HC actual ni MD actual suficiente. La predicción de cantidades puede quedar en cero.")

    # Métricas históricas de pagado.
    hist = comparativo[comparativo["periodo_novedad"].map(period_sort_key) < period_sort_key(mes_pred)].copy()
    if hist.empty:
        alerts.append("No hay historia de pagado anterior al mes a predecir.")
    cal_periods = list(hist["periodo_novedad"].dropna().unique()) + [mes_pred]
    if interfaces is not None and not interfaces.empty:
        cal_periods += list(interfaces["periodo_novedad"].dropna().unique())
    cal = prepare_calendar_by_period(cal_periods)
    hist = hist.merge(cal, on="periodo_novedad", how="left")
    hist["driver"] = hist.apply(lambda r: driver_days(r["concepto"], r.get("dias_mes", 30), r.get("domingos", 4), r.get("festivos", 0)), axis=1)
    hist["rate_pagado"] = np.where((hist["hc"] > 0) & (hist["driver"] > 0), hist["cantidad_pagada"] / hist["hc"] / hist["driver"], np.nan)

    group_cols = ["concepto", "tipo_hora", "cargo_homologado", "area_negocio", "ceco"]
    # Último pago real comparable.
    ultimo_pago = pd.DataFrame()
    if not hist.empty:
        hist_sorted = hist.sort_values("periodo_novedad", key=lambda s: s.map(period_sort_key))
        ultimo_pago = hist_sorted.dropna(subset=["rate_pagado"]).groupby(group_cols, dropna=False).tail(1)
        ultimo_pago = ultimo_pago[group_cols + ["rate_pagado", "periodo_novedad"]].rename(columns={"rate_pagado": "rate_ultimo_pago", "periodo_novedad": "periodo_ultimo_pago"})

    # Promedio ponderado últimos meses: 50/30/20 por defecto interno.
    prom_hist = pd.DataFrame()
    if not hist.empty:
        tmp = hist.dropna(subset=["rate_pagado"]).copy()
        tmp["periodo_orden"] = tmp["periodo_novedad"].map(period_sort_key)
        tmp["rank_recencia"] = tmp.groupby(group_cols, dropna=False)["periodo_orden"].rank(method="first", ascending=False)
        tmp = tmp[tmp["rank_recencia"] <= 3].copy()
        tmp["peso_hist"] = tmp["rank_recencia"].map({1: 0.5, 2: 0.3, 3: 0.2}).fillna(0.0)
        prom_hist = tmp.groupby(group_cols, dropna=False).apply(
            lambda g: pd.Series({"rate_prom_hist": np.average(g["rate_pagado"], weights=g["peso_hist"]) if g["peso_hist"].sum() > 0 else g["rate_pagado"].mean()})
        ).reset_index()

    # Interface más reciente antes del mes a predecir.
    rate_interface = pd.DataFrame()
    if interfaces is not None and not interfaces.empty:
        itf = interfaces[interfaces["periodo_novedad"].map(period_sort_key) < period_sort_key(mes_pred)].copy()
        if not itf.empty:
            # Agregamos por llave y pegamos HC del mismo periodo si existe en comparativo.
            itf_agg = itf.groupby(["periodo_novedad"] + group_cols, dropna=False).agg(cantidad_interface=("cantidad_interface", "sum")).reset_index()
            hcs = comparativo.groupby(["periodo_novedad", "cargo_homologado", "area_negocio"], dropna=False).agg(hc=("hc", "max")).reset_index()
            itf_agg = itf_agg.merge(hcs, on=["periodo_novedad", "cargo_homologado", "area_negocio"], how="left")
            itf_agg = itf_agg.merge(cal, on="periodo_novedad", how="left")
            itf_agg["driver"] = itf_agg.apply(lambda r: driver_days(r["concepto"], r.get("dias_mes", 30), r.get("domingos", 4), r.get("festivos", 0)), axis=1)
            itf_agg["rate_interface"] = np.where((itf_agg["hc"] > 0) & (itf_agg["driver"] > 0), itf_agg["cantidad_interface"] / itf_agg["hc"] / itf_agg["driver"], np.nan)
            itf_agg = itf_agg.sort_values("periodo_novedad", key=lambda s: s.map(period_sort_key))
            rate_interface = itf_agg.dropna(subset=["rate_interface"]).groupby(group_cols, dropna=False).tail(1)
            rate_interface = rate_interface[group_cols + ["rate_interface", "periodo_novedad"]].rename(columns={"periodo_novedad": "periodo_interface"})
        else:
            alerts.append("Las interfaces cargadas no tienen periodo anterior al mes a predecir.")

    # Proyección del mes como señal.
    rate_proy = pd.DataFrame()
    if proyeccion is not None and not proyeccion.empty:
        pr = proyeccion[proyeccion["periodo_novedad"].eq(mes_pred)].copy()
        if not pr.empty:
            pr = pr.groupby(group_cols, dropna=False).agg(cantidad_proyectada_mes=("cantidad_proyectada", "sum"), valor_proyectado_mes=("valor_proyectado", "sum")).reset_index()
            pr = pr.merge(hc_actual, on=["cargo_homologado", "area_negocio"], how="left")
            pr["driver_pred"] = pr["concepto"].map(lambda c: driver_days(c, dias_pred, domingos_pred, festivos_pred))
            pr["rate_proyeccion"] = np.where((pr["hc_actual"] > 0) & (pr["driver_pred"] > 0), pr["cantidad_proyectada_mes"] / pr["hc_actual"] / pr["driver_pred"], np.nan)
            rate_proy = pr[group_cols + ["rate_proyeccion", "cantidad_proyectada_mes", "valor_proyectado_mes"]]

    # Universo de predicción: HC actual x conceptos que aparezcan históricamente/proyección/interface.
    universe = []
    for df in [ultimo_pago, prom_hist, rate_interface, rate_proy]:
        if df is not None and not df.empty:
            universe.append(df[group_cols].drop_duplicates())
    if universe:
        uni = pd.concat(universe, ignore_index=True).drop_duplicates()
    else:
        uni = pd.DataFrame(columns=group_cols)
    # Aseguramos que cada cargo actual tenga los conceptos observados en la compañía.
    if hc_actual is not None and not hc_actual.empty and not uni.empty:
        conceptos_obs = uni[["concepto", "tipo_hora"]].drop_duplicates()
        cargos = hc_actual[["cargo_homologado", "area_negocio"]].drop_duplicates()
        expanded = cargos.merge(conceptos_obs, how="cross")
        uni = pd.concat([uni, expanded[group_cols]], ignore_index=True).drop_duplicates()

    pred = uni.copy()
    for df in [ultimo_pago, prom_hist, rate_interface, rate_proy]:
        if df is not None and not df.empty:
            pred = pred.merge(df, on=group_cols, how="left")
    pred = pred.merge(hc_actual, on=["cargo_homologado", "area_negocio"], how="left")
    pred["hc_actual"] = to_num(pred.get("hc_actual", 0))
    pred["driver_pred"] = pred["concepto"].map(lambda c: driver_days(c, dias_pred, domingos_pred, festivos_pred))

    # Promedio ponderado de tasas disponibles.
    source_rates = [
        ("rate_interface", pesos.get("interface", 0.4)),
        ("rate_ultimo_pago", pesos.get("ultimo_pago", 0.3)),
        ("rate_prom_hist", pesos.get("prom_hist", 0.2)),
        ("rate_proyeccion", pesos.get("proyeccion", 0.1)),
    ]
    def weighted_rate(row):
        vals, ws = [], []
        for col, w in source_rates:
            v = row.get(col, np.nan)
            if pd.notna(v) and np.isfinite(v) and v >= 0 and w > 0:
                vals.append(v); ws.append(w)
        if not vals or sum(ws) == 0:
            return np.nan
        return float(np.average(vals, weights=ws))
    pred["rate_final"] = pred.apply(weighted_rate, axis=1)
    pred["cantidad_estimada"] = pred["rate_final"] * pred["hc_actual"] * pred["driver_pred"]
    pred["cantidad_estimada"] = pred["cantidad_estimada"].fillna(0.0)

    # Valor hora desde MD actual.
    if md_actual is not None and not md_actual.empty:
        vh_full = md_actual.groupby(["cargo_homologado", "area_negocio", "ceco"], dropna=False).agg(valor_hora_ref=("valor_hora", "mean")).reset_index()
        vh_cargo = md_actual.groupby(["cargo_homologado", "area_negocio"], dropna=False).agg(valor_hora_ref_cargo=("valor_hora", "mean")).reset_index()
        vh_area = md_actual.groupby(["area_negocio"], dropna=False).agg(valor_hora_ref_area=("valor_hora", "mean")).reset_index()
        global_vh = float(md_actual["valor_hora"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if "valor_hora" in md_actual else 0.0
        pred = pred.merge(vh_full, on=["cargo_homologado", "area_negocio", "ceco"], how="left")
        pred = pred.merge(vh_cargo, on=["cargo_homologado", "area_negocio"], how="left")
        pred = pred.merge(vh_area, on=["area_negocio"], how="left")
        pred["valor_hora_ref"] = pred["valor_hora_ref"].fillna(pred["valor_hora_ref_cargo"]).fillna(pred["valor_hora_ref_area"]).fillna(global_vh)
        pred = pred.drop(columns=["valor_hora_ref_cargo", "valor_hora_ref_area"], errors="ignore")
    else:
        pred["valor_hora_ref"] = 0.0

    factores = factores.copy()
    factores["concepto"] = factores["concepto"].astype(str).str.upper().str.strip()
    factores["factor"] = to_num(factores["factor"])
    pred = pred.merge(factores[["concepto", "factor"]].drop_duplicates("concepto"), on="concepto", how="left")
    pred["factor"] = pred["factor"].fillna(0.0)
    pred["valor_estimado"] = pred["cantidad_estimada"] * pred["valor_hora_ref"] * pred["factor"]
    pred["mes_estimado"] = mes_pred
    pred["mes_pago_estimado"] = shift_period(mes_pred, 1)
    pred["dif_estimado_vs_proyeccion"] = pred["valor_estimado"] - pred.get("valor_proyectado_mes", 0).fillna(0.0)
    pred["cantidad_proyectada_mes"] = pred.get("cantidad_proyectada_mes", 0).fillna(0.0)
    pred["valor_proyectado_mes"] = pred.get("valor_proyectado_mes", 0).fillna(0.0)

    pred = map_cuentas(pred, cuentas)
    resumen_cuentas = pred.groupby(["mes_estimado", "cuenta", "descripcion_cuenta", "area_negocio", "concepto", "tipo_hora"], dropna=False).agg(
        cantidad_estimada=("cantidad_estimada", "sum"),
        valor_estimado=("valor_estimado", "sum"),
    ).reset_index()
    resumen = pred.groupby(["mes_estimado", "concepto", "tipo_hora", "cargo_homologado", "area_negocio"], dropna=False).agg(
        hc_actual=("hc_actual", "max"),
        cantidad_estimada=("cantidad_estimada", "sum"),
        valor_estimado=("valor_estimado", "sum"),
        valor_proyectado_mes=("valor_proyectado_mes", "sum"),
        dif_estimado_vs_proyeccion=("dif_estimado_vs_proyeccion", "sum"),
    ).reset_index()

    if pred["cuenta"].eq("Sin cuenta").any():
        alerts.append(f"Predicción: {pred['cuenta'].eq('Sin cuenta').sum():,.0f} filas quedaron sin cuenta contable asignada.")
    if pred["valor_hora_ref"].eq(0).any():
        alerts.append(f"Predicción: {pred['valor_hora_ref'].eq(0).sum():,.0f} filas tienen valor hora 0. Revisa salario/MD actual.")
    return pred, resumen, resumen_cuentas, alerts


def map_cuentas(df: pd.DataFrame, cuentas: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cuenta"] = "Sin cuenta"
    out["descripcion_cuenta"] = "Sin cuenta"
    if cuentas is None or cuentas.empty:
        return out
    cta = cuentas.copy()
    cta.columns = [str(c).strip().lower() for c in cta.columns]
    col_con = find_col(cta, ["concepto"], False) or "concepto"
    col_area = find_col(cta, ["area_negocio", "área negocio", "area"], False) or "area_negocio"
    col_pref = find_col(cta, ["prefijo_ceco", "ceco_prefijo", "prefijo", "ceco"], False) or "prefijo_ceco"
    col_cuenta = find_col(cta, ["cuenta"], False) or "cuenta"
    col_desc = find_col(cta, ["descripcion_cuenta", "descripción cuenta", "descripcion"], False) or "descripcion_cuenta"
    for col in [col_con, col_area, col_pref, col_cuenta, col_desc]:
        if col not in cta.columns:
            cta[col] = ""
    cta[col_con] = cta[col_con].astype(str).str.upper().str.strip()
    cta[col_area] = cta[col_area].astype(str).str.strip()
    cta[col_pref] = cta[col_pref].astype(str).str.strip()
    cta = cta[cta[col_cuenta].astype(str).str.strip().ne("")].copy()
    # Aplicación fila a fila: primero concepto + prefijo ceco + área; luego concepto + prefijo; luego concepto + área; luego concepto.
    def pick(row):
        c = str(row.get("concepto", "")).upper().strip()
        area = str(row.get("area_negocio", "")).strip()
        ceco = clean_ceco(row.get("ceco", ""))
        candidates = cta[cta[col_con].eq(c)].copy()
        if candidates.empty:
            return pd.Series({"cuenta": "Sin cuenta", "descripcion_cuenta": "Sin cuenta"})
        candidates["score"] = 0
        candidates.loc[candidates[col_area].eq(area), "score"] += 2
        candidates.loc[candidates[col_pref].apply(lambda p: bool(p) and ceco.startswith(clean_ceco(p))), "score"] += 3
        candidates = candidates.sort_values("score", ascending=False)
        r = candidates.iloc[0]
        if r["score"] == 0 and (str(r[col_area]).strip() or str(r[col_pref]).strip()):
            return pd.Series({"cuenta": "Sin cuenta", "descripcion_cuenta": "Sin cuenta"})
        return pd.Series({"cuenta": str(r[col_cuenta]).strip(), "descripcion_cuenta": str(r[col_desc]).strip()})
    mapped = out.apply(pick, axis=1)
    out["cuenta"] = mapped["cuenta"]
    out["descripcion_cuenta"] = mapped["descripcion_cuenta"]
    return out

# =============================================================
# EXPORTACIÓN A EXCEL
# =============================================================
def add_table_format(writer, sheet_name: str, df: pd.DataFrame, table_name: str):
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    if df.empty:
        worksheet.write(0, 0, "Sin datos")
        return
    max_row, max_col = df.shape
    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, max_row, max_col - 1)
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#F4B183", "font_color": "#1F1F1F", "border": 1})
    for col_num, value in enumerate(df.columns):
        worksheet.write(0, col_num, value, header_fmt)
        width = min(max(len(str(value)) + 2, 12), 35)
        try:
            sample = df[value].astype(str).head(500).map(len).max()
            width = min(max(width, int(sample) + 2), 45)
        except Exception:
            pass
        worksheet.set_column(col_num, col_num, width)
    money_cols = [i for i, c in enumerate(df.columns) if "valor" in c.lower() or "importe" in c.lower() or "provisión" in c.lower() or "provision" in c.lower()]
    qty_cols = [i for i, c in enumerate(df.columns) if "cantidad" in c.lower() or c.lower().startswith("hc") or "factor" in c.lower()]
    money_fmt = workbook.add_format({"num_format": "$ #,##0"})
    num_fmt = workbook.add_format({"num_format": "#,##0.00"})
    for i in money_cols:
        worksheet.set_column(i, i, 16, money_fmt)
    for i in qty_cols:
        worksheet.set_column(i, i, 14, num_fmt)


def to_excel_bytes(sheets: Dict[str, pd.DataFrame], include_chart: bool = True) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet, df in sheets.items():
            safe_sheet = sheet[:31]
            dfx = df.copy()
            if len(dfx) > 1_048_000:
                dfx = dfx.head(1_048_000)
            dfx.to_excel(writer, index=False, sheet_name=safe_sheet)
            add_table_format(writer, safe_sheet, dfx, f"tbl_{safe_sheet.replace(' ', '_')[:20]}")
        if include_chart and "Resumen_mes" in sheets and not sheets["Resumen_mes"].empty:
            workbook = writer.book
            ws = writer.sheets["Resumen_mes"]
            df = sheets["Resumen_mes"]
            cols = list(df.columns)
            if all(c in cols for c in ["periodo_novedad", "valor_pagado", "valor_provisionado", "valor_proyectado"]):
                chart = workbook.add_chart({"type": "line"})
                row_count = len(df)
                for col_name in ["valor_pagado", "valor_provisionado", "valor_proyectado"]:
                    col_idx = cols.index(col_name)
                    chart.add_series({
                        "name": ["Resumen_mes", 0, col_idx],
                        "categories": ["Resumen_mes", 1, cols.index("periodo_novedad"), row_count, cols.index("periodo_novedad")],
                        "values": ["Resumen_mes", 1, col_idx, row_count, col_idx],
                    })
                chart.set_title({"name": "Pagado vs provisión vs proyección"})
                chart.set_x_axis({"name": "Mes novedad"})
                chart.set_y_axis({"name": "Valor"})
                ws.insert_chart("J2", chart, {"x_scale": 1.5, "y_scale": 1.4})
    output.seek(0)
    return output.getvalue()


def _add_indicadores_hc(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas por HC evitando división por cero."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "hc" not in out.columns:
        out["hc"] = 0.0
    hc = pd.to_numeric(out["hc"], errors="coerce").fillna(0)
    for c in ["cantidad_pagada", "cantidad_provisionada", "cantidad_proyectada", "valor_pagado", "valor_provisionado", "valor_proyectado"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    out["horas_pagadas_por_hc"] = np.where(hc > 0, out["cantidad_pagada"] / hc, 0)
    out["horas_provisionadas_por_hc"] = np.where(hc > 0, out["cantidad_provisionada"] / hc, 0)
    out["horas_proyectadas_por_hc"] = np.where(hc > 0, out["cantidad_proyectada"] / hc, 0)
    out["valor_pagado_por_hc"] = np.where(hc > 0, out["valor_pagado"] / hc, 0)
    out["valor_provisionado_por_hc"] = np.where(hc > 0, out["valor_provisionado"] / hc, 0)
    out["valor_proyectado_por_hc"] = np.where(hc > 0, out["valor_proyectado"] / hc, 0)
    return out

def _sin_ceros(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = [c for c in ["valor_pagado", "valor_provisionado", "valor_proyectado", "cantidad_pagada", "cantidad_provisionada", "cantidad_proyectada"] if c in df.columns]
    if not cols:
        return df
    mask = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).abs().sum(axis=1) > 0
    return df.loc[mask].copy()

def resumenes_comparativo(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {}

    work = df.copy()
    for c in ["valor_pagado", "valor_provisionado", "valor_proyectado", "cantidad_pagada", "cantidad_provisionada", "cantidad_proyectada", "hc"]:
        if c not in work.columns:
            work[c] = 0.0
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)

    def sort_cols(d: pd.DataFrame) -> pd.DataFrame:
        if d is None or d.empty or "periodo_novedad" not in d.columns:
            return d
        d = d.copy()
        d["orden"] = d["periodo_novedad"].map(period_sort_key)
        sort_by = ["orden"] + [c for c in ["area_negocio", "cargo_homologado", "concepto", "tipo_hora"] if c in d.columns]
        return d.sort_values(sort_by).drop(columns="orden")

    def agg_by(cols, hc_mode="max"):
        agg = {
            "valor_pagado": ("valor_pagado", "sum"),
            "valor_provisionado": ("valor_provisionado", "sum"),
            "valor_proyectado": ("valor_proyectado", "sum"),
            "dif_valor_pagado_vs_provision": ("dif_valor_pagado_vs_provision", "sum"),
            "dif_valor_pagado_vs_proyeccion": ("dif_valor_pagado_vs_proyeccion", "sum"),
            "cantidad_pagada": ("cantidad_pagada", "sum"),
            "cantidad_provisionada": ("cantidad_provisionada", "sum"),
            "cantidad_proyectada": ("cantidad_proyectada", "sum"),
        }
        # El HC está asignado a varias filas por concepto; se usa max para no duplicarlo dentro de la misma granularidad.
        if "hc" in work.columns:
            agg["hc"] = ("hc", "max" if hc_mode == "max" else "sum")
        out = work.groupby(cols, dropna=False).agg(**agg).reset_index()
        return sort_cols(_add_indicadores_hc(out))

    res_mes = agg_by(["periodo_novedad"], hc_mode="sum")
    res_concepto = agg_by(["periodo_novedad", "concepto", "tipo_hora"])

    # V8.1: resumen gerencial sin CECO, con Área negocio + Cargo homologado + Concepto + Tipo hora.
    resumen_ejecutivo = agg_by(["periodo_novedad", "area_negocio", "cargo_homologado", "concepto", "tipo_hora"])
    resumen_ejecutivo_sin_ceros = _sin_ceros(resumen_ejecutivo)

    resumen_cargo = agg_by(["periodo_novedad", "area_negocio", "cargo_homologado"])
    indicadores_hc = resumen_cargo[[c for c in [
        "periodo_novedad", "area_negocio", "cargo_homologado", "hc",
        "cantidad_pagada", "cantidad_provisionada", "cantidad_proyectada",
        "horas_pagadas_por_hc", "horas_provisionadas_por_hc", "horas_proyectadas_por_hc",
        "valor_pagado_por_hc", "valor_provisionado_por_hc", "valor_proyectado_por_hc"
    ] if c in resumen_cargo.columns]].copy()

    return {
        "Resumen_mes": res_mes,
        "Resumen_concepto": res_concepto,
        "Resumen_Ejecutivo": resumen_ejecutivo,
        "Resumen_Ejec_Sin_Ceros": resumen_ejecutivo_sin_ceros,
        "Resumen_Cargo_Homologado": resumen_cargo,
        "Indicadores_HC": indicadores_hc,
        "Resumen_area": agg_by(["periodo_novedad", "area_negocio"]),
    }


def build_alertas_comparativo(comp: pd.DataFrame, alerts_base: List[str], threshold_pct: float = 0.15) -> pd.DataFrame:
    rows = [{"tipo": "Cargue", "alerta": a} for a in alerts_base]
    if comp is None or comp.empty:
        return standard_alert_df(rows + [{"tipo": "Comparativo", "alerta": "No hay comparativo generado."}])
    # Pagado sin provisión/proyección, provisión sin pago, proyección sin pago.
    paid = comp["valor_pagado"].abs() > 0
    prov = comp["valor_provisionado"].abs() > 0
    proy = comp["valor_proyectado"].abs() > 0
    counts = {
        "Pagado sin provisión": int((paid & ~prov).sum()),
        "Pagado sin proyección": int((paid & ~proy).sum()),
        "Provisión sin pagado": int((prov & ~paid).sum()),
        "Proyección sin pagado": int((proy & ~paid).sum()),
    }
    for k, v in counts.items():
        if v:
            rows.append({"tipo": "Comparativo", "alerta": f"{k}: {v:,.0f} combinaciones."})
    desv = comp[(comp["pct_desv_provision"].abs() > threshold_pct) & prov]
    if not desv.empty:
        rows.append({"tipo": "Desviación", "alerta": f"{len(desv):,.0f} combinaciones superan {threshold_pct:.0%} de desviación pagado vs provisión."})
    return standard_alert_df(rows)



def filtro_multiseleccion(label: str, opciones: List[str], key: str, ayuda: str = "") -> List[str]:
    """Selector liviano: vacío significa todos. Evita cargar la pantalla con todos los chips seleccionados."""
    opciones = list(opciones or [])
    seleccion = st.multiselect(
        f"{label} (vacío = todos)",
        opciones,
        default=[],
        key=key,
        help=ayuda or "Deja este campo vacío para incluir todos los valores. Selecciona uno o varios para filtrar."
    )
    st.markdown(
        f"<div class='filter-note'>{'Todos seleccionados' if not seleccion else str(len(seleccion)) + ' seleccionado(s)'}</div>",
        unsafe_allow_html=True,
    )
    return seleccion if seleccion else opciones


def aplicar_filtros_comparativo(comp: pd.DataFrame, f_periodos, f_conceptos, f_areas, f_tipos, f_cargos) -> pd.DataFrame:
    filt = comp[
        comp["periodo_novedad"].isin(f_periodos)
        & comp["concepto"].isin(f_conceptos)
        & comp["area_negocio"].isin(f_areas)
        & comp["tipo_hora"].isin(f_tipos)
        & comp["cargo_homologado"].isin(f_cargos)
    ].copy()
    return filt

# =============================================================
# UI
# =============================================================
st.markdown(
    """
    <style>
    :root {
        --jmc-orange: #F05A28;
        --jmc-orange-dark: #A74412;
        --jmc-soft: #FFF3EC;
        --jmc-border: #FFD7C2;
    }
    .main .block-container {padding-top: 1.1rem;}
    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    .small-note {font-size: 0.85rem; color: #666;}
    .brand-header {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        background: linear-gradient(90deg, var(--jmc-soft), #FFFFFF);
        border-left: 8px solid var(--jmc-orange);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 8px rgba(240, 90, 40, 0.08);
    }
    .brand-logo {
        width: 3.2rem;
        height: 3.2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: #FFFFFF;
        border: 2px solid var(--jmc-border);
        font-size: 2rem;
    }
    .brand-title {
        font-size: 2rem;
        line-height: 1.1;
        font-weight: 750;
        color: #1F2937;
        margin: 0;
    }
    .brand-subtitle {
        color: #6B7280;
        margin-top: 0.35rem;
        font-size: 0.98rem;
    }
    div.stButton > button:first-child,
    div.stDownloadButton > button:first-child {
        border-color: var(--jmc-orange);
    }
    div.stButton > button[kind="primary"] {
        background: var(--jmc-orange);
        border-color: var(--jmc-orange);
    }
    .filter-note {
        color: #6B7280;
        font-size: 0.86rem;
        margin-top: -0.35rem;
        margin-bottom: 0.45rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="brand-header">
        <div class="brand-logo">🦜</div>
        <div>
            <div class="brand-title">Comparativo y predicción de horas de nómina</div>
            <div class="brand-subtitle">Pagado real vs provisión vs proyección + predicción del mes en curso por concepto, tipo hora, cargo homologado, CECO y área.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "processed" not in st.session_state:
    st.session_state.processed = False

with st.sidebar:
    st.header("Menú")
    menu = st.radio(
        "Selecciona módulo",
        ["1. Cargue y procesamiento", "2. Comparativo histórico", "3. Predicción mes en curso", "4. Instructivo", "5. Plantillas"],
        index=0,
    )

if menu == "1. Cargue y procesamiento":
    st.subheader("1. Cargue de archivos")
    st.info("Carga cada grupo de archivos por separado. No necesitas consolidar todo en una sola hoja; la app consolida internamente.")

    col1, col2 = st.columns(2)
    with col1:
        detalle_file = st.file_uploader("Detalle Horas / Homologación (.xlsb)", type=["xlsb", "xlsx", "xlsm"], accept_multiple_files=False)
        cc_files = st.file_uploader("CC-nóminas base plana (cargue múltiple)", type=["xlsx", "xls", "xlsb", "csv", "txt"], accept_multiple_files=True)
        comp_files = st.file_uploader("Compensatorios Y350 (opcional / cargue múltiple)", type=["xls", "xlsx", "csv", "txt"], accept_multiple_files=True)
    with col2:
        prov_files = st.file_uploader("Consolidado / archivos de Provisión", type=["xlsx", "xlsb", "xls", "csv", "txt"], accept_multiple_files=True)
        proy_files = st.file_uploader("Consolidado / archivos de Proyección", type=["xlsx", "xlsb", "xls", "csv", "txt"], accept_multiple_files=True)
        hc_files = st.file_uploader("Headcount mensual (cargue múltiple)", type=["xlsx", "xlsb", "xls", "csv", "txt"], accept_multiple_files=True)

    if st.button("🚀 Procesar bases", type="primary"):
        all_alerts = []
        progress = st.progress(0, text="Leyendo homologación...")
        hom, hom_df, alerts = build_homologacion(detalle_file)
        all_alerts += alerts
        progress.progress(15, text="Procesando CC-nómina...")
        cc_df, alerts = process_ccnomina(cc_files, hom)
        all_alerts += alerts
        progress.progress(30, text="Procesando compensatorios...")
        comp_df, alerts = process_compensatorios(comp_files, hom)
        all_alerts += alerts
        pagado_df = pd.concat([x for x in [cc_df, comp_df] if x is not None and not x.empty], ignore_index=True) if ((cc_df is not None and not cc_df.empty) or (comp_df is not None and not comp_df.empty)) else pd.DataFrame()
        progress.progress(45, text="Procesando Headcount y construyendo maestro Posición → Función...")
        hc_df, excl_hc, hc_detail, alerts = process_headcount(hc_files, hom)
        all_alerts += alerts
        maestro_pos_func = build_maestro_posicion_funcion(hc_detail, hom)
        if maestro_pos_func is not None and not maestro_pos_func.empty:
            all_alerts.append(f"Maestro Posición → Función: {len(maestro_pos_func):,.0f} combinaciones construidas desde Headcount consolidado.")
        else:
            all_alerts.append("Maestro Posición → Función: no se pudo construir; se usará homologación directa por función/cargo original.")

        progress.progress(58, text="Aplicando maestro Posición → Función al pagado real...")
        pagado_df = apply_maestro_posicion_funcion(pagado_df, maestro_pos_func, hom, fuente="Pagado")

        progress.progress(65, text="Procesando provisión...")
        provision_df, alerts = process_provision(prov_files, hom, hc_detail)
        all_alerts += alerts
        provision_df = apply_maestro_posicion_funcion(provision_df, maestro_pos_func, hom, fuente="Provisión")

        progress.progress(78, text="Procesando proyección...")
        proyeccion_df, alerts = process_proyeccion(proy_files, hom, hc_detail)
        all_alerts += alerts
        proyeccion_df = apply_maestro_posicion_funcion(proyeccion_df, maestro_pos_func, hom, fuente="Proyección")

        progress.progress(90, text="Construyendo comparativo...")
        comparativo = aggregate_sources(pagado_df, provision_df, proyeccion_df, hc_df)
        alertas_df = build_alertas_comparativo(comparativo, all_alerts)
        progress.progress(100, text="Listo")

        st.session_state.hom = hom
        st.session_state.hom_df = hom_df
        st.session_state.pagado_df = pagado_df
        st.session_state.provision_df = provision_df
        st.session_state.proyeccion_df = proyeccion_df
        st.session_state.hc_df = hc_df
        st.session_state.hc_detail = hc_detail
        st.session_state.excl_hc = excl_hc
        st.session_state.maestro_pos_func = maestro_pos_func
        st.session_state.comparativo = comparativo
        st.session_state.alertas = alertas_df
        st.session_state.processed = True
        # Limpia filtros/resúmenes anteriores para que el comparativo se regenere con las nuevas bases.
        for _k in ["filt_comparativo", "resumenes_comparativo", "alertas_comparativo_filtrado", "threshold_comparativo"]:
            if _k in st.session_state:
                del st.session_state[_k]
        st.success("Bases procesadas correctamente.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Registros pagado", f"{len(pagado_df):,.0f}")
        m2.metric("Registros provisión", f"{len(provision_df):,.0f}")
        m3.metric("Registros proyección", f"{len(proyeccion_df):,.0f}")
        m4.metric("Llaves comparativo", f"{len(comparativo):,.0f}")

        with st.expander("Ver alertas de procesamiento", expanded=True):
            display_df(alertas_df)

elif menu == "2. Comparativo histórico":
    st.subheader("2. Comparativo histórico")
    if not st.session_state.get("processed", False):
        st.warning("Primero procesa las bases en el módulo 1.")
        st.stop()

    comp = st.session_state.comparativo.copy()
    st.caption("Recuerda: el pagado real se compara por mes de novedad. Ejemplo: pago 05.2026 corresponde a novedad 04.2026.")

    periodos = sorted(comp["periodo_novedad"].dropna().unique(), key=period_sort_key)
    conceptos = sorted(comp["concepto"].dropna().unique())
    areas = sorted(comp["area_negocio"].dropna().unique())
    tipos = sorted(comp["tipo_hora"].dropna().unique())
    cargos = sorted(comp["cargo_homologado"].dropna().unique())

    with st.expander("Filtros", expanded=True):
        st.info("Deja un filtro vacío para tomar todos los valores. Selecciona solo uno o varios cuando quieras acotar el análisis. Los filtros se aplican solo al presionar el botón.")
        with st.form("form_filtros_comparativo"):
            c1, c2, c3 = st.columns(3)
            with c1:
                f_periodos = filtro_multiseleccion("Mes novedad", periodos, "f_periodos")
                f_conceptos = filtro_multiseleccion("Concepto", conceptos, "f_conceptos")
            with c2:
                f_areas = filtro_multiseleccion("Área negocio", areas, "f_areas")
                f_tipos = filtro_multiseleccion("Tipo hora", tipos, "f_tipos")
            with c3:
                f_cargos = filtro_multiseleccion("Cargo homologado", cargos, "f_cargos")
                threshold = st.slider("Umbral alerta desviación", 0.05, 1.0, 0.15, 0.05, key="threshold_comp")
            aplicar = st.form_submit_button("🔎 Aplicar filtros", type="primary")

    if aplicar or "filt_comparativo" not in st.session_state:
        progress_f = st.progress(0, text="Aplicando filtros...")
        filt = aplicar_filtros_comparativo(comp, f_periodos, f_conceptos, f_areas, f_tipos, f_cargos)
        progress_f.progress(55, text="Calculando resúmenes...")
        resumenes_tmp = resumenes_comparativo(filt)
        progress_f.progress(85, text="Generando alertas...")
        alertas_tmp = build_alertas_comparativo(filt, [], threshold)
        progress_f.progress(100, text="Filtros aplicados")
        st.session_state.filt_comparativo = filt
        st.session_state.resumenes_comparativo = resumenes_tmp
        st.session_state.alertas_comparativo_filtrado = alertas_tmp
        st.session_state.threshold_comparativo = threshold
    else:
        filt = st.session_state.filt_comparativo
        threshold = st.session_state.get("threshold_comparativo", 0.15)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Valor pagado", money_fmt(filt['valor_pagado'].sum()))
    k2.metric("Valor provisión", money_fmt(filt['valor_provisionado'].sum()))
    k3.metric("Valor proyección", money_fmt(filt['valor_proyectado'].sum()))
    k4.metric("Dif. pagado vs provisión", money_fmt(filt['dif_valor_pagado_vs_provision'].sum()))

    resumenes = st.session_state.get("resumenes_comparativo", resumenes_comparativo(filt))
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Resumen ejecutivo", "Resúmenes", "Gráficas", "Detalle", "Alertas / Descargar"])
    with tab1:
        st.markdown("#### Resumen ejecutivo")
        st.caption("Granularidad: Mes novedad + Área negocio + Cargo homologado + Concepto + Tipo hora. No incluye CECO para mantener una lectura gerencial.")
        display_df(resumenes.get("Resumen_Ejec_Sin_Ceros", pd.DataFrame()), height=520)
        st.markdown("#### Resumen por cargo homologado")
        display_df(resumenes.get("Resumen_Cargo_Homologado", pd.DataFrame()), height=360)
    with tab2:
        st.markdown("#### Resumen por mes")
        display_df(resumenes.get("Resumen_mes", pd.DataFrame()))
        st.markdown("#### Resumen por concepto")
        display_df(resumenes.get("Resumen_concepto", pd.DataFrame()))
        st.markdown("#### Indicadores HC")
        display_df(resumenes.get("Indicadores_HC", pd.DataFrame()), height=420)
        if st.session_state.get("maestro_pos_func") is not None and not st.session_state.maestro_pos_func.empty:
            st.markdown("#### Maestro Posición → Función (desde Headcount)")
            st.caption("Este maestro se usa antes de comparar para convertir posiciones/cargos de las fuentes a Función y luego a Cargo homologado.")
            display_df(st.session_state.maestro_pos_func.head(500), height=360)
    with tab3:
        rm = resumenes.get("Resumen_mes", pd.DataFrame())
        if not rm.empty:
            fig = px.line(rm, x="periodo_novedad", y=["valor_pagado", "valor_provisionado", "valor_proyectado"], markers=True, title="Valor mensual: pagado vs provisión vs proyección")
            st.plotly_chart(fig, use_container_width=True)
        rc = resumenes.get("Resumen_Cargo_Homologado", pd.DataFrame())
        if not rc.empty:
            top = rc.groupby("cargo_homologado", dropna=False).agg(dif=("dif_valor_pagado_vs_provision", "sum")).reset_index()
            top["abs_dif"] = top["dif"].abs()
            top = top.sort_values("abs_dif", ascending=False).head(20)
            fig2 = px.bar(top, x="dif", y="cargo_homologado", orientation="h", title="Top cargos por desviación pagado vs provisión")
            st.plotly_chart(fig2, use_container_width=True)
    with tab4:
        display_df(filt, height=520)
    with tab5:
        alertas = st.session_state.get("alertas_comparativo_filtrado", build_alertas_comparativo(filt, [], threshold))
        display_df(alertas)
        sheets = {"Detalle_Comparativo": filt, **resumenes, "Comparativo_Sin_Ceros": _sin_ceros(filt), "Alertas": alertas}
        if st.session_state.get("hom_df") is not None and not st.session_state.hom_df.empty:
            sheets["Homologacion_usada"] = st.session_state.hom_df
        if st.session_state.get("excl_hc") is not None and not st.session_state.excl_hc.empty:
            sheets["Excluidos_HC"] = st.session_state.excl_hc
        if st.session_state.get("maestro_pos_func") is not None and not st.session_state.maestro_pos_func.empty:
            sheets["Maestro_Posicion_Funcion"] = st.session_state.maestro_pos_func
        xbytes = to_excel_bytes(sheets, include_chart=True)
        st.download_button("⬇️ Descargar Excel comparativo", data=xbytes, file_name="comparativo_historico_horas.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif menu == "3. Predicción mes en curso":
    st.subheader("3. Predicción del mes en curso")
    if not st.session_state.get("processed", False):
        st.warning("Primero procesa las bases históricas en el módulo 1.")
        st.stop()

    st.info("Este módulo estima las horas del mes seleccionado que se pagarán el mes siguiente. Ejemplo: 06.2026 laborado → pago 07.2026.")
    col1, col2 = st.columns(2)
    with col1:
        mes_default = datetime.today().strftime("%m.%Y")
        mes_pred = st.text_input("Mes a predecir (MM.AAAA)", value=mes_default)
        mes_pred = period_from_any(mes_pred) or mes_default
        sugerido = calendar_suggestion(mes_pred)
        dias_mes = st.number_input("Días calendario del mes", min_value=1, max_value=31, value=int(sugerido["dias_mes"]))
        domingos = st.number_input("Domingos del mes", min_value=0, max_value=6, value=int(sugerido["domingos"]))
        festivos = st.number_input("Festivos del mes", min_value=0, max_value=6, value=int(sugerido["festivos"]))
        jornada_default = st.number_input("Jornada mensual por defecto para valor hora", min_value=1.0, max_value=300.0, value=220.0, step=1.0)
    with col2:
        interfaces_files = st.file_uploader("Interfaces regionales del mes / cargue múltiple", type=["xlsx", "xls", "csv", "txt"], accept_multiple_files=True)
        md_file = st.file_uploader("Master Data del mes actual", type=["xlsx", "xlsb", "xls", "csv", "txt"], accept_multiple_files=False)
        cuentas_file = st.file_uploader("Tabla de cuentas contables (opcional)", type=["xlsx", "csv", "txt"], accept_multiple_files=False)

    st.markdown("#### Pesos de predicción")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        peso_interface = st.number_input("Peso interface", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
    with p2:
        peso_ultimo = st.number_input("Peso último pago", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
    with p3:
        peso_hist = st.number_input("Peso promedio histórico", min_value=0.0, max_value=1.0, value=0.20, step=0.05)
    with p4:
        peso_proy = st.number_input("Peso proyección", min_value=0.0, max_value=1.0, value=0.10, step=0.05)

    st.markdown("#### Factores por concepto")
    factores_edit = st.data_editor(FACTORES_DEFAULT, num_rows="dynamic", use_container_width=True)

    st.markdown("#### Cuentas contables")
    if cuentas_file is not None:
        try:
            cuentas_df = read_table(cuentas_file)
        except Exception as e:
            st.warning(f"No pude leer cuentas cargadas: {e}. Puedes editar la tabla manual.")
            cuentas_df = CUENTAS_TEMPLATE.copy()
    else:
        cuentas_df = CUENTAS_TEMPLATE.copy()
    cuentas_edit = st.data_editor(cuentas_df, num_rows="dynamic", use_container_width=True)

    if st.button("🔮 Generar predicción", type="primary"):
        all_alerts = []
        with st.spinner("Procesando MD actual..."):
            md_actual, alerts = process_md_actual(md_file, st.session_state.hom, jornada_default)
            all_alerts += alerts
        with st.spinner("Procesando interfaces..."):
            interfaces, alerts = process_interfaces(interfaces_files, md_actual, st.session_state.hom)
            all_alerts += alerts
        # HC actual: preferimos HC procesado del mes si existe; si no, MD.
        hc_actual = st.session_state.hc_df.copy() if st.session_state.get("hc_df") is not None else pd.DataFrame()
        pesos = {"interface": peso_interface, "ultimo_pago": peso_ultimo, "prom_hist": peso_hist, "proyeccion": peso_proy}
        with st.spinner("Calculando predicción..."):
            pred, resumen_pred, resumen_cuentas, alerts = prediccion_mes(
                mes_pred=mes_pred,
                comparativo=st.session_state.comparativo,
                interfaces=interfaces,
                hc_actual=hc_actual,
                md_actual=md_actual,
                proyeccion=st.session_state.proyeccion_df,
                factores=factores_edit,
                cuentas=cuentas_edit,
                pesos=pesos,
                dias_pred=dias_mes,
                domingos_pred=domingos,
                festivos_pred=festivos,
            )
            all_alerts += alerts
        alertas_pred = standard_alert_df(all_alerts)
        st.session_state.pred = pred
        st.session_state.resumen_pred = resumen_pred
        st.session_state.resumen_cuentas = resumen_cuentas
        st.session_state.alertas_pred = alertas_pred
        st.session_state.md_actual = md_actual
        st.session_state.interfaces = interfaces

    if "pred" in st.session_state and st.session_state.pred is not None and not st.session_state.pred.empty:
        pred = st.session_state.pred
        rpred = st.session_state.resumen_pred
        rc = st.session_state.resumen_cuentas
        ap = st.session_state.alertas_pred
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Cantidad estimada", f"{pred['cantidad_estimada'].sum():,.2f}")
        k2.metric("Valor estimado", money_fmt(pred['valor_estimado'].sum()))
        k3.metric("Pago estimado", shift_period(mes_pred, 1))
        k4.metric("Filas sin cuenta", f"{pred['cuenta'].eq('Sin cuenta').sum():,.0f}")
        tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Por cuentas", "Detalle", "Alertas / Descargar"])
        with tab1:
            display_df(rpred)
            fig = px.bar(rpred.groupby(["concepto", "tipo_hora"], dropna=False).agg(valor_estimado=("valor_estimado", "sum")).reset_index(), x="concepto", y="valor_estimado", color="tipo_hora", title="Valor estimado por concepto")
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            display_df(rc)
        with tab3:
            display_df(pred, height=520)
        with tab4:
            display_df(ap)
            sheets = {
                "Prediccion_detalle": pred,
                "Prediccion_resumen": rpred,
                "Prediccion_por_cuenta": rc,
                "Interfaces_usadas": st.session_state.get("interfaces", pd.DataFrame()),
                "MD_actual_usado": st.session_state.get("md_actual", pd.DataFrame()),
                "Factores": factores_edit,
                "Cuentas": cuentas_edit,
                "Alertas": ap,
            }
            xbytes = to_excel_bytes(sheets, include_chart=False)
            st.download_button("⬇️ Descargar Excel predicción", data=xbytes, file_name=f"prediccion_horas_{mes_pred.replace('.', '')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif menu == "4. Instructivo":
    st.subheader("4. Instructivo de uso y lectura")
    st.markdown("""
    ### Objetivo
    Esta herramienta permite comparar históricamente lo pagado, provisionado y proyectado por mes de novedad, concepto, tipo de hora, cargo homologado, CECO y área; además genera una predicción del mes en curso.

    ### Regla de mes vencido
    Las horas se pagan mes vencido: pago 05.2026 corresponde a novedad 04.2026. Por eso el comparativo se realiza por **mes de novedad**.

    ### Homologación de cargos
    La app intenta homologar por SAP + periodo usando el Headcount/MD del periodo. Si el SAP viene como Error, vacío o no cruza, usa la columna Función/Cargo del archivo y aplica la homologación. Si no encuentra equivalencia, lo reporta en alertas.

    ### Predicción
    Para costear la predicción, el MD calcula salario total vigente, lo divide por la jornada vigente y multiplica por el factor del concepto.

    ### Lectura de valores
    Los valores monetarios se muestran como pesos sin decimales y las cantidades con 2 decimales.
    """)

elif menu == "5. Plantillas":
    st.subheader("4. Plantillas de apoyo")
    st.write("Descarga estas plantillas si necesitas parametrizar cuentas o revisar factores.")
    st.download_button(
        "⬇️ Descargar plantilla de cuentas",
        data=to_excel_bytes({"Cuentas": CUENTAS_TEMPLATE}, include_chart=False),
        file_name="plantilla_cuentas_horas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "⬇️ Descargar plantilla de factores",
        data=to_excel_bytes({"Factores": FACTORES_DEFAULT}, include_chart=False),
        file_name="plantilla_factores_horas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.markdown("""
    **Estructura mínima recomendada:**

    - CC-nómina: `Per.para`, `CC-n.`, `Cantidad`, `Importe`, `Ce.coste`, `Función`, `Denominación función`.
    - Provisión: `Source.Name/MES`, `CECO`, `CARGO`, `Valores`, `Total`, `PROVISIÓN`.
    - Proyección: `MES`, `Funcion`, `Ce.coste`, `Y220_Q`, `Y220_$`, etc.
    - Interface: 4 columnas sin encabezado: `SAP`, `Fecha pago`, `Concepto Y540-Y547`, `Cantidad`.
    - MD actual: puede ser el TXT SAP completo o el Excel consolidado. Si viene TXT, aplica la lógica de salario vigente: vigencia por persona, último registro por SAP + concepto según `Modif.el` desc e `Importe` desc, suma salario base + bonos, calcula `Salario total`, jornada vigente y valor hora. Si ya viene `Salario total`, lo usa directo.
    """)

st.caption("Creado por Andrés Huérfano Dávila - Nómina JMC")
