import os
import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pytesseract

# (opcional no Windows local) aponte o executável:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="OCR de Tabelas (Tesseract + OpenCV)", page_icon="🧾", layout="wide")
st.title("🧾 OCR de Tabelas — Tesseract + OpenCV")

with st.sidebar:
    st.header("Configurações")
    langs = st.multiselect("Idiomas do OCR", options=["por", "eng"], default=["por", "eng"])
    psm = st.selectbox("PSM (Page Segmentation Mode)", [3,4,6,7,11,13], index=2,
                       help="6=bloco de texto; 7=linha única; 11=texto esparso; 13=texto esparso em linha")
    min_line_len = st.slider("Tamanho mín. de linhas (px)", 20, 400, 120, step=10,
                             help="Aumente se as linhas da tabela estiverem grossas/longas")
    merge_tol = st.slider("Tolerância p/ unir linhas (px)", 2, 25, 10, step=1,
                          help="Agrupa linhas/colunas próximas como uma só")
    cell_pad = st.slider("Padding interno da célula (px)", 0, 8, 2, step=1,
                         help="Evita recortar bordas na hora do OCR")
    conf_min = st.slider("Confiança mínima (visual)", 0, 100, 60, step=1)

    aplicar_deskew = st.checkbox("Desinclinar (deskew) automático", value=True)
    bin_thresh = st.selectbox("Binarização", ["OTSU", "Adaptativa"], index=0)

upload = st.file_uploader("📎 Envie uma imagem (PNG/JPG/JPEG) com TABELA — linhas visíveis funcionam melhor", type=["png","jpg","jpeg"])

def to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def deskew(img_gray: np.ndarray) -> np.ndarray:
    # estimativa simples de ângulo via momentos; funciona bem para tabelas pouco inclinadas
    coords = np.column_stack(np.where(img_gray < 255))
    if coords.size == 0:
        return img_gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def binarize(gray: np.ndarray, mode="OTSU") -> np.ndarray:
    if mode == "OTSU":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    else:
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 31, 10)

def detect_grid(bin_img: np.ndarray, min_len: int, merge_tol: int):
    # invertido para linhas serem "brancas" em fundo preto na morfologia
    inv = 255 - bin_img

    # horizontais
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    horiz = cv2.erode(inv, hor_kernel, iterations=1)
    horiz = cv2.dilate(horiz, hor_kernel, iterations=1)

    # verticais
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    vert = cv2.erode(inv, ver_kernel, iterations=1)
    vert = cv2.dilate(vert, ver_kernel, iterations=1)

    # projeções para obter posições (picos)
    y_sum = np.sum(horiz > 0, axis=1)  # horizontais → soma por linha
    x_sum = np.sum(vert > 0, axis=0)   # verticais → soma por coluna

    def peak_positions(arr, tol):
        idxs = np.where(arr > 0)[0]
        if len(idxs) == 0:
            return []
        groups = []
        current = [idxs[0]]
        for v in idxs[1:]:
            if v - current[-1] <= tol:
                current.append(v)
            else:
                groups.append(current)
                current = [v]
        groups.append(current)
        # usa a média de cada grupo como posição de linha/coluna
        return [int(np.mean(g)) for g in groups]

    ys = peak_positions(y_sum, merge_tol)
    xs = peak_positions(x_sum, merge_tol)

    # remove bordas redundantes (às vezes pega borda externa duas vezes)
    ys = sorted(list(set(ys)))
    xs = sorted(list(set(xs)))

    return xs, ys, horiz, vert

def ocr_cell(img, x1, y1, x2, y2, pad=2, lang="eng", psm=6):
    h, w = img.shape[:2]
    x1p = max(0, x1+pad); y1p = max(0, y1+pad)
    x2p = min(w, x2-pad); y2p = min(h, y2-pad)
    roi = img[y1p:y2p, x1p:x2p]
    if roi.size == 0:
        return "", 0
    cfg = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(roi, lang=lang, config=cfg, output_type=pytesseract.Output.DATAFRAME)
    data = data[data.conf != -1]
    if data.empty:
        return "", 0
    text = " ".join([t for t in data.text.fillna("") if t.strip() != ""]).strip()
    conf = float(data.conf.replace(-1, np.nan).dropna().mean()) if hasattr(data.conf, "mean") else 0
    return text, conf

def build_table(image_bgr, xs, ys, pad, lang, psm):
    # cria células entre linhas adjacentes
    # Nota: xs/ys são trilhas (linhas de grade). Precisamos de intervalos [xs[i], xs[i+1]) e idem para ys.
    if len(xs) < 2 or len(ys) < 2:
        return None, None, []

    # garante monotônico
    xs = sorted(xs); ys = sorted(ys)
    # OCR por célula e matriz de resultados
    n_cols = len(xs)-1
    n_rows = len(ys)-1
    data = []
    conf_mat = []
    boxes = []
    # usa RGB para OCR (Tesseract trabalha melhor que BGR puro)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    lang_str = "+".join(lang) if isinstance(lang, (list, tuple)) else lang

    for i in range(n_rows):
        row_texts = []
        row_conf = []
        for j in range(n_cols):
            x1, x2 = xs[j], xs[j+1]
            y1, y2 = ys[i], ys[i+1]
            txt, conf = ocr_cell(image_rgb, x1, y1, x2, y2, pad=pad, lang=lang_str, psm=psm)
            row_texts.append(txt)
            row_conf.append(conf)
            boxes.append((x1, y1, x2, y2, conf))
        data.append(row_texts)
        conf_mat.append(row_conf)

    df = pd.DataFrame(data)
    conf_df = pd.DataFrame(conf_mat)
    return df, conf_df, boxes

def draw_boxes(image_bgr, boxes, conf_min=0):
    vis = image_bgr.copy()
    for (x1, y1, x2, y2, conf) in boxes:
        color = (0, 255, 0) if conf >= conf_min else (0, 165, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
    return vis

if upload is None:
    st.info("Envie uma imagem contendo **tabela com linhas** para extrair como DataFrame.")
    with st.expander("Diagnóstico do Tesseract"):
        try:
            st.write("Versão:", pytesseract.get_tesseract_version())
            st.write("Executável:", getattr(pytesseract.pytesseract, "tesseract_cmd", "auto (PATH)"))
        except Exception as e:
            st.warning(f"Não foi possível detectar a versão. Detalhe: {e}")
else:
    pil = Image.open(upload).convert("RGB")
    img_bgr = to_cv2(pil)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if aplicar_deskew:
        gray = deskew(gray)
        img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    bin_img = binarize(gray, bin_thresh)

    xs, ys, horiz, vert = detect_grid(bin_img, min_line_len, merge_tol)

    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Pré-visualização")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"Linhas detectadas: {len(xs)} colunas-grade; {len(ys)} linhas-grade")

    with col2:
        st.subheader("Mapa de Linhas (diagnóstico)")
        diag = cv2.merge([
            (255 - vert),    # B
            np.zeros_like(vert),  # G
            (255 - horiz)    # R
        ])
        st.image(diag, caption="Azul=verticais, Vermelho=hots", use_container_width=True)

    if len(xs) < 2 or len(ys) < 2:
        st.error("Não consegui detectar uma grade de tabela suficiente. Tente aumentar ‘Tamanho mín. de linhas’ ou reduzir ‘Tolerância’.")
        st.stop()

    df, conf_df, boxes = build_table(img_bgr, xs, ys, cell_pad, langs, psm)
    if df is None:
        st.error("Falha ao reconstruir a grade da tabela.")
        st.stop()

    st.subheader("Tabela (OCR)")
    # tenta usar a primeira linha como header se fizer sentido (mais texto/menos números)
    use_header = st.checkbox("Usar primeira linha como cabeçalho", value=True)
    out_df = df.copy()
    if use_header and len(out_df) >= 2:
        out_df.columns = [c if c != "" else f"col_{i+1}" for i, c in enumerate(out_df.iloc[0].tolist())]
        out_df = out_df.iloc[1:].reset_index(drop=True)

    st.dataframe(out_df, use_container_width=True)

    # downloads
    csv = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV", data=csv, file_name="tabela_ocr.csv", mime="text/csv")

    # overlay
    st.subheader("Overlay das células")
    overlay = draw_boxes(img_bgr, boxes, conf_min=conf_min)
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

    with st.expander("Confiabilidade por célula (média dos tokens)"):
        st.dataframe(conf_df, use_container_width=True)

    with st.expander("Diagnóstico do Tesseract"):
        try:
            st.write("Versão:", pytesseract.get_tesseract_version())
            st.write("Executável:", getattr(pytesseract.pytesseract, "tesseract_cmd", "auto (PATH)"))
        except Exception as e:
            st.warning(f"Não foi possível detectar a versão. Detalhe: {e}")
