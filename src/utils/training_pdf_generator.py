"""
Módulo de Geração de Relatórios Técnicos em PDF para Treinamento de Kendo (SenpAI)
Gera relatórios detalhados, elegantes e embasados na literatura de Kendo (FIK/AJKF),
cobrindo os 3 Pilares Fundamentais (Movimentação, Precisão e Constância),
diagnóstico pedagógico da IA, pontos fortes, oportunidades de melhoria e drills prescritivos.
"""

import io
import re
import datetime
from typing import Dict, Any, List, Optional

from reportlab.lib.pagesizes import A4  # pyrefly: ignore[missing-source-for-stubs]
from reportlab.lib import colors  # pyrefly: ignore[missing-source-for-stubs]
from reportlab.pdfgen import canvas  # pyrefly: ignore[missing-source-for-stubs]
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak, HRFlowable  # pyrefly: ignore[missing-source-for-stubs]
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # pyrefly: ignore[missing-source-for-stubs]
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY  # pyrefly: ignore[missing-source-for-stubs]

from src.analytics.training_analyzer import TRAINING_MODALITIES_METADATA, get_kendoka_id_aliases


def _resolve_kendoka_pdf_name(kp: Dict[str, Any], saved_names: Dict[str, str], idx_fallback: int = 0) -> str:
    k_id = str(kp.get("kendoka_id") or f"KENSHI_{idx_fallback + 1:02d}").strip()
    k_def = str(kp.get("default_name") or f"Kendoca {idx_fallback + 1}").strip()
    for al in get_kendoka_id_aliases(k_id, fallback_index=idx_fallback):
        val = saved_names.get(al)
        if val and val.strip():
            return val.strip()
    return str(kp.get("custom_name") or k_def).strip()


def _clean_text_for_pdf(text: Any) -> str:
    """
    Higieniza o texto para renderização segura em fontes padrão do ReportLab (Helvetica),
    convertendo emojis em identificadores legíveis e preservando a acentuação em português.
    """
    if text is None:
        return ""
    s = str(text)

    # Substituição de emojis por representações textuais elegantes
    emoji_map = {
        "🥋": "[Kendo] ",
        "⏱️": "[Tempo] ",
        "⏱": "[Tempo] ",
        "🎯": "[Alvo] ",
        "🏆": "[Excelente] ",
        "🥇": "[Muito Bom] ",
        "🥈": "[Satisfatório] ",
        "⚠️": "[Atenção] ",
        "⚠": "[Atenção] ",
        "✅": "[OK] ",
        "💡": "[Insight] ",
        "🏋️": "[Drill] ",
        "🏋": "[Drill] ",
        "🔍": "[Detecção] ",
        "⚙️": "[Manual] ",
        "⚙": "[Manual] ",
        "🔁": "[Reps] ",
        "📊": "[Métricas] ",
        "👁️": "[Presença] ",
        "👁": "[Presença] ",
        "🔄": "[Re-ID] ",
        "📖": "[Princípio] ",
        "🌐": "[Referência] ",
        "📦": "[Pacote] ",
        "📥": "[Download] ",
        "📄": "[PDF] ",
        "🔗": "[Link] ",
        "🏷️": "[Tag] ",
        "🏷": "[Tag] ",
        "❌": "[X] ",
        "•": "-",
    }
    for emoji_char, rep in emoji_map.items():
        s = s.replace(emoji_char, rep)

    # Remove outros caracteres fora da faixa Latin-1 / ASCII para prevenir fontes ausentes
    # Mantém caracteres padrão, acentuação (á, é, í, ó, ú, ã, õ, ç etc.) e pontuação comum
    cleaned_chars = []
    for ch in s:
        code = ord(ch)
        if code < 256:
            cleaned_chars.append(ch)
        elif code in (8211, 8212):  # en-dash, em-dash
            cleaned_chars.append("-")
        elif code in (8216, 8217):  # aspas simples curvas
            cleaned_chars.append("'")
        elif code in (8220, 8221):  # aspas duplas curvas
            cleaned_chars.append('"')
        elif code == 8230:  # ellipsis
            cleaned_chars.append("...")
        else:
            # Caracteres CJK/outros são ignorados ou substituídos
            pass

    res = "".join(cleaned_chars)
    # Protege caracteres especiais de XML/HTML para o Paragraph do ReportLab
    res = res.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return res.strip()


class NumberedCanvas(canvas.Canvas):
    """
    Canvas em dois passos para numeração dinâmica de páginas ('Página X de Y')
    e rodapé institucional padronizado do SenpAI.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        start_page_fn = getattr(self, "_startPage", None)
        if callable(start_page_fn):
            start_page_fn()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Linha fina superior do rodapé
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(36, 32, A4[0] - 36, 32)

        # Rodapé esquerdo e direito
        footer_left = "SenpAI - Avaliação de Treinamento e Aprendizado de Kendo (FIK/AJKF Standards)"
        footer_right = f"Página {self.getPageNumber()} de {page_count}"
        self.drawString(36, 20, footer_left)
        self.drawRightString(A4[0] - 36, 20, footer_right)

        self.restoreState()


def generate_training_pdf_report(
    analysis_result: Dict[str, Any],
    kendoka_names: Optional[Dict[str, str]] = None,
    modality_key: Optional[str] = None
) -> bytes:
    """
    Gera o relatório consolidado completo de análise técnica em PDF.
    
    Args:
        analysis_result: Dicionário com os resultados completos da análise do SenpAI.
        kendoka_names: Mapeamento customizado de nomes de kendocas ({id: nome}).
        modality_key: Identificador da modalidade (ex: 'kirikaeshi', 'kihon').

    Returns:
        bytes: Conteúdo binário do documento PDF pronto para download.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42
    )

    saved_names = kendoka_names or {}
    train_data = analysis_result.get("training_analysis", {})
    if not train_data:
        train_data = analysis_result

    mod_key = modality_key or train_data.get("modality", "kihon")
    meta = TRAINING_MODALITIES_METADATA.get(mod_key, TRAINING_MODALITIES_METADATA.get("kihon", {}))

    mod_display_name = meta.get("name", mod_key.replace("_", " ").title())
    mod_cat = meta.get("category", "Geral")
    mod_desc = meta.get("description", "")
    focus_areas = meta.get("focus_areas", [])
    det_method = train_data.get("detection_method", "AUTO_DETECTED")
    from src.analytics.training_analyzer import modality_confidence_label
    det_conf_text = modality_confidence_label(det_method, train_data.get("detection_confidence"))
    justification = train_data.get("justification", "")
    learned_principles = train_data.get("learned_principles", [])
    web_sources = train_data.get("web_sources", [])

    dur_sec = float(analysis_result.get("duration_seconds", train_data.get("duration_seconds", 0.0)))
    fps = float(analysis_result.get("processing_fps", analysis_result.get("fps", 30.0)))
    kendokas_raw = train_data.get("kendokas", [])
    sub_sessions_raw = train_data.get("sub_sessions", [])
    modality_blocks_raw = train_data.get("modality_blocks", [])
    now_str = datetime.datetime.now().strftime("%d/%m/%Y às %H:%M")

    # Estilos de Tipografia
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        alignment=TA_LEFT
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#475569"),
        alignment=TA_LEFT
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1E1B4B"),
        spaceBefore=12,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#334155"),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
        alignment=TA_LEFT
    )

    body_bold = ParagraphStyle(
        "Body_Bold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    callout_style = ParagraphStyle(
        "Callout_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
        alignment=TA_JUSTIFY
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0F172A")
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell_style,
        fontName="Helvetica-Bold"
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    story = []

    # ==========================================
    # 1. CABEÇALHO DO RELATÓRIO & IDENTIDADE SENPAI
    # ==========================================
    header_table_data = [
        [
            Paragraph("<b>SENPAI</b> • Inteligência Artificial para Kendo", ParagraphStyle("Brand", fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor("#4F46E5"))),
            Paragraph(f"Emissão: {now_str}", ParagraphStyle("EmitDate", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#64748B"), alignment=TA_RIGHT))
        ],
        [
            Paragraph("RELATÓRIO TÉCNICO DE AVALIAÇÃO DE TREINAMENTO", title_style),
            ""
        ],
        [
            Paragraph(f"Modalidade: <b>{_clean_text_for_pdf(mod_display_name)}</b> — <i>{_clean_text_for_pdf(mod_cat)}</i>", subtitle_style),
            ""
        ]
    ]

    t_header = Table(header_table_data, colWidths=[380, 143])
    t_header.setStyle(TableStyle([
        ("SPAN", (0, 1), (1, 1)),
        ("SPAN", (0, 2), (1, 2)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 8))

    # Tabela de Metadados da Sessão
    method_label = {"AUTO_DETECTED": "Regra fixa", "MODELO": "Classificador de modalidade"}.get(det_method, "Seleção Manual") + f" ({det_conf_text})"
    meta_table_data = [
        [
            Paragraph("<b>Duração da Sessão:</b>", body_style),
            Paragraph(f"{dur_sec:.1f} segundos", body_style),
            Paragraph("<b>Taxa de Processamento:</b>", body_style),
            Paragraph(f"{fps:.1f} FPS", body_style),
        ],
        [
            Paragraph("<b>Total de Praticantes:</b>", body_style),
            Paragraph(f"{len(kendokas_raw)} kendoca(s)", body_style),
            Paragraph("<b>Método de Detecção:</b>", body_style),
            Paragraph(method_label, body_style),
        ]
    ]
    t_meta = Table(meta_table_data, colWidths=[120, 140, 120, 143])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 12))

    # ==========================================
    # 1.B CRONOGRAMA DAS SESSÕES E AVALIAÇÃO CONCENTRADA POR BLOCO DE MODALIDADE
    # ==========================================
    if sub_sessions_raw:
        story.append(Paragraph("Cronograma das Sessões Identificadas na Sequência", h1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#4F46E5"), spaceBefore=2, spaceAfter=8))

        sched_headers = [
            Paragraph("Sessão", table_header_style),
            Paragraph("Intervalo (Início - Fim)", table_header_style),
            Paragraph("Duração", table_header_style),
            Paragraph("Modalidade Identificada", table_header_style),
            Paragraph("Categoria", table_header_style),
            Paragraph("Reps", table_header_style),
            Paragraph("Detecção", table_header_style),
        ]
        sched_rows = [sched_headers]
        for idx_ss, ss in enumerate(sub_sessions_raw):
            ss_idx = ss.get("segment_index", idx_ss + 1)
            s_ts = ss.get("start_timestamp", "00:00")
            e_ts = ss.get("end_timestamp", "00:00")
            s_dur = float(ss.get("duration_seconds", 0.0))
            s_mname = ss.get("modality_name", ss.get("modality_key", "Kihon"))
            s_mcat = ss.get("modality_category", "Geral")
            s_reps = int(ss.get("total_repetitions", 0))
            s_conf = round(float(ss.get("detection_confidence", 0.90)) * 100, 0)
            s_meth = "Manual" if "MANUAL" in str(ss.get("detection_method", "")).upper() else f"IA ({int(s_conf)}%)"
            sched_rows.append([
                Paragraph(f"<b>Sessão #{ss_idx}</b>", table_cell_bold),
                Paragraph(f"<b>{_clean_text_for_pdf(s_ts)} a {_clean_text_for_pdf(e_ts)}</b>", table_cell_style),
                Paragraph(f"{s_dur:.1f}s", table_cell_style),
                Paragraph(f"<b>{_clean_text_for_pdf(s_mname)}</b>", table_cell_style),
                Paragraph(_clean_text_for_pdf(s_mcat), table_cell_style),
                Paragraph(str(s_reps), table_cell_style),
                Paragraph(_clean_text_for_pdf(s_meth), table_cell_style),
            ])

        t_sched = Table(sched_rows, colWidths=[58, 92, 50, 138, 80, 40, 65])
        t_sched.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E1B4B")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_sched)
        story.append(Spacer(1, 12))

    if modality_blocks_raw:
        story.append(Paragraph("Avaliação Separada e Concentrada por Bloco de Modalidade", h1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#4F46E5"), spaceBefore=2, spaceAfter=8))

        for idx_mb, mb in enumerate(modality_blocks_raw):
            b_idx = mb.get("block_index", idx_mb + 1)
            b_mkey = mb.get("modality_key", "kihon")
            b_meta = TRAINING_MODALITIES_METADATA.get(b_mkey, {})
            b_mname = mb.get("modality_name") or b_meta.get("name", b_mkey)
            b_mcat = mb.get("modality_category") or b_meta.get("category", "Geral")
            b_desc = mb.get("description") or b_meta.get("description", "")
            b_focus = mb.get("focus_areas") or b_meta.get("focus_areas", [])
            b_intervals = mb.get("intervals_label", "Sessão Completa")
            b_dur = float(mb.get("total_duration_seconds", 0.0))
            b_reps = int(mb.get("total_repetitions", 0))
            b_scount = int(mb.get("session_count", max(1, len(mb.get("intervals", [])))))
            b_kendokas = mb.get("kendokas", [])

            block_banner = Table(
                [
                    [
                        Paragraph(
                            f"<font color='#FFFFFF'><b>BLOCO DE MODALIDADE #{b_idx}: {_clean_text_for_pdf(b_mname).upper()}</b></font><br/>"
                            f"<font color='#C7D2FE' size=8>Categoria: {_clean_text_for_pdf(b_mcat)} | Intervalos Concentrados: <b>{_clean_text_for_pdf(b_intervals)}</b></font>",
                            ParagraphStyle("BlockBannerLeft", parent=body_style, textColor=colors.white, leading=12)
                        ),
                        Paragraph(
                            f"<font color='#E0E7FF' size=8><b>Duração Acumulada:</b> {b_dur:.1f}s<br/>"
                            f"<b>Sessões Agrupadas:</b> {b_scount} | <b>Reps:</b> {b_reps}</font>",
                            ParagraphStyle("BlockBannerRight", parent=body_style, textColor=colors.white, alignment=TA_RIGHT, leading=11)
                        )
                    ]
                ],
                colWidths=[355, 168]
            )
            block_banner.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#312E81")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#4F46E5")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(block_banner)
            story.append(Spacer(1, 4))

            if b_desc or b_focus:
                b_desc_table = Table(
                    [[
                        Paragraph(
                            (f"<b>Fundamento da Modalidade:</b> {_clean_text_for_pdf(b_desc)}<br/>" if b_desc else "")
                            + (f"<b>Focos Técnicos:</b> {_clean_text_for_pdf(' • '.join(b_focus))}" if b_focus else ""),
                            callout_style
                        )
                    ]],
                    colWidths=[523]
                )
                b_desc_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF2FF")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D2FE")),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(b_desc_table)
                story.append(Spacer(1, 4))

            if b_kendokas:
                b_rows = [[
                    Paragraph("Praticante no Bloco", table_header_style),
                    Paragraph("Papel", table_header_style),
                    Paragraph("Geral", table_header_style),
                    Paragraph("Movimento", table_header_style),
                    Paragraph("Precisão", table_header_style),
                    Paragraph("Constância", table_header_style),
                    Paragraph("Reps", table_header_style),
                    Paragraph("Cadência", table_header_style),
                ]]
                for idx_bk, bkp in enumerate(b_kendokas):
                    bk_id = bkp.get("kendoka_id") or f"KENSHI_{idx_bk+1:02d}"
                    bk_name = _resolve_kendoka_pdf_name(bkp, saved_names, idx_bk)
                    bk_role = bkp.get("role", "Praticante")
                    bk_pil = bkp.get("pillars", {})
                    bk_ov = float(bk_pil.get("overall_score", 0.0))
                    bk_mov = float(bk_pil.get("movimentacao", bk_pil.get("forma", 0.0)))
                    bk_prec = float(bk_pil.get("precisao", 0.0))
                    bk_const = float(bk_pil.get("constancia", 0.0))
                    bk_reps = int(bk_pil.get("total_repetitions", 0))
                    bk_cad = float(bk_pil.get("cadence_cpm", 0.0))

                    b_rows.append([
                        Paragraph(f"<b>{_clean_text_for_pdf(bk_name)}</b>", table_cell_style),
                        Paragraph(_clean_text_for_pdf(bk_role), table_cell_style),
                        Paragraph(f"<b>{bk_ov:.1f}</b>", table_cell_bold),
                        Paragraph(f"{bk_mov:.1f}%", table_cell_style),
                        Paragraph(f"{bk_prec:.1f}%", table_cell_style),
                        Paragraph(f"{bk_const:.1f}%", table_cell_style),
                        Paragraph(str(bk_reps), table_cell_style),
                        Paragraph(f"{bk_cad:.1f} cpm", table_cell_style),
                    ])

                t_bk = Table(b_rows, colWidths=[130, 68, 50, 60, 60, 60, 40, 55])
                t_bk.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(t_bk)
                story.append(Spacer(1, 4))

                # Diagnóstico resumido de cada praticante dentro deste bloco de modalidade
                for idx_bk, bkp in enumerate(b_kendokas):
                    bk_id = bkp.get("kendoka_id") or f"KENSHI_{idx_bk+1:02d}"
                    bk_name = _resolve_kendoka_pdf_name(bkp, saved_names, idx_bk)
                    bk_str = bkp.get("strengths", [])
                    bk_imp = bkp.get("improvements", [])
                    bk_ex = bkp.get("recommended_exercises", [])
                    if bk_str or bk_imp or bk_ex:
                        lines_bk = [f"<b>Diagnóstico de {_clean_text_for_pdf(bk_name)} em {_clean_text_for_pdf(b_mname)}:</b>"]
                        if bk_str:
                            lines_bk.append(f"• [OK] <b>Pontos Fortes:</b> {_clean_text_for_pdf(' | '.join(bk_str[:2]))}")
                        if bk_imp:
                            lines_bk.append(f"• [Atenção] <b>Ajustes:</b> {_clean_text_for_pdf(' | '.join(bk_imp[:2]))}")
                        if bk_ex:
                            ex0 = bk_ex[0]
                            lines_bk.append(
                                f"• [Drill] <b>Prescrição ({_clean_text_for_pdf(ex0.get('name', 'Drill'))}):</b> "
                                f"{_clean_text_for_pdf(ex0.get('prescription', ''))}"
                            )
                        t_bk_diag = Table([[Paragraph("<br/>".join(lines_bk), body_style)]], colWidths=[523])
                        t_bk_diag.setStyle(TableStyle([
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ]))
                        story.append(t_bk_diag)
                        story.append(Spacer(1, 3))

            story.append(Spacer(1, 8))

    # ==========================================
    # 2. FUNDAMENTAÇÃO TÉCNICA & DIRETRIZES
    # ==========================================
    story.append(Paragraph("1. Fundamentação Técnica e Metodológica da Modalidade", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#6366F1"), spaceBefore=2, spaceAfter=8))

    desc_box_data = [
        [
            Paragraph(
                f"<b>Descrição Oficial:</b> {_clean_text_for_pdf(mod_desc)}<br/><br/>"
                f"<b>Focos Principais de Avaliação:</b> {_clean_text_for_pdf(' • '.join(focus_areas))}"
                + (f"<br/><br/><b>Diagnóstico da IA:</b> {_clean_text_for_pdf(justification)}" if justification else ""),
                callout_style
            )
        ]
    ]
    t_desc = Table(desc_box_data, colWidths=[523])
    t_desc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF2FF")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#C7D2FE")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(t_desc)
    story.append(Spacer(1, 8))

    if learned_principles or web_sources:
        princ_items = []
        if learned_principles:
            for p_item in learned_principles:
                princ_items.append(Paragraph(f"• <b>{_clean_text_for_pdf(p_item)}</b>", body_style))
        if web_sources:
            for w_item in web_sources:
                princ_items.append(Paragraph(f"• <i>{_clean_text_for_pdf(w_item.get('title', 'Referência'))}</i> ({_clean_text_for_pdf(w_item.get('type', 'Web'))})", body_style))

        t_princ = Table([[p] for p in princ_items], colWidths=[523])
        t_princ.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t_princ)
        story.append(Spacer(1, 10))

    # ==========================================
    # 3. QUADRO GERAL COMPARATIVO DOS KENDOCAS
    # ==========================================
    story.append(Paragraph("2. Quadro Comparativo de Praticantes (Os 3 Pilares)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#6366F1"), spaceBefore=2, spaceAfter=8))

    summary_headers = [
        Paragraph("Praticante", table_header_style),
        Paragraph("Papel", table_header_style),
        Paragraph("Geral", table_header_style),
        Paragraph("Movimento", table_header_style),
        Paragraph("Precisão", table_header_style),
        Paragraph("Constância", table_header_style),
        Paragraph("Reps", table_header_style),
        Paragraph("Cadência", table_header_style),
        Paragraph("Presença", table_header_style),
    ]
    summary_rows = [summary_headers]

    for idx_k, kp in enumerate(kendokas_raw):
        k_id = kp.get("kendoka_id") or f"KENSHI_{idx_k+1:02d}"
        k_def = kp.get("default_name", f"Kendoca {idx_k+1}")
        k_name = _resolve_kendoka_pdf_name(kp, saved_names, idx_k)
        role = kp.get("role", "Praticante")
        pil = kp.get("pillars", {})
        ov_sc = float(pil.get("overall_score", 0.0))
        mov_sc = float(pil.get("movimentacao", pil.get("forma", 0.0)))
        prec_sc = float(pil.get("precisao", 0.0))
        const_sc = float(pil.get("constancia", 0.0))
        reps = int(pil.get("total_repetitions", 0))
        cad = float(pil.get("cadence_cpm", 0.0))
        pres = float(kp.get("presence_seconds", 0.0))

        summary_rows.append([
            Paragraph(f"<b>{_clean_text_for_pdf(k_name)}</b><br/><font color='#64748B' size=6>{k_id}</font>", table_cell_style),
            Paragraph(_clean_text_for_pdf(role), table_cell_style),
            Paragraph(f"<b>{ov_sc:.1f}</b>", table_cell_bold),
            Paragraph(f"{mov_sc:.1f}%", table_cell_style),
            Paragraph(f"{prec_sc:.1f}%", table_cell_style),
            Paragraph(f"{const_sc:.1f}%", table_cell_style),
            Paragraph(str(reps), table_cell_style),
            Paragraph(f"{cad:.1f} cpm", table_cell_style),
            Paragraph(f"{pres:.1f}s", table_cell_style),
        ])

    t_summary = Table(
        summary_rows,
        colWidths=[110, 60, 45, 52, 52, 54, 40, 55, 55]
    )
    t_summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E1B4B")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 14))

    # ==========================================
    # 4. DIAGNÓSTICO INDIVIDUAL MINUCIOSO
    # ==========================================
    story.append(Paragraph("3. Diagnóstico Biomecânico e Técnico por Praticante", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#6366F1"), spaceBefore=2, spaceAfter=8))

    for idx_k, kp in enumerate(kendokas_raw):
        k_id = kp.get("kendoka_id") or f"KENSHI_{idx_k+1:02d}"
        k_def = kp.get("default_name", f"Kendoca {idx_k+1}")
        k_name = _resolve_kendoka_pdf_name(kp, saved_names, idx_k)
        role = kp.get("role", "Praticante")
        pil = kp.get("pillars", {})
        ov_sc = float(pil.get("overall_score", 0.0))
        mov_sc = float(pil.get("movimentacao", pil.get("forma", 0.0)))
        prec_sc = float(pil.get("precisao", 0.0))
        const_sc = float(pil.get("constancia", 0.0))
        reps = int(pil.get("total_repetitions", 0))
        cad = float(pil.get("cadence_cpm", 0.0))
        cad_std = float(pil.get("cadence_std_dev_seconds", 0.0))
        pres = float(kp.get("presence_seconds", 0.0))
        reentries = int(kp.get("reentry_count", 0))

        strengths = kp.get("strengths", [])
        improvements = kp.get("improvements", [])
        exercises = kp.get("recommended_exercises", [])
        deviations = kp.get("kinesthetic_deviations", {})

        mov_subs = pil.get("movimentacao_submetrics", pil.get("forma_submetrics", {}))
        prec_subs = pil.get("precisao_submetrics", {})
        const_subs = pil.get("constancia_submetrics", {})

        if ov_sc >= 85:
            perf_tag = "EXCELENTE (NÍVEL AVANÇADO)"
            tag_color = colors.HexColor("#15803D")
            tag_bg = colors.HexColor("#DCFCE7")
        elif ov_sc >= 70:
            perf_tag = "MUITO BOM (NÍVEL INTERMEDIÁRIO)"
            tag_color = colors.HexColor("#0369A1")
            tag_bg = colors.HexColor("#E0F2FE")
        elif ov_sc >= 55:
            perf_tag = "SATISFATÓRIO (EM DESENVOLVIMENTO)"
            tag_color = colors.HexColor("#A16207")
            tag_bg = colors.HexColor("#FEF9C3")
        else:
            perf_tag = "NECESSITA AJUSTES DE FUNDAMENTO"
            tag_color = colors.HexColor("#B91C1C")
            tag_bg = colors.HexColor("#FEE2E2")

        k_card_elements = []

        # Card Header com Nome e Nota
        k_header_table = Table(
            [
                [
                    Paragraph(f"<b>{_clean_text_for_pdf(k_name)}</b> <font color='#64748B'>({k_def} • {_clean_text_for_pdf(role)})</font>", h2_style),
                    Paragraph(f"<font color='{tag_color.hexval()}'><b>{perf_tag} — {ov_sc:.1f}/100</b></font>", ParagraphStyle("Badge", parent=table_cell_bold, alignment=TA_RIGHT))
                ]
            ],
            colWidths=[330, 193]
        )
        k_header_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), tag_bg),
            ("BOX", (0, 0), (-1, -1), 1, tag_color),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        k_card_elements.append(k_header_table)
        k_card_elements.append(Spacer(1, 4))

        # Tabela dos 3 Pilares e Submétricas
        pillars_detail_data = [
            [
                Paragraph("<b>PILAR 1: MOVIMENTAÇÃO</b>", table_header_style),
                Paragraph("<b>PILAR 2: PRECISÃO</b>", table_header_style),
                Paragraph("<b>PILAR 3: CONSTÂNCIA</b>", table_header_style),
            ],
            [
                Paragraph(
                    f"<b>Score Geral: {mov_sc:.1f}%</b><br/>"
                    f"• Verticalidade (Shisei): {mov_subs.get('verticalidade_coluna', 0):.1f}%<br/>"
                    f"• Nivelamento de Ombros: {mov_subs.get('nivelamento_ombros', 0):.1f}%<br/>"
                    f"• Base e Pés (Ashi-gamae): {mov_subs.get('alinhamento_base_pes', 0):.1f}%<br/>"
                    f"• Amplitude Furikaburi: {mov_subs.get('amplitude_furikaburi', 0):.1f}%",
                    table_cell_style
                ),
                Paragraph(
                    f"<b>Score Geral: {prec_sc:.1f}%</b><br/>"
                    f"• Alvo (Datotsu-bui): {prec_subs.get('trajetoria_alvo', 0):.1f}%<br/>"
                    f"• Ki-Ken-Tai-Ichi: {prec_subs.get('kikentai_sincronismo', 0):.1f}%<br/>"
                    f"• Linha Central (Chushin): {prec_subs.get('controle_linha_centro', 0):.1f}%<br/>"
                    f"• Hasuji (Alinhamento Lâmina): {prec_subs.get('hasuji_angle_consistency', 85.0):.1f}%",
                    table_cell_style
                ),
                Paragraph(
                    f"<b>Score Geral: {const_sc:.1f}%</b><br/>"
                    f"• Regularidade de Ritmo: {const_subs.get('regularidade_ritmo', 0):.1f}%<br/>"
                    f"• Resistência (Stamina): {const_subs.get('resistencia_fadiga', 0):.1f}%<br/>"
                    f"• Cadência: {cad:.1f} rep/min<br/>"
                    f"• Desvio Padrão: {cad_std:.2f}s",
                    table_cell_style
                )
            ]
        ]
        t_pil = Table(pillars_detail_data, colWidths=[174, 174, 175])
        t_pil.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        k_card_elements.append(t_pil)
        k_card_elements.append(Spacer(1, 5))

        # Diagnóstico Pedagógico: Pontos Fortes e Oportunidades de Melhoria
        diag_rows = []
        if strengths:
            diag_rows.append([
                Paragraph("<b>Pontos Fortes Biomecânicos Observados:</b>", body_bold),
            ])
            for st_item in strengths:
                diag_rows.append([Paragraph(f"• [OK] {_clean_text_for_pdf(st_item)}", body_style)])

        if improvements:
            diag_rows.append([
                Paragraph("<b>Oportunidades de Correção Técnica:</b>", body_bold),
            ])
            for imp_item in improvements:
                diag_rows.append([Paragraph(f"• [Atenção] {_clean_text_for_pdf(imp_item)}", body_style)])

        if exercises:
            diag_rows.append([
                Paragraph("<b>Prescrições e Drills Recomendados para o Dojo:</b>", body_bold),
            ])
            for ex in exercises:
                ex_name = ex.get("name", "Drill de Kendo")
                ex_target = ex.get("target", "Fundamentos")
                ex_presc = ex.get("prescription", "")
                diag_rows.append([
                    Paragraph(f"• <b>{_clean_text_for_pdf(ex_name)}</b> <i>(Foco: {_clean_text_for_pdf(ex_target)})</i>: {_clean_text_for_pdf(ex_presc)}", body_style)
                ])

        if diag_rows:
            t_diag = Table(diag_rows, colWidths=[523])
            t_diag.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]))
            k_card_elements.append(t_diag)

        k_card_elements.append(Spacer(1, 10))
        story.append(KeepTogether(k_card_elements))

    # ==========================================
    # 5. METODOLOGIA E CERTIFICAÇÃO SENPAI
    # ==========================================
    story.append(Spacer(1, 6))
    method_box = [
        [
            Paragraph(
                "<b>Nota Metodológica e Validação de IA:</b><br/>"
                "A avaliação computacional do SenpAI utiliza rastreamento cinemático 3D de esqueleto articular "
                "(MediaPipe Pose + YOLOv8) com taxa de amostragem em tempo real, integrando parâmetros regulamentares "
                "da Federação Internacional de Kendo (FIK) e da All Japan Kendo Federation (AJKF). As métricas dos "
                "3 Pilares avaliam a biomecânica (postura, ângulo de Hasuji, sincronismo Ki-Ken-Tai-Ichi e cadência). "
                "Recomenda-se a utilização deste laudo em conjunto com o acompanhamento presencial do Sensei responsável.",
                callout_style
            )
        ]
    ]
    t_method = Table(method_box, colWidths=[523])
    t_method.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_method)

    # Constrói o PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()


def generate_individual_kendoka_pdf_report(
    kendoka_profile: Dict[str, Any],
    modality_meta: Optional[Dict[str, Any]] = None,
    custom_name: Optional[str] = None,
    modality_blocks: Optional[List[Dict[str, Any]]] = None,
    sub_sessions: Optional[List[Dict[str, Any]]] = None
) -> bytes:
    """
    Gera um relatório individual especializado em PDF para um único kendoca,
    incluindo a separação concentrada por blocos de modalidade de treinamento.
    """
    kp = kendoka_profile
    meta = modality_meta or TRAINING_MODALITIES_METADATA.get("kihon", {})
    k_id = kp.get("kendoka_id", "KENSHI_01")
    c_name = custom_name or kp.get("custom_name") or kp.get("default_name", "Kendoca")

    # Filtrar modality_blocks para conter apenas o kendoka atual (ou construir a partir de modality_breakdown)
    filtered_blocks: List[Dict[str, Any]] = []
    if modality_blocks:
        for mb in modality_blocks:
            mb_copy = dict(mb)
            k_in_mb = [
                k for k in mb.get("kendokas", [])
                if k.get("kendoka_id") == k_id or k.get("default_name") == kp.get("default_name")
            ]
            if not k_in_mb and mb.get("kendokas"):
                k_in_mb = [mb["kendokas"][0]]
            mb_copy["kendokas"] = k_in_mb
            filtered_blocks.append(mb_copy)
    elif kp.get("modality_breakdown"):
        for idx_mb, bd in enumerate(kp.get("modality_breakdown", [])):
            m_key_bd = bd.get("modality_key", meta.get("key", "kihon"))
            m_meta_bd = TRAINING_MODALITIES_METADATA.get(m_key_bd, {})
            filtered_blocks.append({
                "block_index": idx_mb + 1,
                "modality_key": m_key_bd,
                "modality_name": bd.get("modality_name") or m_meta_bd.get("name", m_key_bd),
                "modality_category": bd.get("modality_category") or m_meta_bd.get("category", "Geral"),
                "description": m_meta_bd.get("description", ""),
                "focus_areas": m_meta_bd.get("focus_areas", []),
                "intervals": bd.get("intervals", []),
                "intervals_label": bd.get("intervals_label", "Sessão"),
                "session_count": bd.get("session_count", 1),
                "total_duration_seconds": bd.get("total_duration_seconds", kp.get("presence_seconds", 60.0)),
                "total_repetitions": bd.get("pillars", {}).get("total_repetitions", 0),
                "kendokas": [{
                    "kendoka_id": k_id,
                    "default_name": kp.get("default_name", "Kendoca 1"),
                    "custom_name": c_name,
                    "role": kp.get("role", "Praticante"),
                    "presence_seconds": bd.get("presence_seconds", kp.get("presence_seconds", 60.0)),
                    "pillars": bd.get("pillars", kp.get("pillars", {})),
                    "strengths": bd.get("strengths", kp.get("strengths", [])),
                    "improvements": bd.get("improvements", kp.get("improvements", [])),
                    "recommended_exercises": bd.get("recommended_exercises", kp.get("recommended_exercises", []))
                }]
            })

    synth_res = {
        "duration_seconds": kp.get("presence_seconds", 60.0),
        "processing_fps": 30.0,
        "training_analysis": {
            "modality": meta.get("key", "kihon"),
            "detection_method": "AUTO_DETECTED",
            "detection_confidence": 0.95,
            "description": meta.get("description", ""),
            "justification": f"Relatório individual especializado para o praticante {c_name}.",
            "learned_principles": meta.get("focus_areas", []),
            "web_sources": [],
            "kendokas": [kp],
            "sub_sessions": sub_sessions or [],
            "modality_blocks": filtered_blocks
        }
    }
    return generate_training_pdf_report(synth_res, kendoka_names={k_id: c_name}, modality_key=meta.get("key", "kihon"))
