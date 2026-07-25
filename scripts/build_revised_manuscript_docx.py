"""Build a formatted DOCX from the revised manuscript Markdown.

This small dependency-free writer is used because the environment does not
contain python-docx or a document conversion utility. It intentionally keeps
the manuscript text as the source of truth and renders equations as readable
monospace linear notation.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML_NS = "http://www.w3.org/XML/1998/namespace"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def q(name: str) -> str:
    prefix, local = name.split(":", 1)
    namespace = {
        "w": W_NS,
        "r": R_NS,
        "xml": XML_NS,
        "wp": WP_NS,
        "a": A_NS,
        "pic": PIC_NS,
        "m": M_NS,
    }[prefix]
    return f"{{{namespace}}}{local}"


def rel_q(name: str) -> str:
    return f"{{{PKG_REL_NS}}}{name}"


ET.register_namespace("w", W_NS)
ET.register_namespace("r", R_NS)
ET.register_namespace("xml", XML_NS)
ET.register_namespace("wp", WP_NS)
ET.register_namespace("a", A_NS)
ET.register_namespace("pic", PIC_NS)
ET.register_namespace("m", M_NS)
ET.register_namespace("cp", "http://schemas.openxmlformats.org/package/2006/metadata/core-properties")
ET.register_namespace("dc", "http://purl.org/dc/elements/1.1/")
ET.register_namespace("dcterms", "http://purl.org/dc/terms/")
ET.register_namespace("ep", "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties")
ET.register_namespace("vt", "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes")


def add_text_run(parent: ET.Element, text: str, *, bold: bool = False, code: bool = False) -> None:
    if not text:
        return
    run = ET.SubElement(parent, q("w:r"))
    if bold or code:
        props = ET.SubElement(run, q("w:rPr"))
        if bold:
            ET.SubElement(props, q("w:b"))
        if code:
            fonts = ET.SubElement(props, q("w:rFonts"))
            fonts.set(q("w:ascii"), "Courier New")
            fonts.set(q("w:hAnsi"), "Courier New")
            size = ET.SubElement(props, q("w:sz"))
            size.set(q("w:val"), "18")
            color = ET.SubElement(props, q("w:color"))
            color.set(q("w:val"), "404040")
    node = ET.SubElement(run, q("w:t"))
    if text[:1].isspace() or text[-1:].isspace():
        node.set(f"{{{XML_NS}}}space", "preserve")
    node.text = text


def add_inline_runs(parent: ET.Element, text: str) -> None:
    pattern = re.compile(r"(\*\*.*?\*\*|`.*?`)")
    position = 0
    for match in pattern.finditer(text):
        add_text_run(parent, text[position : match.start()])
        token = match.group(0)
        if token.startswith("**"):
            add_text_run(parent, token[2:-2], bold=True)
        else:
            add_text_run(parent, token[1:-1], code=True)
        position = match.end()
    add_text_run(parent, text[position:])


def set_paragraph_style(paragraph: ET.Element, style_id: str) -> None:
    props = paragraph.find(q("w:pPr"))
    if props is None:
        props = ET.Element(q("w:pPr"))
        paragraph.insert(0, props)
    style = ET.SubElement(props, q("w:pStyle"))
    style.set(q("w:val"), style_id)


def add_paragraph(
    body: ET.Element,
    text: str = "",
    *,
    style: str = "Normal",
    code: bool = False,
    placeholder: bool = False,
    blockquote: bool = False,
) -> ET.Element:
    paragraph = ET.SubElement(body, q("w:p"))
    set_paragraph_style(paragraph, style)
    props = paragraph.find(q("w:pPr"))
    if props is not None and blockquote:
        indent = ET.SubElement(props, q("w:ind"))
        indent.set(q("w:left"), "420")
        shading = ET.SubElement(props, q("w:shd"))
        shading.set(q("w:fill"), "F2F2F2")
        border = ET.SubElement(props, q("w:pBdr"))
        left = ET.SubElement(border, q("w:left"))
        left.set(q("w:val"), "single")
        left.set(q("w:sz"), "16")
        left.set(q("w:space"), "8")
        left.set(q("w:color"), "808080")
    if code:
        add_text_run(paragraph, text, code=True)
    elif placeholder:
        add_text_run(paragraph, text, bold=True)
    else:
        add_inline_runs(paragraph, text)
    return paragraph


def add_image_paragraph(body: ET.Element, relationship_id: str, image_path: Path, image_number: int) -> None:
    paragraph = ET.SubElement(body, q("w:p"))
    set_paragraph_style(paragraph, "Normal")
    props = paragraph.find(q("w:pPr"))
    if props is None:
        props = ET.SubElement(paragraph, q("w:pPr"))
    alignment = ET.SubElement(props, q("w:jc"))
    alignment.set(q("w:val"), "center")
    run = ET.SubElement(paragraph, q("w:r"))
    drawing = ET.SubElement(run, q("w:drawing"))
    inline = ET.SubElement(drawing, q("wp:inline"))
    sizes = {
        "proposed_architecture.svg": (5943600, 3396343),
        "proposed_architecture.png": (5943600, 3343275),
        "msc_module.svg": (5943600, 2227500),
        "msc_module.png": (5943600, 2227500),
        "lrse_gdf_modules.svg": (5943600, 3058285),
        "lrse_gdf_modules.png": (5943600, 3058285),
        "gdf_detail_branch.svg": (5943600, 2377440),
        "gdf_detail_branch.png": (5943600, 2377440),
        "evaluation_workflow.svg": (5943600, 2758029),
        "evaluation_workflow.png": (5943600, 2758029),
    }
    cx, cy = sizes.get(image_path.name, (5943600, 3000000))
    extent = ET.SubElement(inline, q("wp:extent"))
    extent.set("cx", str(cx))
    extent.set("cy", str(cy))
    effect = ET.SubElement(inline, q("wp:effectExtent"))
    for side in ("l", "t", "r", "b"):
        effect.set(side, "0")
    doc_pr = ET.SubElement(inline, q("wp:docPr"))
    doc_pr.set("id", str(image_number))
    doc_pr.set("name", image_path.name)
    graphic_frame = ET.SubElement(inline, q("wp:cNvGraphicFramePr"))
    locks = ET.SubElement(graphic_frame, q("a:graphicFrameLocks"))
    locks.set("noChangeAspect", "1")
    graphic = ET.SubElement(inline, q("a:graphic"))
    graphic_data = ET.SubElement(graphic, q("a:graphicData"))
    graphic_data.set("uri", PIC_NS)
    picture = ET.SubElement(graphic_data, q("pic:pic"))
    non_visual = ET.SubElement(picture, q("pic:nvPicPr"))
    c_nv_pr = ET.SubElement(non_visual, q("pic:cNvPr"))
    c_nv_pr.set("id", "0")
    c_nv_pr.set("name", image_path.name)
    ET.SubElement(non_visual, q("pic:cNvPicPr"))
    fill = ET.SubElement(picture, q("pic:blipFill"))
    blip = ET.SubElement(fill, q("a:blip"))
    blip.set(q("r:embed"), relationship_id)
    stretch = ET.SubElement(fill, q("a:stretch"))
    ET.SubElement(stretch, q("a:fillRect"))
    shape = ET.SubElement(picture, q("pic:spPr"))
    transform = ET.SubElement(shape, q("a:xfrm"))
    offset = ET.SubElement(transform, q("a:off"))
    offset.set("x", "0")
    offset.set("y", "0")
    ext = ET.SubElement(transform, q("a:ext"))
    ext.set("cx", str(cx))
    ext.set("cy", str(cy))
    geometry = ET.SubElement(shape, q("a:prstGeom"))
    geometry.set("prst", "rect")
    ET.SubElement(geometry, q("a:avLst"))


def format_equation_text(text: str) -> str:
    replacements = {
        "z0_half": "z₀,half",
        "z0": "z₀",
        "T_hat": "T̂",
        "F_fuse": "Ffuse",
        "L_total": "Ltotal",
        "L_seg": "Lseg",
        "L_BCE": "LBCE",
        "L_Dice": "LDice",
        "F_beta": "Fβ",
        "S_alpha": "Sα",
        "mE_phi": "mEφ",
        "maxE_phi": "maxEφ",
        "P_c": "Pc",
        "C_d": "Cd",
        "C_g": "Cg",
        "H_b": "Hb",
        "A_l": "Al",
        "Z_l": "Zl",
        "Q_l": "Ql",
        "S_l": "Sl",
        "U_l": "Ul",
        "D_l": "Dl",
        "F_l": "Fl",
        "gamma_b": "γb",
        "gamma_c": "γc",
        "lambda2": "λ2",
        "lambda3": "λ3",
        "lambda4": "λ4",
        "lambda": "λ",
        "gamma": "γ",
        "sigma": "σ",
        "alpha": "α",
        "beta": "β",
        "epsilon": "ε",
        "phi": "φ",
        "tau": "τ",
        "odot": "⊙",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def add_equation(body: ET.Element, text: str) -> None:
    paragraph = ET.SubElement(body, q("w:p"))
    props = ET.SubElement(paragraph, q("w:pPr"))
    alignment = ET.SubElement(props, q("w:jc"))
    alignment.set(q("w:val"), "center")
    math_paragraph = ET.SubElement(paragraph, q("m:oMathPara"))
    math = ET.SubElement(math_paragraph, q("m:oMath"))
    run = ET.SubElement(math, q("m:r"))
    text_node = ET.SubElement(run, q("m:t"))
    text_node.text = format_equation_text(text)


def add_cell_text(cell: ET.Element, text: str, *, header: bool = False) -> None:
    paragraph = ET.SubElement(cell, q("w:p"))
    set_paragraph_style(paragraph, "TableText")
    if header:
        add_inline_runs(paragraph, f"**{text}**")
    else:
        add_inline_runs(paragraph, text)


def add_table(body: ET.Element, rows: list[list[str]]) -> None:
    if not rows:
        return
    table = ET.SubElement(body, q("w:tbl"))
    props = ET.SubElement(table, q("w:tblPr"))
    width = ET.SubElement(props, q("w:tblW"))
    width.set(q("w:w"), "9360")
    width.set(q("w:type"), "dxa")
    layout = ET.SubElement(props, q("w:tblLayout"))
    layout.set(q("w:type"), "fixed")
    borders = ET.SubElement(props, q("w:tblBorders"))
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = ET.SubElement(borders, q(f"w:{edge}"))
        border.set(q("w:val"), "single")
        border.set(q("w:sz"), "4")
        border.set(q("w:space"), "0")
        border.set(q("w:color"), "B7B7B7")
    cell_count = max(len(row) for row in rows)
    grid = ET.SubElement(table, q("w:tblGrid"))
    column_width = max(720, 9360 // cell_count)
    for _ in range(cell_count):
        col = ET.SubElement(grid, q("w:gridCol"))
        col.set(q("w:w"), str(column_width))
    for row_index, row in enumerate(rows):
        tr = ET.SubElement(table, q("w:tr"))
        if row_index == 0:
            tr_props = ET.SubElement(tr, q("w:trPr"))
            cant_split = ET.SubElement(tr_props, q("w:cantSplit"))
            cant_split.set(q("w:val"), "1")
        for value in row:
            cell = ET.SubElement(tr, q("w:tc"))
            cell_props = ET.SubElement(cell, q("w:tcPr"))
            cell_width = ET.SubElement(cell_props, q("w:tcW"))
            cell_width.set(q("w:w"), str(column_width))
            cell_width.set(q("w:type"), "dxa")
            if row_index == 0:
                shading = ET.SubElement(cell_props, q("w:shd"))
                shading.set(q("w:fill"), "D9E2F3")
            margins = ET.SubElement(cell_props, q("w:tcMar"))
            for side in ("top", "left", "bottom", "right"):
                margin = ET.SubElement(margins, q(f"w:{side}"))
                margin.set(q("w:w"), "70")
                margin.set(q("w:type"), "dxa")
            add_cell_text(cell, value, header=row_index == 0)


def parse_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        if not line.strip().startswith("|"):
            continue
        values = [part.strip() for part in line.strip().strip("|").split("|")]
        if values and all(re.fullmatch(r":?-{3,}:?", value) for value in values):
            continue
        rows.append(values)
    return rows


def parse_markdown(markdown: str) -> list[tuple[str, object]]:
    lines = markdown.splitlines()
    blocks: list[tuple[str, object]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("```"):
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            block_kind = "equation" if any("=" in code_line for code_line in code_lines) else "code"
            blocks.append((block_kind, code_lines))
            continue
        image_match = re.match(r"^!\[(.*?)\]\(([^)]+)\)$", line.strip())
        if image_match:
            blocks.append(("image", (image_match.group(2), image_match.group(1))))
            index += 1
            continue
        if line.startswith(">"):
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].startswith(">"):
                quote_lines.append(lines[index][1:].lstrip())
                index += 1
            blocks.append(("quote", " ".join(quote_lines)))
            continue
        if line.lstrip().startswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            blocks.append(("table", parse_table(table_lines)))
            continue
        if re.match(r"^(#{1,3})\s+", line):
            match = re.match(r"^(#{1,3})\s+(.*)$", line)
            assert match is not None
            blocks.append((f"h{len(match.group(1))}", match.group(2).strip()))
            index += 1
            continue
        if re.match(r"^\d+\.\s+", line):
            list_lines: list[str] = []
            while index < len(lines) and re.match(r"^\d+\.\s+", lines[index]):
                list_lines.append(re.sub(r"^\d+\.\s+", "", lines[index]))
                index += 1
            blocks.append(("numbered", list_lines))
            continue
        if re.match(r"^-\s+", line):
            list_lines = []
            while index < len(lines) and re.match(r"^-\s+", lines[index]):
                list_lines.append(re.sub(r"^-\s+", "", lines[index]))
                index += 1
            blocks.append(("bulleted", list_lines))
            continue
        paragraph_lines = [line.strip()]
        index += 1
        while index < len(lines):
            next_line = lines[index]
            if (
                not next_line.strip()
                or next_line.startswith(">")
                or next_line.startswith("```")
                or next_line.lstrip().startswith("|")
                or re.match(r"^(#{1,3})\s+", next_line)
                or re.match(r"^(\d+|-)\.\s+", next_line)
                or next_line.startswith("- ")
            ):
                break
            paragraph_lines.append(next_line.strip())
            index += 1
        blocks.append(("paragraph", " ".join(paragraph_lines)))
    return blocks


def styles_xml() -> bytes:
    root = ET.Element(q("w:styles"))
    doc_defaults = ET.SubElement(root, q("w:docDefaults"))
    run_defaults = ET.SubElement(doc_defaults, q("w:rPrDefault"))
    run_props = ET.SubElement(run_defaults, q("w:rPr"))
    fonts = ET.SubElement(run_props, q("w:rFonts"))
    fonts.set(q("w:ascii"), "Arial")
    fonts.set(q("w:hAnsi"), "Arial")
    size = ET.SubElement(run_props, q("w:sz"))
    size.set(q("w:val"), "22")

    def style(style_id: str, name: str, *, based_on: str = "Normal", size_value: str = "22", bold: bool = False) -> ET.Element:
        node = ET.SubElement(root, q("w:style"))
        node.set(q("w:type"), "paragraph")
        node.set(q("w:styleId"), style_id)
        name_node = ET.SubElement(node, q("w:name"))
        name_node.set(q("w:val"), name)
        based = ET.SubElement(node, q("w:basedOn"))
        based.set(q("w:val"), based_on)
        props = ET.SubElement(node, q("w:rPr"))
        style_size = ET.SubElement(props, q("w:sz"))
        style_size.set(q("w:val"), size_value)
        if bold:
            ET.SubElement(props, q("w:b"))
        return node

    normal = style("Normal", "Normal", size_value="22")
    normal_props = normal.find(q("w:pPr"))
    if normal_props is None:
        normal_props = ET.SubElement(normal, q("w:pPr"))
    spacing = ET.SubElement(normal_props, q("w:spacing"))
    spacing.set(q("w:after"), "150")
    spacing.set(q("w:line"), "276")
    spacing.set(q("w:lineRule"), "auto")

    for style_id, name, size_value in (("Title", "Title", "34"), ("Heading1", "Heading 1", "28"), ("Heading2", "Heading 2", "24"), ("Heading3", "Heading 3", "22")):
        node = style(style_id, name, size_value=size_value, bold=True)
        props = ET.SubElement(node, q("w:pPr"))
        keep = ET.SubElement(props, q("w:keepNext"))
        keep.set(q("w:val"), "1")

    caption = style("Caption", "Caption", size_value="20", bold=False)
    caption_props = ET.SubElement(caption, q("w:pPr"))
    caption_spacing = ET.SubElement(caption_props, q("w:spacing"))
    caption_spacing.set(q("w:before"), "100")
    caption_spacing.set(q("w:after"), "80")

    table_text = style("TableText", "Table Text", size_value="17")
    table_props = ET.SubElement(table_text, q("w:pPr"))
    table_spacing = ET.SubElement(table_props, q("w:spacing"))
    table_spacing.set(q("w:after"), "20")
    table_spacing.set(q("w:line"), "220")
    table_spacing.set(q("w:lineRule"), "auto")

    for style_id, name in (("ListBullet", "List Bullet"), ("ListNumber", "List Number")):
        node = style(style_id, name, size_value="22")
        props = ET.SubElement(node, q("w:pPr"))
        indent = ET.SubElement(props, q("w:ind"))
        indent.set(q("w:left"), "720")
        indent.set(q("w:hanging"), "360")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def document_xml(markdown: str, image_relationships: dict[str, str], input_root: Path) -> bytes:
    document = ET.Element(q("w:document"))
    body = ET.SubElement(document, q("w:body"))
    blocks = parse_markdown(markdown)
    for kind, value in blocks:
        if kind == "h1":
            add_paragraph(body, str(value), style="Title")
        elif kind == "h2":
            add_paragraph(body, str(value), style="Heading1")
        elif kind == "h3":
            add_paragraph(body, str(value), style="Heading2")
        elif kind == "paragraph":
            add_paragraph(body, str(value), style="Normal")
        elif kind == "image":
            image_path, _alt = value  # type: ignore[misc]
            source = input_root / str(image_path)
            image_number = list(image_relationships).index(str(image_path)) + 1
            add_image_paragraph(body, image_relationships[str(image_path)], source, image_number)
        elif kind == "quote":
            add_paragraph(body, str(value), style="Caption", blockquote=True)
        elif kind == "code":
            for line in value:  # type: ignore[union-attr]
                add_paragraph(body, str(line), style="Normal", code=True)
        elif kind == "equation":
            for line in value:  # type: ignore[union-attr]
                if line.strip():
                    add_equation(body, str(line))
        elif kind == "table":
            add_table(body, value)  # type: ignore[arg-type]
        elif kind == "numbered":
            for item_index, item in enumerate(value, start=1):  # type: ignore[union-attr]
                add_paragraph(body, f"{item_index}. {item}", style="ListNumber")
        elif kind == "bulleted":
            for item in value:  # type: ignore[union-attr]
                add_paragraph(body, f"- {item}", style="ListBullet")

    sect_pr = ET.SubElement(body, q("w:sectPr"))
    page_size = ET.SubElement(sect_pr, q("w:pgSz"))
    page_size.set(q("w:w"), "12240")
    page_size.set(q("w:h"), "15840")
    page_margins = ET.SubElement(sect_pr, q("w:pgMar"))
    page_margins.set(q("w:top"), "1080")
    page_margins.set(q("w:right"), "1080")
    page_margins.set(q("w:bottom"), "1080")
    page_margins.set(q("w:left"), "1080")
    page_margins.set(q("w:header"), "720")
    page_margins.set(q("w:footer"), "720")
    page_margins.set(q("w:gutter"), "0")
    return ET.tostring(document, encoding="utf-8", xml_declaration=True)


def relationships_xml() -> bytes:
    root = ET.Element(rel_q("Relationships"))
    relationship = ET.SubElement(root, rel_q("Relationship"))
    relationship.set("Id", "rId1")
    relationship.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument")
    relationship.set("Target", "word/document.xml")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def document_relationships_xml(image_relationships: dict[str, str]) -> bytes:
    root = ET.Element(rel_q("Relationships"))
    styles = ET.SubElement(root, rel_q("Relationship"))
    styles.set("Id", "rId1")
    styles.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles")
    styles.set("Target", "styles.xml")
    settings = ET.SubElement(root, rel_q("Relationship"))
    settings.set("Id", "rId2")
    settings.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings")
    settings.set("Target", "settings.xml")
    for image_path, relationship_id in image_relationships.items():
        image = ET.SubElement(root, rel_q("Relationship"))
        image.set("Id", relationship_id)
        image.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
        image.set("Target", f"media/{Path(image_path).name}")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def content_types_xml(image_extensions: set[str]) -> bytes:
    root = ET.Element("{http://schemas.openxmlformats.org/package/2006/content-types}Types")
    namespace = "{http://schemas.openxmlformats.org/package/2006/content-types}"
    default = ET.SubElement(root, f"{namespace}Default")
    default.set("Extension", "rels")
    default.set("ContentType", "application/vnd.openxmlformats-package.relationships+xml")
    default = ET.SubElement(root, f"{namespace}Default")
    default.set("Extension", "xml")
    default.set("ContentType", "application/xml")
    if "svg" in image_extensions:
        default = ET.SubElement(root, f"{namespace}Default")
        default.set("Extension", "svg")
        default.set("ContentType", "image/svg+xml")
    if "png" in image_extensions:
        default = ET.SubElement(root, f"{namespace}Default")
        default.set("Extension", "png")
        default.set("ContentType", "image/png")
    overrides = [
        ("/word/document.xml", "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"),
        ("/word/styles.xml", "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"),
        ("/word/settings.xml", "application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"),
        ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml"),
        ("/docProps/app.xml", "application/vnd.openxmlformats-officedocument.extended-properties+xml"),
    ]
    for part_name, content_type in overrides:
        override = ET.SubElement(root, f"{namespace}Override")
        override.set("PartName", part_name)
        override.set("ContentType", content_type)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def settings_xml() -> bytes:
    root = ET.Element(q("w:settings"))
    compatibility = ET.SubElement(root, q("w:compat"))
    version = ET.SubElement(compatibility, q("w:compatSetting"))
    version.set(q("w:name"), "compatibilityMode")
    version.set(q("w:uri"), "http://schemas.microsoft.com/office/word")
    version.set(q("w:val"), "15")
    zoom = ET.SubElement(root, q("w:zoom"))
    zoom.set(q("w:percent"), "100")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def core_properties_xml() -> bytes:
    root = ET.Element("{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}coreProperties")
    title = ET.SubElement(root, "{http://purl.org/dc/elements/1.1/}title")
    title.text = "Revised manuscript - ConvNeXt-Tiny U-Net for colorectal polyp segmentation"
    creator = ET.SubElement(root, "{http://purl.org/dc/elements/1.1/}creator")
    creator.text = "Authors"
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def app_properties_xml() -> bytes:
    root = ET.Element("{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Properties")
    application = ET.SubElement(root, "{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Application")
    application.text = "OpenCode dependency-free manuscript builder"
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build(input_path: Path, output_path: Path) -> None:
    markdown = input_path.read_text(encoding="utf-8")
    image_relationships: dict[str, str] = {}
    for kind, value in parse_markdown(markdown):
        if kind != "image":
            continue
        image_path, _alt = value  # type: ignore[misc]
        image_path = str(image_path)
        source = input_path.parent / image_path
        if not source.is_file():
            raise FileNotFoundError(f"Figure asset not found: {source}")
        if image_path not in image_relationships:
            image_relationships[image_path] = f"rId{len(image_relationships) + 3}"
    parts = {
        "[Content_Types].xml": content_types_xml({Path(image_path).suffix.lower().lstrip(".") for image_path in image_relationships}),
        "_rels/.rels": relationships_xml(),
        "word/document.xml": document_xml(markdown, image_relationships, input_path.parent),
        "word/_rels/document.xml.rels": document_relationships_xml(image_relationships),
        "word/styles.xml": styles_xml(),
        "word/settings.xml": settings_xml(),
        "docProps/core.xml": core_properties_xml(),
        "docProps/app.xml": app_properties_xml(),
    }
    for image_path in image_relationships:
        parts[f"word/media/{Path(image_path).name}"] = (input_path.parent / image_path).read_bytes()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: python scripts/build_revised_manuscript_docx.py INPUT.md OUTPUT.docx")
        return 2
    input_path = Path(argv[1])
    output_path = Path(argv[2])
    if not input_path.is_file():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1
    build(input_path, output_path)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
