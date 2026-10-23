# -*- coding: utf-8 -*-
"""生成「图片型(无文本层)」扫描件样例，用于演示 MinerU 的 OCR 能力。"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "data" / "扫描件_检索增强生成综述.pdf"

FONT_TITLE = "C:/Windows/Fonts/simhei.ttf"
FONT_BODY = "C:/Windows/Fonts/simfang.ttf"

W, H = 1240, 1754  # A4 @ 150 DPI


def wrap(draw, text, font, max_width):
    lines, current = [], ""
    for ch in text:
        if draw.textlength(current + ch, font=font) > max_width:
            lines.append(current)
            current = ch
        else:
            current += ch
    if current:
        lines.append(current)
    return lines


def page_1():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title = ImageFont.truetype(FONT_TITLE, 44)
    body = ImageFont.truetype(FONT_BODY, 28)

    d.text((W // 2, 80), "检索增强生成综述", font=title, fill="black", anchor="ma")
    d.line([(100, 140), (W - 100, 140)], fill="black", width=2)

    paragraphs = [
        "摘要：检索增强生成（Retrieval-Augmented Generation, RAG）通过在生成阶段引入外部知识库，有效缓解了大语言模型的幻觉问题，并显著提升了知识密集型任务的可信度。",
        "一、引言。大语言模型虽然具备强大的生成能力，但其训练数据存在时间截止与覆盖不足的问题。当用户提出超出模型知识范围的问题时，模型容易产生看似合理却与事实不符的回答。",
        "二、核心流程。RAG 的基本流程包括：文档加载与切分、向量化、相似度检索、提示拼接与生成。其中向量维度与嵌入模型必须保持一致，否则插入向量库时会直接报错。",
        "三、评价指标。常用的评估维度包括忠实度（Faithfulness）、答案相关性（Answer Relevancy）与上下文精度（Context Precision）。",
    ]
    y = 200
    for p in paragraphs:
        for line in wrap(d, p, body, W - 240):
            d.text((120, y), line, font=body, fill="black")
            y += 52
        y += 24

    img.save(str(OUT), "PDF", resolution=150.0)
    return img


def page_2():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title = ImageFont.truetype(FONT_TITLE, 40)
    cell = ImageFont.truetype(FONT_TITLE, 26)

    d.text((W // 2, 80), "四、主流方案对比", font=title, fill="black", anchor="ma")

    rows = [
        ("方案", "特点"),
        ("朴素 RAG", "检索后直接拼接上下文，实现简单"),
        ("混合检索", "向量 + BM25 关键字，兼顾语义与精确匹配"),
        ("重排序", "对召回结果二次排序，提升上下文精度"),
    ]
    x0, y0, cw, rh = 140, 180, 480, 76
    for i, (a, b) in enumerate(rows):
        top = y0 + i * rh
        d.rectangle([x0, top, x0 + 2 * cw, top + rh], outline="black", width=3)
        d.line([(x0 + cw, top), (x0 + cw, top + rh)], fill="black", width=3)
        d.text((x0 + 20, top + rh // 2), a, font=cell, fill="black", anchor="lm")
        d.text((x0 + cw + 20, top + rh // 2), b, font=cell, fill="black", anchor="lm")

    img.save(str(OUT), "PDF", resolution=150.0, append=True)


page_1()
page_2()
print("已生成：", OUT.name)
