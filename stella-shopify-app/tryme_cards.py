"""
Try Me Card Generator — PlanèteBeauty
Generates recto (pyramid olfactive) and verso (logo + QR + code) cards.
Pre-generates templates per product, stamps code at order time, outputs A4 PDF.
"""

import io
import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional

import httpx
import qrcode
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("stella.tryme_cards")

# ── Card dimensions (px at 300 DPI) ──
# Physical card ~90x85mm → at 300 DPI: 1063x1004 px
# We use round numbers close to this
CARD_W = 1060
CARD_H = 1000
DPI = 300

# A4 at 300 DPI: 2480x3508 px
A4_W = 2480
A4_H = 3508

# Grid: 2 cols x 3 rows
COLS = 2
ROWS = 3

# Margins for crop marks
MARGIN_X = (A4_W - COLS * CARD_W) // (COLS + 1)  # ~120px between cards
MARGIN_Y = (A4_H - ROWS * CARD_H) // (ROWS + 1)  # ~170px between cards

# Colors
BG_COLOR = (255, 253, 248)      # warm white
TEXT_COLOR = (30, 30, 30)        # near black
ACCENT_COLOR = (196, 149, 106)  # gold/bronze
MUTED_COLOR = (140, 140, 140)   # grey
CIRCLE_BG = (245, 240, 232)     # light beige for note circles
LINE_COLOR = (220, 215, 205)    # subtle divider

# Paths
CARDS_DIR = Path("static/tryme-cards")
FONTS_DIR = Path("static/fonts")

# ── Font helpers ──
def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Get font, fallback to default if not found."""
    try:
        if bold:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except (OSError, IOError):
        try:
            if bold:
                return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _text_center(draw: ImageDraw.Draw, text: str, y: int, font: ImageFont.FreeTypeFont, fill, card_w: int = CARD_W):
    """Draw centered text."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (card_w - tw) // 2
    draw.text((x, y), text, font=font, fill=fill)


def _download_image(url: str, size: tuple = (160, 160)) -> Optional[Image.Image]:
    """Download and resize an image from URL."""
    try:
        r = httpx.get(url, timeout=10, follow_redirects=True)
        if r.status_code == 200:
            img = Image.open(io.BytesIO(r.content)).convert("RGBA")
            img = img.resize(size, Image.LANCZOS)
            return img
    except Exception as e:
        logger.warning(f"Failed to download image {url}: {e}")
    return None


def _draw_circle_note(card: Image.Image, draw: ImageDraw.Draw, cx: int, cy: int, radius: int,
                       note_name: str, note_img: Optional[Image.Image], label: str):
    """Draw a circular note with image and label."""
    font_label = _get_font(20)
    font_note = _get_font(22, bold=True)

    # Draw label above (TÊTE, CŒUR, FOND)
    bbox = draw.textbbox((0, 0), label, font=font_label)
    lw = bbox[2] - bbox[0]
    draw.text((cx - lw // 2, cy - radius - 35), label, font=font_label, fill=MUTED_COLOR)

    # Draw circle background
    x0, y0 = cx - radius, cy - radius
    x1, y1 = cx + radius, cy + radius
    draw.ellipse([x0, y0, x1, y1], fill=CIRCLE_BG, outline=LINE_COLOR, width=2)

    # Paste note image inside circle (with circular mask)
    if note_img:
        img_size = radius * 2 - 16
        resized = note_img.resize((img_size, img_size), Image.LANCZOS)
        # Create circular mask
        mask = Image.new("L", (img_size, img_size), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse([0, 0, img_size - 1, img_size - 1], fill=255)
        card.paste(resized, (cx - img_size // 2, cy - img_size // 2), mask)

    # Draw note name below circle
    bbox = draw.textbbox((0, 0), note_name, font=font_note)
    nw = bbox[2] - bbox[0]
    draw.text((cx - nw // 2, cy + radius + 8), note_name, font=font_note, fill=TEXT_COLOR)


# ── RECTO GENERATION ──

def generate_recto(product_title: str, notes: dict, note_images: dict) -> Image.Image:
    """
    Generate recto card with pyramid olfactive.
    notes: {"tete": "Vanille", "coeur": "Santal", "fond": "Musc", "tete_sec": [...], "coeur_sec": [...], "fond_sec": [...]}
    note_images: {"tete": Image, "coeur": Image, "fond": Image}
    """
    card = Image.new("RGB", (CARD_W, CARD_H), BG_COLOR)
    draw = ImageDraw.Draw(card)

    # Title
    font_title = _get_font(36, bold=True)
    font_subtitle = _get_font(22)

    # Clean product title (remove brand prefix for card)
    short_title = product_title
    for brand in ["Jousset Parfums ", "Plume Impression ", "Silona Paris ",
                   "Les Mignardises by Jousset "]:
        if short_title.startswith(brand):
            short_title = short_title[len(brand):]
            break

    # Remove concentration suffix
    import re
    short_title = re.sub(r'\s*(eau de parfum|extrait de parfum|le parfum|parfum|edp).*$', '', short_title, flags=re.IGNORECASE).strip()

    _text_center(draw, "PLANÈTEBEAUTY", 30, _get_font(18), ACCENT_COLOR)
    _text_center(draw, short_title, 65, font_title, TEXT_COLOR)

    # Divider
    draw.line([(CARD_W // 4, 120), (3 * CARD_W // 4, 120)], fill=LINE_COLOR, width=2)

    # "Pyramide Olfactive" label
    _text_center(draw, "Pyramide Olfactive", 140, _get_font(24), MUTED_COLOR)

    # Three note circles in pyramid layout
    circle_r = 80  # radius
    top_y = 310     # tête
    mid_y = 530     # coeur
    bot_y = 750     # fond

    # Tête (center top)
    _draw_circle_note(card, draw, CARD_W // 2, top_y, circle_r,
                       notes.get("tete", "—"), note_images.get("tete"), "TÊTE")

    # Coeur (left-center)
    _draw_circle_note(card, draw, CARD_W // 3, mid_y, circle_r,
                       notes.get("coeur", "—"), note_images.get("coeur"), "CŒUR")

    # Fond (right-center)
    _draw_circle_note(card, draw, 2 * CARD_W // 3, bot_y, circle_r,
                       notes.get("fond", "—"), note_images.get("fond"), "FOND")

    # Secondary notes as small text
    font_sec = _get_font(16)
    sec_y = 900
    sec_parts = []
    for key, label in [("tete_sec", "Tête"), ("coeur_sec", "Cœur"), ("fond_sec", "Fond")]:
        secs = notes.get(key, [])
        if secs:
            sec_parts.append(f"{label}: {', '.join(secs)}")
    if sec_parts:
        sec_text = " · ".join(sec_parts)
        if len(sec_text) > 70:
            sec_text = sec_text[:67] + "..."
        _text_center(draw, sec_text, sec_y, font_sec, MUTED_COLOR)

    # "TRY ME" badge at bottom
    _text_center(draw, "TRY ME", 950, _get_font(20, bold=True), ACCENT_COLOR)

    return card


# ── VERSO TEMPLATE GENERATION ──

def generate_verso_template(product_title: str, product_handle: str, logo_img: Optional[Image.Image] = None) -> Image.Image:
    """
    Generate verso template with logo, QR code, and placeholder for discount code.
    """
    card = Image.new("RGB", (CARD_W, CARD_H), BG_COLOR)
    draw = ImageDraw.Draw(card)

    # Logo at top
    if logo_img:
        logo_w = 400
        ratio = logo_w / logo_img.width
        logo_h = int(logo_img.height * ratio)
        resized_logo = logo_img.resize((logo_w, logo_h), Image.LANCZOS)
        paste_x = (CARD_W - logo_w) // 2
        if resized_logo.mode == "RGBA":
            card.paste(resized_logo, (paste_x, 40), resized_logo)
        else:
            card.paste(resized_logo, (paste_x, 40))
    else:
        _text_center(draw, "PLANÈTEBEAUTY", 60, _get_font(30, bold=True), TEXT_COLOR)

    # Product name
    font_product = _get_font(24, bold=True)
    _text_center(draw, product_title[:45], 200, font_product, TEXT_COLOR)

    # Divider
    draw.line([(CARD_W // 4, 250), (3 * CARD_W // 4, 250)], fill=LINE_COLOR, width=2)

    # "Votre code Try Me" label
    _text_center(draw, "Votre code Try Me", 280, _get_font(22), MUTED_COLOR)

    # Code placeholder (will be stamped at order time)
    # Draw a dashed box where the code goes
    code_box_y = 320
    code_box_h = 80
    box_x0 = CARD_W // 4
    box_x1 = 3 * CARD_W // 4
    draw.rectangle([box_x0, code_box_y, box_x1, code_box_y + code_box_h],
                    fill=(250, 247, 240), outline=ACCENT_COLOR, width=2)
    _text_center(draw, "CODE", code_box_y + 25, _get_font(28, bold=True), MUTED_COLOR)

    # Instructions
    font_instr = _get_font(18)
    y_instr = 430
    instructions = [
        "Utilisez ce code sur planetebeauty.com",
        "pour déduire le montant de votre Try Me",
        "de l'achat du format standard.",
        "",
        "Valable 30 jours · Usage unique",
        "Cumulable avec le code PB5",
    ]
    for line in instructions:
        _text_center(draw, line, y_instr, font_instr, MUTED_COLOR if line else TEXT_COLOR)
        y_instr += 28

    # QR Code
    product_url = f"https://planetebeauty.com/products/{product_handle}"
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=5, border=2)
    qr.add_data(product_url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color=(30, 30, 30), back_color=(255, 253, 248)).convert("RGB")
    qr_size = 200
    qr_img = qr_img.resize((qr_size, qr_size), Image.LANCZOS)
    card.paste(qr_img, ((CARD_W - qr_size) // 2, 680))

    # QR label
    _text_center(draw, "Scanner pour découvrir", 890, _get_font(16), MUTED_COLOR)
    _text_center(draw, "le format complet", 912, _get_font(16), MUTED_COLOR)

    # Footer
    _text_center(draw, "planetebeauty.com", 960, _get_font(18, bold=True), ACCENT_COLOR)

    return card


# ── STAMP CODE ON VERSO ──

def stamp_code_on_verso(verso_template: Image.Image, code: str, order_name: str = "") -> Image.Image:
    """Stamp the actual discount code onto the verso template."""
    card = verso_template.copy()
    draw = ImageDraw.Draw(card)

    # Overwrite the code placeholder area
    code_box_y = 320
    code_box_h = 80
    box_x0 = CARD_W // 4
    box_x1 = 3 * CARD_W // 4
    draw.rectangle([box_x0, code_box_y, box_x1, code_box_y + code_box_h],
                    fill=(250, 247, 240), outline=ACCENT_COLOR, width=2)

    font_code = _get_font(34, bold=True)
    _text_center(draw, code, code_box_y + 20, font_code, ACCENT_COLOR)

    # Order reference (small, bottom)
    if order_name:
        font_ref = _get_font(14)
        _text_center(draw, f"Réf: {order_name}", 940, font_ref, MUTED_COLOR)

    return card


# ── A4 PDF GENERATION ──

def generate_a4_pdf(cards: list[tuple[Image.Image, Image.Image]], output_path: str):
    """
    Generate A4 PDF with cards (recto/verso).
    cards: list of (recto_img, verso_img) tuples.
    Fills 6 cards per page. Recto on page 1, verso on page 2 (mirrored for double-sided).
    """
    if not cards:
        return

    # Pad to multiple of 6
    while len(cards) % 6 != 0:
        cards.append(cards[-1])  # duplicate last card to fill

    pages = []
    for batch_start in range(0, len(cards), 6):
        batch = cards[batch_start:batch_start + 6]

        # Recto page
        recto_page = Image.new("RGB", (A4_W, A4_H), (255, 255, 255))
        recto_draw = ImageDraw.Draw(recto_page)

        # Verso page (mirrored horizontally for double-sided printing)
        verso_page = Image.new("RGB", (A4_W, A4_H), (255, 255, 255))
        verso_draw = ImageDraw.Draw(verso_page)

        for idx, (recto, verso) in enumerate(batch):
            row = idx // COLS
            col = idx % COLS
            x = MARGIN_X + col * (CARD_W + MARGIN_X)
            y = MARGIN_Y + row * (CARD_H + MARGIN_Y)

            # Recto: normal order
            recto_page.paste(recto, (x, y))

            # Verso: mirror columns (col 0 → col 1, col 1 → col 0) for double-sided
            mirror_col = (COLS - 1) - col
            vx = MARGIN_X + mirror_col * (CARD_W + MARGIN_X)
            verso_page.paste(verso, (vx, y))

            # Crop marks on both pages
            mark_len = 30
            for page_draw, px, py in [(recto_draw, x, y), (verso_draw, vx if page_draw == verso_draw else x, y)]:
                pass  # Skip complex crop marks for now

        # Draw crop marks (simple corner marks)
        for page, page_draw in [(recto_page, recto_draw), (verso_page, verso_draw)]:
            for idx in range(len(batch)):
                row = idx // COLS
                col = idx % COLS
                if page == recto_page:
                    x = MARGIN_X + col * (CARD_W + MARGIN_X)
                else:
                    mirror_col = (COLS - 1) - (idx % COLS)
                    x = MARGIN_X + mirror_col * (CARD_W + MARGIN_X)
                y = MARGIN_Y + row * (CARD_H + MARGIN_Y)
                m = 20  # mark length
                c = (180, 180, 180)
                # Top-left
                page_draw.line([(x - 10, y), (x - 10 - m, y)], fill=c, width=1)
                page_draw.line([(x, y - 10), (x, y - 10 - m)], fill=c, width=1)
                # Top-right
                page_draw.line([(x + CARD_W + 10, y), (x + CARD_W + 10 + m, y)], fill=c, width=1)
                page_draw.line([(x + CARD_W, y - 10), (x + CARD_W, y - 10 - m)], fill=c, width=1)
                # Bottom-left
                page_draw.line([(x - 10, y + CARD_H), (x - 10 - m, y + CARD_H)], fill=c, width=1)
                page_draw.line([(x, y + CARD_H + 10), (x, y + CARD_H + 10 + m)], fill=c, width=1)
                # Bottom-right
                page_draw.line([(x + CARD_W + 10, y + CARD_H), (x + CARD_W + 10 + m, y + CARD_H)], fill=c, width=1)
                page_draw.line([(x + CARD_W, y + CARD_H + 10), (x + CARD_W, y + CARD_H + 10 + m)], fill=c, width=1)

        pages.append(recto_page)
        pages.append(verso_page)

    # Save as PDF
    if pages:
        pages[0].save(output_path, "PDF", resolution=DPI, save_all=True, append_images=pages[1:])
        logger.info(f"A4 PDF generated: {output_path} ({len(pages)} pages, {len(cards)} cards)")


# ── PRE-GENERATION ──

async def pregenerate_card_assets(product_id: str, product_title: str, product_handle: str,
                                    notes: dict, note_image_urls: dict, logo_url: str) -> dict:
    """
    Pre-generate recto + verso template for a product.
    Stores as PNG in static/tryme-cards/.
    Returns paths.
    """
    CARDS_DIR.mkdir(parents=True, exist_ok=True)

    # Download note images
    note_images = {}
    for key in ["tete", "coeur", "fond"]:
        url = note_image_urls.get(key)
        if url:
            note_images[key] = _download_image(url, (160, 160))

    # Download logo
    logo_img = _download_image(logo_url, (400, 134)) if logo_url else None

    # Generate recto
    recto = generate_recto(product_title, notes, note_images)
    recto_path = CARDS_DIR / f"recto_{product_id}.png"
    recto.save(str(recto_path), "PNG")

    # Generate verso template
    verso = generate_verso_template(product_title, product_handle, logo_img)
    verso_path = CARDS_DIR / f"verso_{product_id}.png"
    verso.save(str(verso_path), "PNG")

    logger.info(f"Card assets pre-generated for {product_title} ({product_id})")
    return {"recto": str(recto_path), "verso": str(verso_path)}


def get_card_paths(product_id: str) -> tuple[Optional[str], Optional[str]]:
    """Get pre-generated card paths for a product."""
    recto = CARDS_DIR / f"recto_{product_id}.png"
    verso = CARDS_DIR / f"verso_{product_id}.png"
    r = str(recto) if recto.exists() else None
    v = str(verso) if verso.exists() else None
    return r, v


def generate_order_pdf(cards_data: list[dict], output_path: str):
    """
    Generate A4 PDF for an order.
    cards_data: [{"product_id": "...", "code": "TM-...", "order_name": "#1234"}]
    """
    cards = []
    for cd in cards_data:
        recto_path, verso_path = get_card_paths(cd["product_id"])
        if not recto_path or not verso_path:
            logger.warning(f"Missing card assets for product {cd['product_id']}")
            continue
        recto = Image.open(recto_path)
        verso_template = Image.open(verso_path)
        verso = stamp_code_on_verso(verso_template, cd["code"], cd.get("order_name", ""))
        cards.append((recto, verso))

    if cards:
        generate_a4_pdf(cards, output_path)
        return True
    return False
