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

def _clean_product_title(product_title: str) -> str:
    """Remove brand prefix and concentration suffix for card display."""
    import re
    short = product_title
    for brand in ["Les Mignardises by Jousset ", "Jousset Parfums ", "Plume Impression ", "Silona Paris "]:
        if short.lower().startswith(brand.lower()):
            short = short[len(brand):]
            break
    short = re.sub(r'\s+(eau de parfum|extrait de parfum|le parfum|edp|edt)\s*$', '', short, flags=re.IGNORECASE).strip()
    short = re.sub(r'\s+parfum\s*$', '', short, flags=re.IGNORECASE).strip()
    return short


def generate_recto(product_title: str, notes: dict, note_images: dict) -> Image.Image:
    """
    Generate recto card with pyramid olfactive — FULL SIZE layout.
    notes: {"tete": "Vanille", "coeur": "Santal", "fond": "Musc", "tete_sec": [...], "coeur_sec": [...], "fond_sec": [...]}
    note_images: {"tete": Image, "coeur": Image, "fond": Image}
    """
    card = Image.new("RGB", (CARD_W, CARD_H), BG_COLOR)
    draw = ImageDraw.Draw(card)

    short_title = _clean_product_title(product_title)

    # Header: brand + title
    _text_center(draw, "PLANÈTEBEAUTY", 20, _get_font(16), ACCENT_COLOR)
    _text_center(draw, short_title, 50, _get_font(38, bold=True), TEXT_COLOR)

    # Thin divider
    draw.line([(80, 100), (CARD_W - 80, 100)], fill=LINE_COLOR, width=1)

    # "Pyramide Olfactive" label
    _text_center(draw, "PYRAMIDE OLFACTIVE", 112, _get_font(18), MUTED_COLOR)

    # Three BIG note circles in pyramid layout
    circle_r = 120  # radius — was 80, now 50% bigger

    # Pyramid positions: tête top-center, coeur left, fond right
    tete_y = 280
    coeur_y = 545
    fond_y = 545

    # Tête (center top)
    _draw_circle_note(card, draw, CARD_W // 2, tete_y, circle_r,
                       notes.get("tete", "—"), note_images.get("tete"), "TÊTE")

    # Coeur (left)
    _draw_circle_note(card, draw, CARD_W // 4 + 20, coeur_y, circle_r,
                       notes.get("coeur", "—"), note_images.get("coeur"), "CŒUR")

    # Fond (right)
    _draw_circle_note(card, draw, 3 * CARD_W // 4 - 20, fond_y, circle_r,
                       notes.get("fond", "—"), note_images.get("fond"), "FOND")

    # Pyramid connecting lines (subtle)
    tete_cx, tete_cy = CARD_W // 2, tete_y
    coeur_cx, coeur_cy = CARD_W // 4 + 20, coeur_y
    fond_cx, fond_cy = 3 * CARD_W // 4 - 20, fond_y
    line_c = (230, 225, 215)
    draw.line([(tete_cx, tete_cy + circle_r + 30), (coeur_cx + circle_r - 10, coeur_cy - circle_r - 30)], fill=line_c, width=1)
    draw.line([(tete_cx, tete_cy + circle_r + 30), (fond_cx - circle_r + 10, fond_cy - circle_r - 30)], fill=line_c, width=1)
    draw.line([(coeur_cx + circle_r + 10, coeur_cy), (fond_cx - circle_r - 10, fond_cy)], fill=line_c, width=1)

    # Secondary notes below
    font_sec = _get_font(18)
    sec_y = 730
    for key, label, yoff in [("tete_sec", "Tête", 0), ("coeur_sec", "Cœur", 24), ("fond_sec", "Fond", 48)]:
        secs = notes.get(key, [])
        if secs:
            text = f"{label} : {', '.join(secs[:3])}"
            _text_center(draw, text, sec_y + yoff, font_sec, MUTED_COLOR)

    # Divider before footer
    draw.line([(120, 860), (CARD_W - 120, 860)], fill=LINE_COLOR, width=1)

    # Footer: TRY ME badge
    _text_center(draw, "✦  TRY ME  ✦", 885, _get_font(26, bold=True), ACCENT_COLOR)

    # Tagline
    _text_center(draw, "Découvrez avant de craquer", 930, _get_font(18), MUTED_COLOR)

    # planetebeauty.com
    _text_center(draw, "planetebeauty.com", 965, _get_font(16), ACCENT_COLOR)

    return card


# ── VERSO TEMPLATE GENERATION ──

def generate_verso_template(product_title: str, product_handle: str, logo_img: Optional[Image.Image] = None) -> Image.Image:
    """
    Generate verso template — FULL SIZE layout with big QR + code placeholder.
    """
    card = Image.new("RGB", (CARD_W, CARD_H), BG_COLOR)
    draw = ImageDraw.Draw(card)

    short_title = _clean_product_title(product_title)

    # Logo at top (big)
    if logo_img:
        logo_w = 500
        ratio = logo_w / logo_img.width
        logo_h = int(logo_img.height * ratio)
        resized_logo = logo_img.resize((logo_w, logo_h), Image.LANCZOS)
        paste_x = (CARD_W - logo_w) // 2
        if resized_logo.mode == "RGBA":
            card.paste(resized_logo, (paste_x, 30), resized_logo)
        else:
            card.paste(resized_logo, (paste_x, 30))
        logo_bottom = 30 + logo_h + 10
    else:
        _text_center(draw, "PLANÈTEBEAUTY", 50, _get_font(34, bold=True), TEXT_COLOR)
        logo_bottom = 100

    # Product name
    _text_center(draw, short_title, logo_bottom + 10, _get_font(28, bold=True), TEXT_COLOR)

    # Divider
    div_y = logo_bottom + 55
    draw.line([(80, div_y), (CARD_W - 80, div_y)], fill=LINE_COLOR, width=1)

    # "VOTRE CODE TRY ME" label
    _text_center(draw, "VOTRE CODE TRY ME", div_y + 18, _get_font(20), MUTED_COLOR)

    # Code placeholder box (BIG)
    code_box_y = div_y + 55
    code_box_h = 90
    box_margin = 120
    draw.rounded_rectangle([box_margin, code_box_y, CARD_W - box_margin, code_box_y + code_box_h],
                            radius=12, fill=(250, 247, 240), outline=ACCENT_COLOR, width=2)
    _text_center(draw, "CODE PROMO", code_box_y + 28, _get_font(30, bold=True), (200, 195, 185))

    # Instructions (compact)
    font_instr = _get_font(19)
    y_i = code_box_y + code_box_h + 25
    for text in [
        "Déduisez le montant de votre Try Me",
        "de l'achat du format standard.",
        "",
        "✓ Valable 30 jours",
        "✓ Usage unique",
        "✓ Cumulable avec le code PB5",
    ]:
        if text:
            _text_center(draw, text, y_i, font_instr, TEXT_COLOR if text.startswith("✓") else MUTED_COLOR)
        y_i += 30

    # QR Code (BIG — 280px)
    product_url = f"https://planetebeauty.com/products/{product_handle}"
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=2)
    qr.add_data(product_url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color=(30, 30, 30), back_color=(255, 253, 248)).convert("RGB")
    qr_size = 280
    qr_img = qr_img.resize((qr_size, qr_size), Image.LANCZOS)
    qr_y = y_i + 15
    card.paste(qr_img, ((CARD_W - qr_size) // 2, qr_y))

    # QR label
    _text_center(draw, "Scannez pour commander", qr_y + qr_size + 10, _get_font(18), MUTED_COLOR)

    # Footer
    _text_center(draw, "planetebeauty.com", 965, _get_font(18, bold=True), ACCENT_COLOR)

    return card


# ── STAMP CODE ON VERSO ──

def stamp_code_on_verso(verso_template: Image.Image, code: str, order_name: str = "") -> Image.Image:
    """Stamp the actual discount code onto the verso template.
    Finds the code placeholder box by scanning for the beige rectangle area."""
    card = verso_template.copy()
    draw = ImageDraw.Draw(card)

    # The code box position depends on logo height, so we search for it
    # by looking for the "CODE PROMO" placeholder text area
    # Safe approach: clear and redraw the box region
    # Box is at ~120px margin, somewhere between y=200-350
    # We scan for the beige-colored box
    box_margin = 120
    # Find the box by checking pixel colors
    box_y = None
    for y in range(150, 400):
        px = card.getpixel((box_margin + 5, y))
        if px[0] >= 248 and px[1] >= 245 and px[2] >= 238:  # beige fill
            box_y = y
            break

    if box_y is None:
        box_y = 250  # fallback

    box_h = 90
    draw.rounded_rectangle([box_margin, box_y, CARD_W - box_margin, box_y + box_h],
                            radius=12, fill=(250, 247, 240), outline=ACCENT_COLOR, width=2)

    font_code = _get_font(38, bold=True)
    _text_center(draw, code, box_y + 22, font_code, ACCENT_COLOR)

    # Order reference
    if order_name:
        _text_center(draw, f"Réf: {order_name}", 940, _get_font(14), MUTED_COLOR)

    return card


# ── INDIVIDUAL CARD PDF (pre-cut cards, printed from top tray) ──

def generate_single_card_pdf(recto: Image.Image, verso: Image.Image, output_path: str):
    """
    Generate PDF with 1 card: page 1 = recto, page 2 = verso.
    Card is printed on pre-cut cardstock (~90x85mm) from printer top tray.
    """
    recto.save(output_path, "PDF", resolution=DPI, save_all=True, append_images=[verso])
    logger.info(f"Single card PDF: {output_path}")


def generate_multi_card_pdf(cards: list[tuple[Image.Image, Image.Image]], output_path: str):
    """
    Generate PDF with multiple cards: alternating recto/verso pages.
    Each card = 2 pages (recto then verso). Printer does auto recto/verso.
    """
    if not cards:
        return

    pages = []
    for recto, verso in cards:
        pages.append(recto)
        pages.append(verso)

    if pages:
        pages[0].save(output_path, "PDF", resolution=DPI, save_all=True, append_images=pages[1:])
        logger.info(f"Multi-card PDF: {output_path} ({len(cards)} cards, {len(pages)} pages)")


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
            note_images[key] = _download_image(url, (240, 240))

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
    Generate PDF for an order (individual pre-cut cards, recto/verso auto).
    cards_data: [{"product_id": "...", "code": "TM-...", "order_name": "#1234"}]
    Each card = 2 pages (recto + verso). Printer handles double-sided auto.
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
        generate_multi_card_pdf(cards, output_path)
        return True
    return False
