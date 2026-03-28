"""
Try Me Card Generator — PlaneteBeauty
HD version: 2120x2000 px (300 DPI), Inter + Playfair Display fonts.
Recto: bottle left (ratio preserved) + pyramid olfactive right (big circles).
Verso: logo + QR code + code promo placeholder.
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
from PIL import Image, ImageDraw, ImageFont, ImageFilter

logger = logging.getLogger("stella.tryme_cards")

# ── Card dimensions (2x resolution for HD print) ──
# Physical card ~90x85mm @ 300 DPI = 1063x1004 → doubled to 2120x2000
CARD_W = 2120
CARD_H = 2000
DPI = 300

# A4 at 300 DPI
A4_W = 2480
A4_H = 3508
COLS = 2
ROWS = 3

# Colors
BG_COLOR = (255, 253, 248)
TEXT_COLOR = (30, 30, 30)
ACCENT_COLOR = (200, 152, 78)   # #C8984E gold
MUTED_COLOR = (130, 125, 118)
CIRCLE_BG = (243, 238, 228)
CIRCLE_BORDER = (210, 200, 185)
LINE_COLOR = (220, 215, 205)

# Paths
CARDS_DIR = Path("static/tryme-cards")
FONTS_DIR = Path("static/fonts")


# ── Font helpers (Inter + Playfair) ──

_font_cache = {}

def _get_font(size: int, style: str = "regular") -> ImageFont.FreeTypeFont:
    """Load font with caching. Styles: regular, bold, serif, serif-bold."""
    key = (size, style)
    if key in _font_cache:
        return _font_cache[key]

    font_map = {
        "regular": "Inter.ttf",
        "bold": "Inter.ttf",
        "serif": "PlayfairDisplay.ttf",
        "serif-bold": "PlayfairDisplay.ttf",
    }
    fname = font_map.get(style, "Inter.ttf")

    # Try multiple paths
    paths = [
        FONTS_DIR / fname,
        Path(f"/app/static/fonts/{fname}"),
        Path(f"static/fonts/{fname}"),
    ]

    for p in paths:
        try:
            font = ImageFont.truetype(str(p), size)
            _font_cache[key] = font
            return font
        except (OSError, IOError):
            continue

    # Fallback to system DejaVu
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        _font_cache[key] = font
        return font
    except (OSError, IOError):
        font = ImageFont.load_default()
        _font_cache[key] = font
        return font


def _text_center(draw: ImageDraw.Draw, text: str, y: int, font, fill, card_w: int = CARD_W):
    """Draw centered text."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (card_w - tw) // 2
    draw.text((x, y), text, font=font, fill=fill)


def _text_width(draw: ImageDraw.Draw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _download_image(url: str, max_size: int = 500) -> Optional[Image.Image]:
    """Download image, preserve original ratio, fit within max_size square."""
    try:
        r = httpx.get(url, timeout=15, follow_redirects=True)
        if r.status_code == 200:
            img = Image.open(io.BytesIO(r.content)).convert("RGBA")
            # Preserve ratio
            ratio = min(max_size / img.width, max_size / img.height)
            if ratio < 1:
                new_w = int(img.width * ratio)
                new_h = int(img.height * ratio)
                img = img.resize((new_w, new_h), Image.LANCZOS)
            return img
    except Exception as e:
        logger.warning(f"Failed to download image {url}: {e}")
    return None


def _draw_circle_note(card: Image.Image, draw: ImageDraw.Draw,
                       cx: int, cy: int, radius: int,
                       note_name: str, note_img: Optional[Image.Image], label: str):
    """Draw a circular note with image and label — HD version."""
    font_label = _get_font(28, "regular")
    font_note = _get_font(32, "bold")

    # Label above (TETE, COEUR, FOND)
    lw = _text_width(draw, label, font_label)
    draw.text((cx - lw // 2, cy - radius - 50), label, font=font_label, fill=ACCENT_COLOR)

    # Circle with subtle shadow effect
    shadow_offset = 4
    draw.ellipse([cx - radius + shadow_offset, cy - radius + shadow_offset,
                  cx + radius + shadow_offset, cy + radius + shadow_offset],
                 fill=(230, 225, 218))
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                 fill=CIRCLE_BG, outline=CIRCLE_BORDER, width=3)

    # Paste note image inside circle
    if note_img:
        img_size = int(radius * 1.7)
        resized = note_img.copy()
        ratio = min(img_size / resized.width, img_size / resized.height)
        new_w = int(resized.width * ratio)
        new_h = int(resized.height * ratio)
        resized = resized.resize((new_w, new_h), Image.LANCZOS)
        # Circular mask
        mask = Image.new("L", (new_w, new_h), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse([0, 0, new_w - 1, new_h - 1], fill=255)
        card.paste(resized, (cx - new_w // 2, cy - new_h // 2), mask)

    # Note name below
    nw = _text_width(draw, note_name, font_note)
    draw.text((cx - nw // 2, cy + radius + 12), note_name, font=font_note, fill=TEXT_COLOR)


def _clean_product_title(product_title: str) -> str:
    """Remove brand prefix and concentration suffix for card display."""
    import re
    short = product_title
    brands = [
        "Les Mignardises by Jousset ", "Jousset Parfums ", "Plume Impression ",
        "Silona Paris ", "Badar ", "FOMOWA ", "L'artisan Parfumeur ",
        "Essential Parfums ", "Goldfield & Banks ", "BDK Parfums ",
        "Bohoboco ", "Jul et Mad Paris ", "Maison Mataha ",
    ]
    for brand in brands:
        if short.lower().startswith(brand.lower()):
            short = short[len(brand):]
            break
    short = re.sub(r'\s+(eau de parfum|extrait de parfum|le parfum|edp|edt|parfum)\s*$', '', short, flags=re.IGNORECASE).strip()
    return short


# ══════════════════════ RECTO ══════════════════════

def generate_recto(product_title: str, notes: dict, note_images: dict,
                    product_img: Optional[Image.Image] = None) -> Image.Image:
    """Generate recto: bottle left (45%) + pyramid right (55%)."""
    card = Image.new("RGB", (CARD_W, CARD_H), BG_COLOR)
    draw = ImageDraw.Draw(card)
    short_title = _clean_product_title(product_title)

    # ── Header ──
    _text_center(draw, "PLANETEBEAUTY", 25, _get_font(28, "regular"), ACCENT_COLOR)

    # Title — serif for elegance
    title_font = _get_font(52, "serif-bold")
    # Truncate if too long
    if _text_width(draw, short_title, title_font) > CARD_W - 80:
        title_font = _get_font(42, "serif-bold")
    _text_center(draw, short_title, 70, title_font, TEXT_COLOR)

    # Gold line
    draw.line([(80, 140), (CARD_W - 80, 140)], fill=ACCENT_COLOR, width=2)

    # ── Layout: 45% left bottle, 55% right pyramid ──
    left_w = int(CARD_W * 0.45)
    right_x = left_w

    # ── Left: Product bottle (preserve ratio!) ──
    if product_img:
        max_h = 1500
        max_w = left_w - 80
        img = product_img.copy()
        # PRESERVE original proportions
        ratio = min(max_w / img.width, max_h / img.height)
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # Center in left column
        px = (left_w - new_w) // 2
        py = 180 + (max_h - new_h) // 2
        if img.mode == "RGBA":
            card.paste(img, (px, py), img)
        else:
            card.paste(img, (px, py))

    # ── Right: Pyramid olfactive ──
    pyramid_cx = right_x + (CARD_W - right_x) // 2
    circle_r = 155  # BIG circles

    # Pyramid label
    _text_center(draw, "PYRAMIDE OLFACTIVE", 165, _get_font(24, "regular"), MUTED_COLOR, card_w=(CARD_W - right_x))
    # Offset to right column center
    lbl_font = _get_font(24, "regular")
    lbl_w = _text_width(draw, "PYRAMIDE OLFACTIVE", lbl_font)
    draw.text((pyramid_cx - lbl_w // 2, 165), "PYRAMIDE OLFACTIVE", font=lbl_font, fill=MUTED_COLOR)

    # Three circles: tete=top, coeur=mid, fond=bottom
    tete_y = 420
    coeur_y = 900
    fond_y = 1380

    _draw_circle_note(card, draw, pyramid_cx, tete_y, circle_r,
                       notes.get("tete", "—"), note_images.get("tete"), "TETE")
    _draw_circle_note(card, draw, pyramid_cx, coeur_y, circle_r,
                       notes.get("coeur", "—"), note_images.get("coeur"), "COEUR")
    _draw_circle_note(card, draw, pyramid_cx, fond_y, circle_r,
                       notes.get("fond", "—"), note_images.get("fond"), "FOND")

    # Connecting gold lines
    draw.line([(pyramid_cx, tete_y + circle_r + 50), (pyramid_cx, coeur_y - circle_r - 50)],
              fill=ACCENT_COLOR, width=2)
    draw.line([(pyramid_cx, coeur_y + circle_r + 50), (pyramid_cx, fond_y - circle_r - 50)],
              fill=ACCENT_COLOR, width=2)

    # ── Footer ──
    draw.line([(80, 1750), (CARD_W - 80, 1750)], fill=LINE_COLOR, width=1)

    # Secondary notes
    font_sec = _get_font(26, "regular")
    sec_parts = []
    for key in ["tete_sec", "coeur_sec", "fond_sec"]:
        secs = notes.get(key, [])
        sec_parts.extend(secs[:2])
    if sec_parts:
        sec_text = " · ".join(sec_parts[:6])
        if _text_width(draw, sec_text, font_sec) > CARD_W - 120:
            sec_text = " · ".join(sec_parts[:4])
        _text_center(draw, sec_text, 1770, font_sec, MUTED_COLOR)

    # TRY ME badge
    _text_center(draw, "TRY ME", 1830, _get_font(48, "serif-bold"), ACCENT_COLOR)
    _text_center(draw, "Testez avant de craquer", 1900, _get_font(28, "regular"), MUTED_COLOR)
    _text_center(draw, "planetebeauty.com", 1950, _get_font(24, "regular"), ACCENT_COLOR)

    return card


# ══════════════════════ VERSO ══════════════════════

def generate_verso_template(product_title: str, product_handle: str,
                             logo_img: Optional[Image.Image] = None) -> Image.Image:
    """Generate verso template — HD layout."""
    card = Image.new("RGB", (CARD_W, CARD_H), BG_COLOR)
    draw = ImageDraw.Draw(card)
    short_title = _clean_product_title(product_title)

    # Logo at top
    logo_bottom = 60
    if logo_img:
        logo_w = 800
        ratio = logo_w / logo_img.width
        logo_h = int(logo_img.height * ratio)
        resized_logo = logo_img.resize((logo_w, logo_h), Image.LANCZOS)
        paste_x = (CARD_W - logo_w) // 2
        if resized_logo.mode == "RGBA":
            card.paste(resized_logo, (paste_x, 50), resized_logo)
        else:
            card.paste(resized_logo, (paste_x, 50))
        logo_bottom = 50 + logo_h + 20
    else:
        _text_center(draw, "PLANETEBEAUTY", 80, _get_font(56, "serif-bold"), TEXT_COLOR)
        logo_bottom = 160

    # Product name
    _text_center(draw, short_title, logo_bottom + 10, _get_font(44, "serif-bold"), TEXT_COLOR)

    # Gold divider
    div_y = logo_bottom + 75
    draw.line([(150, div_y), (CARD_W - 150, div_y)], fill=ACCENT_COLOR, width=2)

    # "VOTRE CODE TRY ME"
    _text_center(draw, "VOTRE CODE TRY ME", div_y + 30, _get_font(34, "regular"), MUTED_COLOR)

    # Code placeholder box
    code_box_y = div_y + 90
    code_box_h = 140
    box_margin = 200
    draw.rounded_rectangle([box_margin, code_box_y, CARD_W - box_margin, code_box_y + code_box_h],
                            radius=16, fill=(250, 247, 240), outline=ACCENT_COLOR, width=3)
    _text_center(draw, "CODE PROMO", code_box_y + 40, _get_font(50, "bold"), (210, 205, 195))

    # Instructions
    font_instr = _get_font(30, "regular")
    y_i = code_box_y + code_box_h + 50
    instructions = [
        "Deduisez le montant de votre Try Me",
        "sur le format standard de ce parfum.",
        "",
        "* Valable 30 jours",
        "* Usage unique",
        "* Cumulable avec le code PB5",
    ]
    for text in instructions:
        if text:
            color = TEXT_COLOR if text.startswith("*") else MUTED_COLOR
            _text_center(draw, text, y_i, font_instr, color)
        y_i += 48

    # QR Code
    product_url = f"https://planetebeauty.com/products/{product_handle}"
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=2)
    qr.add_data(product_url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color=(30, 30, 30), back_color=BG_COLOR).convert("RGB")
    qr_size = 420
    qr_img = qr_img.resize((qr_size, qr_size), Image.LANCZOS)
    qr_y = y_i + 30
    card.paste(qr_img, ((CARD_W - qr_size) // 2, qr_y))

    # QR label
    _text_center(draw, "Scannez pour commander", qr_y + qr_size + 15, _get_font(28, "regular"), MUTED_COLOR)

    # Footer
    _text_center(draw, "planetebeauty.com", CARD_H - 60, _get_font(30, "bold"), ACCENT_COLOR)

    return card


# ══════════════════════ STAMP CODE ══════════════════════

def stamp_code_on_verso(verso_template: Image.Image, code: str, order_name: str = "") -> Image.Image:
    """Stamp actual code onto verso template."""
    card = verso_template.copy()
    draw = ImageDraw.Draw(card)

    # Find the beige code box by scanning pixels
    box_margin = 200
    box_y = None
    for y in range(200, 700):
        px = card.getpixel((box_margin + 10, y))
        if px[0] >= 248 and px[1] >= 245 and px[2] >= 238:
            box_y = y
            break

    if box_y is None:
        box_y = 400

    box_h = 140
    draw.rounded_rectangle([box_margin, box_y, CARD_W - box_margin, box_y + box_h],
                            radius=16, fill=(250, 247, 240), outline=ACCENT_COLOR, width=3)

    font_code = _get_font(64, "bold")
    _text_center(draw, code, box_y + 30, font_code, ACCENT_COLOR)

    if order_name:
        _text_center(draw, f"Ref: {order_name}", CARD_H - 100, _get_font(24, "regular"), MUTED_COLOR)

    return card


# ══════════════════════ PDF GENERATION ══════════════════════

def generate_single_card_pdf(recto: Image.Image, verso: Image.Image, output_path: str):
    """PDF with 1 card: page 1 = recto, page 2 = verso (auto recto/verso print)."""
    # Scale down to physical card size for PDF
    card_w_pts = int(90 / 25.4 * 72)   # 90mm in points
    card_h_pts = int(85 / 25.4 * 72)   # 85mm in points

    r = recto.copy().resize((card_w_pts * 4, card_h_pts * 4), Image.LANCZOS)
    v = verso.copy().resize((card_w_pts * 4, card_h_pts * 4), Image.LANCZOS)

    r.save(output_path, "PDF", resolution=DPI, save_all=True, append_images=[v])
    logger.info(f"Single card PDF: {output_path}")


def generate_a4_pdf(cards: list[tuple[Image.Image, Image.Image]], output_path: str):
    """Generate A4 PDF with 6 cards per page, crop marks, recto/verso pages."""
    if not cards:
        return

    # Scale card to fit A4 grid
    cell_w = A4_W // COLS
    cell_h = A4_H // ROWS
    margin_x = (A4_W - COLS * cell_w) // 2
    margin_y = (A4_H - ROWS * cell_h) // 2

    pages = []

    # Process in batches of 6
    for batch_start in range(0, len(cards), 6):
        batch = cards[batch_start:batch_start + 6]

        # Recto page
        recto_page = Image.new("RGB", (A4_W, A4_H), (255, 255, 255))
        for i, (recto, verso) in enumerate(batch):
            col = i % COLS
            row = i // COLS
            x = margin_x + col * cell_w
            y = margin_y + row * cell_h
            scaled = recto.resize((cell_w - 20, cell_h - 20), Image.LANCZOS)
            recto_page.paste(scaled, (x + 10, y + 10))
        pages.append(recto_page)

        # Verso page (mirrored columns for double-sided)
        verso_page = Image.new("RGB", (A4_W, A4_H), (255, 255, 255))
        for i, (recto, verso) in enumerate(batch):
            col = (COLS - 1) - (i % COLS)  # Mirror for recto/verso alignment
            row = i // COLS
            x = margin_x + col * cell_w
            y = margin_y + row * cell_h
            scaled = verso.resize((cell_w - 20, cell_h - 20), Image.LANCZOS)
            verso_page.paste(scaled, (x + 10, y + 10))
        pages.append(verso_page)

    if pages:
        pages[0].save(output_path, "PDF", resolution=DPI, save_all=True, append_images=pages[1:])
        logger.info(f"A4 PDF: {output_path} ({len(cards)} cards)")


# ══════════════════════ PRE-GENERATION ══════════════════════

async def pregenerate_card_assets(product_id: str, product_title: str, product_handle: str,
                                    notes: dict, note_image_urls: dict, logo_url: str,
                                    product_image_url: str = None) -> dict:
    """Pre-generate recto + verso template for a product."""
    CARDS_DIR.mkdir(parents=True, exist_ok=True)

    # Download note images (bigger for HD)
    note_images = {}
    for key in ["tete", "coeur", "fond"]:
        url = note_image_urls.get(key)
        if url:
            note_images[key] = _download_image(url, 400)

    # Download product bottle (preserve ratio, max 800px)
    product_img = None
    if product_image_url:
        product_img = _download_image(product_image_url, 800)

    # Download logo
    logo_img = None
    if logo_url:
        logo_img = _download_image(logo_url, 1000)

    # Generate recto
    recto = generate_recto(product_title, notes, note_images, product_img)
    recto_path = CARDS_DIR / f"recto_{product_id}.png"
    recto.save(str(recto_path), "PNG")

    # Generate verso template
    verso = generate_verso_template(product_title, product_handle, logo_img)
    verso_path = CARDS_DIR / f"verso_{product_id}.png"
    verso.save(str(verso_path), "PNG")

    logger.info(f"Pre-generated cards for {product_title} ({product_id})")
    return {
        "product_id": product_id,
        "recto": str(recto_path),
        "verso": str(verso_path),
    }


def get_card_paths(product_id: str) -> dict:
    """Get paths for pre-generated card assets."""
    recto = CARDS_DIR / f"recto_{product_id}.png"
    verso = CARDS_DIR / f"verso_{product_id}.png"
    return {
        "recto": str(recto) if recto.exists() else None,
        "verso": str(verso) if verso.exists() else None,
    }


def generate_order_pdf(cards_data: list, output_path: str) -> bool:
    """Generate PDF for an order with all Try Me cards.
    cards_data = list of {"recto": path, "verso": path, "code": str, "order_name": str}
    """
    try:
        card_pairs = []
        for cd in cards_data:
            recto_path = cd.get("recto")
            verso_path = cd.get("verso")
            code = cd.get("code", "")
            order_name = cd.get("order_name", "")

            if not recto_path or not verso_path:
                continue
            if not Path(recto_path).exists() or not Path(verso_path).exists():
                continue

            recto = Image.open(recto_path).convert("RGB")
            verso_tpl = Image.open(verso_path).convert("RGB")
            verso = stamp_code_on_verso(verso_tpl, code, order_name)
            card_pairs.append((recto, verso))

        if not card_pairs:
            return False

        if len(card_pairs) == 1:
            generate_single_card_pdf(card_pairs[0][0], card_pairs[0][1], output_path)
        else:
            generate_a4_pdf(card_pairs, output_path)
        return True
    except Exception as e:
        logger.error(f"Error generating order PDF: {e}")
        return False
