---
name: frontend-design
description: Use when working on UI/UX for the Stella Shopify app — HTML/CSS/JS in stella-shopify-app/static, designing or refactoring components, building cards or layouts, fixing visual bugs, or adding new pages. Provides design tokens, accessibility rules, and component patterns aligned with Shopify Polaris and PlanèteBeauty branding.
---

# Frontend Design — Stella / PlanèteBeauty

This skill drives all visual and UX decisions for the Stella Shopify app
(`stella-shopify-app/static/`). Apply it whenever you write or modify HTML,
CSS, JS, or templates that render to the merchant or customer.

## 1. Design tokens

Use these tokens instead of hard-coded values. Define them once in CSS custom
properties at the root of the stylesheet.

```css
:root {
  /* Brand */
  --pb-primary: #0a3d62;        /* PlanèteBeauty deep blue */
  --pb-primary-hover: #082e49;
  --pb-accent: #f5a623;         /* warm gold */
  --pb-bg: #fafbfc;
  --pb-surface: #ffffff;
  --pb-border: #e1e4e8;

  /* Text */
  --pb-text: #1a1a1a;
  --pb-text-muted: #6b7280;
  --pb-text-inverse: #ffffff;

  /* Status */
  --pb-success: #008060;        /* Polaris green */
  --pb-warning: #b98900;
  --pb-critical: #d72c0d;

  /* Spacing — 4px scale */
  --pb-space-1: 4px;
  --pb-space-2: 8px;
  --pb-space-3: 12px;
  --pb-space-4: 16px;
  --pb-space-5: 24px;
  --pb-space-6: 32px;
  --pb-space-8: 48px;

  /* Radius */
  --pb-radius-sm: 4px;
  --pb-radius-md: 8px;
  --pb-radius-lg: 12px;

  /* Typography */
  --pb-font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --pb-text-xs: 12px;
  --pb-text-sm: 14px;
  --pb-text-base: 16px;
  --pb-text-lg: 18px;
  --pb-text-xl: 24px;
  --pb-text-2xl: 32px;

  /* Elevation */
  --pb-shadow-sm: 0 1px 2px rgba(0,0,0,0.06);
  --pb-shadow-md: 0 4px 12px rgba(0,0,0,0.08);
  --pb-shadow-lg: 0 12px 32px rgba(0,0,0,0.12);
}
```

## 2. Layout rules

- Use **CSS Grid** for page-level layouts and **Flexbox** for component-level
  alignment. Never use floats.
- Max content width: `1200px`, centered with `margin-inline: auto` and
  `padding-inline: var(--pb-space-5)`.
- Vertical rhythm: stack siblings with a consistent gap (`gap: var(--pb-space-4)`)
  rather than per-element margins.
- Mobile-first: write base styles for ≤640px, then add `@media (min-width: 768px)`
  and `@media (min-width: 1024px)` breakpoints.

## 3. Components

### Card (default container)
```css
.pb-card {
  background: var(--pb-surface);
  border: 1px solid var(--pb-border);
  border-radius: var(--pb-radius-md);
  padding: var(--pb-space-5);
  box-shadow: var(--pb-shadow-sm);
}
```

### Button
```css
.pb-btn {
  display: inline-flex;
  align-items: center;
  gap: var(--pb-space-2);
  padding: var(--pb-space-3) var(--pb-space-4);
  border-radius: var(--pb-radius-sm);
  font: 500 var(--pb-text-sm)/1 var(--pb-font);
  cursor: pointer;
  transition: background 120ms ease;
  border: 1px solid transparent;
}
.pb-btn--primary { background: var(--pb-primary); color: var(--pb-text-inverse); }
.pb-btn--primary:hover { background: var(--pb-primary-hover); }
.pb-btn--secondary {
  background: var(--pb-surface);
  color: var(--pb-text);
  border-color: var(--pb-border);
}
.pb-btn:focus-visible {
  outline: 2px solid var(--pb-primary);
  outline-offset: 2px;
}
.pb-btn:disabled { opacity: 0.5; cursor: not-allowed; }
```

### Input
- Always pair with a `<label>` (use `for=` or wrap).
- Min height `40px`, padding `var(--pb-space-3)`, border `1px solid var(--pb-border)`.
- Focus state: border color `var(--pb-primary)`, no `outline: none` without a replacement.

## 4. Accessibility — non-negotiable

- Every interactive element must be reachable by `Tab` and have a visible focus
  ring.
- All images need `alt`. Decorative images use `alt=""`.
- Color contrast: ≥ 4.5:1 for body text, ≥ 3:1 for ≥18px or bold.
- Don't convey information by color alone — pair with icon or label.
- Use semantic HTML (`<button>`, `<nav>`, `<main>`, `<header>`). Avoid `<div>`
  with `onclick`.
- Support `prefers-reduced-motion`: disable non-essential transitions.

## 5. State & feedback

- Loading: skeleton placeholders, not spinners, for content > 200ms.
- Empty state: short title + one-sentence helper + primary action.
- Error: red border + helper text below the field. Never use a modal for inline
  errors.
- Success: toast for non-blocking confirmation, banner for in-page state.

## 6. Workflow when editing UI

1. Read the existing stylesheet and reuse tokens / components first. Don't add
   parallel styles.
2. If a new pattern is needed in 2+ places, lift it to a reusable class.
3. Verify with the dev server — load the page, exercise the golden path on
   desktop and mobile widths, check focus order with `Tab`, and check
   `prefers-reduced-motion`.
4. Keep CSS specificity flat: one class per rule, no `!important` unless
   overriding a third-party style.

## 7. Things to avoid

- Inline styles in HTML/JSX (except dynamic values bound to a token).
- Pixel values outside the spacing scale.
- New color values that don't map to a token.
- Magic z-index numbers — define a small ladder (`--pb-z-dropdown: 100`,
  `--pb-z-modal: 1000`, `--pb-z-toast: 1100`) and use it.
- Animations longer than 250ms for UI feedback.
