---
name: Scientific Clinical Intelligence
colors:
  surface: '#f6f9ff'
  surface-dim: '#cfdbe8'
  surface-bright: '#f6f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#ebf5ff'
  surface-container: '#e3effc'
  surface-container-high: '#ddeaf7'
  surface-container-highest: '#d8e4f1'
  on-surface: '#111d26'
  on-surface-variant: '#3f484d'
  inverse-surface: '#26323b'
  inverse-on-surface: '#e6f2ff'
  outline: '#70787d'
  outline-variant: '#bfc8cd'
  surface-tint: '#0d6682'
  primary: '#00526a'
  on-primary: '#ffffff'
  primary-container: '#176b87'
  on-primary-container: '#b6e7ff'
  inverse-primary: '#8ad0ef'
  secondary: '#5b4dbd'
  on-secondary: '#ffffff'
  secondary-container: '#998cff'
  on-secondary-container: '#2e1990'
  tertiary: '#6d4100'
  on-tertiary: '#ffffff'
  tertiary-container: '#895818'
  on-tertiary-container: '#ffd9b2'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#bde9ff'
  primary-fixed-dim: '#8ad0ef'
  on-primary-fixed: '#001f2a'
  on-primary-fixed-variant: '#004d64'
  secondary-fixed: '#e4dfff'
  secondary-fixed-dim: '#c7bfff'
  on-secondary-fixed: '#170065'
  on-secondary-fixed-variant: '#4333a4'
  tertiary-fixed: '#ffddbb'
  tertiary-fixed-dim: '#faba71'
  on-tertiary-fixed: '#2b1700'
  on-tertiary-fixed-variant: '#673d00'
  background: '#f6f9ff'
  on-background: '#111d26'
  surface-variant: '#d8e4f1'
  canvas-bg: '#F6F8FA'
  surface-white: '#FFFFFF'
  surface-subtle: '#F1F4F6'
  border-divider: '#DCE2E7'
  text-charcoal: '#17212B'
  text-secondary: '#5F6B76'
  text-metadata: '#87929C'
  accent-soft-bg: '#E7F3F6'
  quantum-soft-bg: '#F0EEFC'
  status-success: '#237A57'
  status-warning: '#9A6A16'
  status-error: '#B74343'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '600'
    lineHeight: 44px
    letterSpacing: -0.02em
  headline-xl-mobile:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.01em
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  label-lg:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.02em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1.5rem
  gutter-mobile: 1rem
  margin: 2rem
  margin-mobile: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
  space-2xl: 2rem
  space-3xl: 2.5rem
---

# SIH Hybrid Quantum–Classical Disease Detection
# Dashboard Design System & Product UI Specification

**File:** `SIH_Dashboard_DESIGN.md`  
**Role:** Authoritative visual/UX/design specification for Google Stitch and the Streamlit implementation.  
**Project:** Smart India Hackathon 2026 — Hybrid Quantum Machine Learning Platform for Early Disease Detection.

---

## 4. Visual Language

### 4.1 Direction
Light, neutral, scientific interface.
- off-white application canvas (`#F6F8FA`)
- white analytical surfaces (`#FFFFFF`)
- secondary panels / subtle surface (`#F1F4F6`)
- borders/dividers (`#DCE2E7`)
- charcoal typography (`#17212B`)
- cool gray secondary text (`#5F6B76`)
- metadata text (`#87929C`)
- restrained teal/blue product accent (`#176B87`)
- soft accent backgrounds (`#E7F3F6`)
- subtle indigo/purple reserved for quantum content (`#6558C8`)
- quantum soft backgrounds (`#F0EEFC`)
- muted semantic status: success (`#237A57`), warning (`#9A6A16`), error (`#B74343`)

Absolutely avoid: neon, gradients, glowing borders, glassmorphism, heavy shadows, dark sci-fi HUD styling.

### 4.2 Typography & Spacing
Typography: Modern highly-readable sans-serif (Inter / system font stack).
Spacing: 8-point rhythm (4, 8, 12, 16, 24, 32, 40, 48).
Cards: Subtle borders (#DCE2E7), white surface (#FFFFFF), moderate radius (6-8px), minimal shadow, generous internal padding (20-24px).
