import re
import numpy as np

#===============================================
# Color definitions (ANSI + hex)
#===============================================
RIGHT_ISO_HEX  = "#4E95D9"   # right-orthogonal (▶, ▼)
LEFT_ISO_HEX   = "#CC5900"   # left-orthogonal  (◀, ▲)
NONORTH_HEX    = "#FFC000"   # non-orthogonal core (●)

RIGHT_ISO_COLOR = "\x1b[38;2;78;149;217m"    # 0x4E, 0x95, 0xD9
LEFT_ISO_COLOR  = "\x1b[38;2;204;89;0m"      # 0xCC, 0x59, 0x00
NONORTH_COLOR   = "\x1b[38;2;255;192;0m"     # 0xFF, 0xC0, 0x00
RESET_COLOR     = "\x1b[0m"

RIGHT_ISO_ARROW = RIGHT_ISO_COLOR + "▶" + RESET_COLOR
LEFT_ISO_ARROW  = LEFT_ISO_COLOR  + "◀" + RESET_COLOR
UP_TRIANGLE     = LEFT_ISO_COLOR  + "▲" + RESET_COLOR
DOWN_TRIANGLE   = RIGHT_ISO_COLOR + "▼" + RESET_COLOR
CORE_DOT        = NONORTH_COLOR   + "●" + RESET_COLOR

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)

#===============================================
# ANSI-safe slicing for truncated output
#===============================================
def _ansi_chunks(s: str):
    out = []
    i = 0
    for m in _ANSI_RE.finditer(s):
        a, b = m.span()
        if a > i:
            out.append(("txt", s[i:a]))
        out.append(("ansi", s[a:b]))
        i = b
    if i < len(s):
        out.append(("txt", s[i:]))
    return out

def _ansi_slice_visible(s: str, start: int, width: int, reset: str = RESET_COLOR) -> str:
    if width <= 0:
        return ""

    chunks = _ansi_chunks(s)
    vis = 0
    out = []
    active = ""

    for kind, txt in chunks:
        if kind == "ansi":
            active = txt
            if vis >= start and out is not None:
                out.append(txt)
            continue

        j = 0
        n = len(txt)
        while j < n:
            if vis >= start + width:
                break

            if vis < start:
                skip = min(n - j, start - vis)
                vis += skip
                j += skip
                continue

            take = min(n - j, (start + width) - vis)
            if take > 0:
                if not out and active:
                    out.append(active)
                out.append(txt[j:j + take])
                vis += take
                j += take

        if vis >= start + width:
            break

    if out:
        out.append(reset)
    return "".join(out)

#===============================================
# Utility functions
#===============================================
def print_multi_line(*lines, max_width=None):
    """Print multiple lines, with a maximum width.

    ANSI color escape sequences are treated as zero-width for layout,
    and truncation is ANSI-safe.
    """
    if max_width is None:
        import shutil
        max_width, _ = shutil.get_terminal_size()

    visible = [_strip_ansi(ln) for ln in lines]
    max_line_length = max(len(ln) for ln in visible)

    if max_line_length <= max_width:
        for ln in lines:
            print(ln)
        return

    max_width = max(1, max_width - 10)  # room for ellipses and padding
    n_lines = len(lines)
    n_blocks = (max_line_length - 1) // max_width + 1

    for i in range(n_blocks):
        start = i * max_width

        if i == 0:
            for j, ln in enumerate(lines):
                left = "..." if j == n_lines // 2 else "   "
                right = "..." if j == n_lines // 2 else "   "
                piece = _ansi_slice_visible(ln, start, max_width)
                print(left, piece, right)
            print(("{:^" + str(max_width) + "}").format("..."))
            continue

        if i == n_blocks - 1:
            for ln in lines:
                piece = _ansi_slice_visible(ln, start, max_width)
                print("   ", piece)
            continue

        for j, ln in enumerate(lines):
            left = "..." if j == n_lines // 2 else "   "
            right = "..." if j == n_lines // 2 else "   "
            piece = _ansi_slice_visible(ln, start, max_width)
            print(left, piece, right)

        print(("{:^" + str(max_width) + "}").format("..."))

#===============================================
# MPS / MPO display
#===============================================
def _mps_site_symbol(mps, j):
    N = mps.N
    canform = getattr(mps, "orthoform", getattr(mps, "canform", None))
    pivot = getattr(mps, "pivot_idx", None)

    if pivot is not None:
        if j < pivot:
            return RIGHT_ISO_ARROW
        if j == pivot:
            return CORE_DOT
        return LEFT_ISO_ARROW

    if canform == "Left":
        return CORE_DOT if j == 0 else LEFT_ISO_ARROW
    if canform == "Right":
        return CORE_DOT if j == N - 1 else RIGHT_ISO_ARROW

    return CORE_DOT

def show_mps(mps, max_width=None, show_bonds=True):
    """
    Pretty-print an MPS chain with optional bond-dimension display.
    """
    l1 = ""
    l2 = ""

    for i in range(mps.N - 1):
        bdim = mps.bond_size(i)
        digits = len(str(bdim))

        if bdim == 1:
            bar_char = "┈"
        elif bdim < 100:
            bar_char = "-"
        else:
            bar_char = "━"

        bar = bar_char * max(digits, 1)

        if show_bonds:
            l1 += f"│{bdim}"
        else:
            l1 += "│" + " " * digits

        sym = _mps_site_symbol(mps, i)
        l2 += sym + bar

    if show_bonds:
        l1 += "│"
    else:
        if digits > 1:
            l1 += " " * (digits - 1) + "│"
        else:
            l1 += "│"

    l2 += _mps_site_symbol(mps, mps.N - 1)

    print_multi_line(l1, l2, max_width=max_width)

# Drop-in replacement for show_trp with symbols/colors consistent with show_mps/show_mpo.
def _pad_ansi(s: str, width: int) -> str:
    vis = len(_strip_ansi(s))
    if vis >= width:
        return s
    return s + (" " * (width - vis))

def show_trp(trp, max_sites=8, col_gap="  ", phys_len=2, max_width=None, show_header=True):
    """
    Matrix-style rendering of a TRP (columns are MPS objects).
    """
    cols = getattr(trp, "cols", None)
    if cols is None:
        cols = [trp[j] for j in range(len(trp))]

    if show_header:
        print(repr(trp))

    if len(cols) == 0:
        print_multi_line("⎡⎤", "⎣⎦", max_width=max_width)
        return

    def render_mps_column(mps):
        N = mps.N

        if N <= max_sites:
            head = list(range(N))
            tail = []
            cut = False
        else:
            h = max_sites // 2
            t = max_sites - h
            head = list(range(h))
            tail = list(range(N - t, N))
            cut = True

        lines = []

        def site_line(i):
            sym = _mps_site_symbol(mps, i)
            return sym + ("─" * phys_len)

        def bond_line(i):
            return "│" + str(mps.bond_size(i))

        for i in head:
            lines.append(site_line(i))
            if i < N - 1 and (not cut or i < head[-1]):
                lines.append(bond_line(i))

        if cut:
            lines.append("⋮")
            for j, i in enumerate(tail):
                if j > 0:
                    lines.append(bond_line(i - 1))
                lines.append(site_line(i))

        return lines

    col_lines = [render_mps_column(m) for m in cols]

    H = max(len(cl) for cl in col_lines)
    for cl in col_lines:
        if len(cl) < H:
            pad = H - len(cl)
            pos = cl.index("⋮") if "⋮" in cl else len(cl) // 2
            for _ in range(pad):
                cl.insert(pos, " ")

    widths = [max(len(_strip_ansi(s)) for s in cl) for cl in col_lines]

    def L(r):
        return "⎡" if r == 0 else ("⎣" if r == H - 1 else "⎢")
    def R(r):
        return "⎤" if r == 0 else ("⎦" if r == H - 1 else "⎥")

    rows = []
    for r in range(H):
        inside = col_gap.join(_pad_ansi(cl[r], w) for cl, w in zip(col_lines, widths))
        rows.append(f"{L(r)} {inside} {R(r)}")

    print_multi_line(*rows, max_width=max_width)

def _mpo_site_symbol(mpo, j):
    """
    Symbol for site j based on mpo.orthform and mpo.pivot_idx.
    """
    N = mpo.N
    canform = getattr(mpo, "orthform", getattr(mpo, "canform", "None"))
    pivot = getattr(mpo, "pivot_idx", None)

    if canform == "Right":
        if j == 0:
            return CORE_DOT
        return LEFT_ISO_ARROW

    if canform == "Left":
        if j == N - 1:
            return CORE_DOT
        return RIGHT_ISO_ARROW

    if canform in {"Mixed", "center", "canonical"} and pivot is not None:
        if j < pivot:
            return RIGHT_ISO_ARROW
        if j == pivot:
            return CORE_DOT
        return LEFT_ISO_ARROW

    return CORE_DOT

def show_mpo(mpo, max_width=None):
    """
    Pretty-print an MPO chain showing bond dimensions, canonical structure,
    and the bottom legs under each bond readout.
    """
    l1 = ""
    l2 = ""
    l3 = ""

    for i in range(mpo.N - 1):
        bdim = mpo.bond_size(i)
        strl = len(str(bdim))

        if bdim == 1:
            bar_char = "┈"
        elif bdim < 100:
            bar_char = "-"
        else:
            bar_char = "━"
        bar = bar_char * max(strl, 1)

        l1 += f"│{bdim}"

        sym_i = _mpo_site_symbol(mpo, i)
        l2 += sym_i + bar

        l3 += "│" + " " * strl

    l1 += "│"
    l2 += _mpo_site_symbol(mpo, mpo.N - 1)
    l3 += "│"

    print_multi_line(l1, l2, l3, max_width=max_width)
