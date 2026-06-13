from __future__ import annotations
from typing import Optional, Tuple, Union, List
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import to_rgba  # hex->rgba conversion
try:
    import plotly.io as pio
    pio.renderers.default = "plotly_mimetype"
except Exception:
    pass


class BaseTensorView:
    def __init__(
        self,
        *,
        cmap: str = "viridis",
        normalize: str = "none",  
        min_fraction: float = 0.05,
        min_value: Optional[float] = None,
        background: str = "rgba(0,0,0,0)",
        transparent: bool = True,
        show_colorbar: bool = False,
        title: Optional[str] = None,
        aspect: str = "cube",
        font_color: str = "white",
        color_scale: str = "raw",
        log_eps: float = 1e-12,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap_vmin: float = 0.0,
        cmap_vmax: float = 1.0,
    ):
        self.cmap = cmap
        self.normalize = normalize
        self.min_fraction = float(min_fraction)
        self.min_value = float(min_value) if min_value is not None else None
        self.background = background
        self.transparent_mode = bool(transparent)
        self.show_colorbar = bool(show_colorbar)
        self.title = title
        self.aspect = aspect
        self.font_color = font_color
        self.color_scale = color_scale
        self.log_eps = float(log_eps)
        self.vmin_override = float(vmin) if vmin is not None else None
        self.vmax_override = float(vmax) if vmax is not None else None
        self.cmap_vmin = float(cmap_vmin)
        self.cmap_vmax = float(cmap_vmax)

        self._fig: Optional[go.Figure] = None  # lazy-built Plotly figure

    def figure(self, *, force_rebuild: bool = False) -> go.Figure:
        """
        Build and cache the Plotly figure once. Set force_rebuild=True to regenerate.
        """
        if self._fig is None or force_rebuild:
            flat_traces: List[go.BaseTraceType] = []
            for t in self._build_traces():
                if t is None:
                    continue
                if isinstance(t, (list, tuple)):
                    flat_traces.extend([x for x in t if x is not None])
                else:
                    flat_traces.append(t)

            fig = go.Figure(data=flat_traces)
            fig.update_layout(**self._make_layout())
            self._fig = fig
        return self._fig

    def show(self, *, force_rebuild: bool = False) -> None:
        """
        Display using a pure-HTML inline embed (no iframe), bundling Plotly JS.
        """
        fig = self.figure(force_rebuild=force_rebuild)
        from IPython.display import HTML, display  # harmless import if not in notebook
        html = fig.to_html(full_html=False, include_plotlyjs="inline")
        display(HTML(html))

    def show_html(self, *, force_rebuild: bool = False) -> None:
        fig = self.figure(force_rebuild=force_rebuild)
        from IPython.display import HTML, display
        display(HTML(fig.to_html(full_html=False, include_plotlyjs="inline")))

    def __call__(self) -> None:
        self.show()

    def __getattr__(self, name):
        fig = self.figure()
        return getattr(fig, name)

    def to_html(
        self,
        *,
        include_plotlyjs: Union[bool, str] = "cdn",
        full_html: bool = True,
    ) -> str:
        """
        Get a self-contained HTML string, useful for saving or embedding.
        """
        fig = self.figure()
        return fig.to_html(full_html=full_html, include_plotlyjs=include_plotlyjs)

    # ------------- Layout & style scaffolding -------------

    def _make_layout(self) -> dict:
        scene_axes = self._scene_axes()
        layout = dict(
            scene=dict(
                **scene_axes,
                aspectmode=self.aspect,
                bgcolor=self.background,
            ),
            paper_bgcolor=self.background,
            plot_bgcolor=self.background,
            margin=dict(l=0, r=120, t=40 if self.title else 0, b=0),
            title=dict(
                text=self.title or "",
                font=dict(color=self.font_color),
            ),
        )
        if self.transparent_mode:
            layout["paper_bgcolor"] = "rgba(0,0,0,0)"
            layout["plot_bgcolor"] = "rgba(0,0,0,0)"
            layout["scene"]["bgcolor"] = "rgba(0,0,0,0)"
        return layout

    # ------------- Color & scale utilities -------------

    def _rgba_string(self, color: str, alpha: float) -> str:
        r, g, b, _ = to_rgba(color, alpha=alpha)
        return f"rgba({int(255*r)},{int(255*g)},{int(255*b)},{alpha:.3f})"

    def _color_scale(self, c: np.ndarray) -> tuple[np.ndarray, float, float, dict]:
        """
        Returns (values_for_colormap, vmin, vmax, colorbar_kwargs)
        """
        if self.color_scale == "log":
            c = np.maximum(c, self.log_eps)
            log_c = np.log10(c)
            vmin = self.vmin_override if self.vmin_override is not None else float(log_c.min())
            vmax = self.vmax_override if self.vmax_override is not None else float(log_c.max())
            tick_min, tick_max = int(np.floor(vmin)), int(np.ceil(vmax))
            tickvals = np.arange(tick_min, tick_max + 1)
            ticktext = [f"1e{p}" for p in tickvals]
            return log_c, vmin, vmax, dict(
                tickvals=tickvals,
                ticktext=ticktext,
                tickfont=dict(color=self.font_color),
                title=dict(font=dict(color=self.font_color)),
                tickmode="array"
            )
        elif self.color_scale == "raw":
            vmin = self.vmin_override if self.vmin_override is not None else float(c.min())
            vmax = self.vmax_override if self.vmax_override is not None else float(c.max())
            tickvals = np.linspace(vmin, vmax, 5) if vmin < vmax else np.array([vmin])
            ticktext = [f"{t:.3g}" for t in tickvals]
            return c, vmin, vmax, dict(
                tickvals=tickvals, ticktext=ticktext, tickmode="array",
                tickfont=dict(color=self.font_color),
                title=dict(font=dict(color=self.font_color)),
                len=0.9, thickness=16, outlinewidth=0,
            )
        else:  # linear
            vmin = self.vmin_override if self.vmin_override is not None else float(c.min())
            vmax = self.vmax_override if self.vmax_override is not None else float(c.max())
            tickvals = np.linspace(vmin, vmax, 5) if vmin < vmax else np.array([vmin])
            ticktext = [f"{t:.3g}" for t in tickvals]
            return c, vmin, vmax, dict(
                tickvals=tickvals, ticktext=ticktext, tickmode="array",
                tickfont=dict(color=self.font_color),
                title=dict(font=dict(color=self.font_color)),
                len=0.9, thickness=16, outlinewidth=0,
            )

    # ------------- Abstract hooks for children -------------

    def _build_traces(self) -> List[go.BaseTraceType | None]:
        raise NotImplementedError
        

    def _scene_axes(self) -> dict:
        raise NotImplementedError


# ============================================================
# ThreeTensorView: voxel-specific child using BaseTensorView
# ============================================================
class ThreeTensorView(BaseTensorView):
    def __init__(
        self,
        A: np.ndarray,
        *,
        cmap: str = "viridis",
        normalize: str = "none",
        min_fraction: float = 0.05,
        min_value: Optional[float] = None,
        background: str = "rgba(0,0,0,0)",
        transparent: bool = True,
        show_colorbar: bool = False,
        title: Optional[str] = None,
        aspect: str = "cube",           # keep cube aspect mode
        font_color: str = "white",
        color_scale: str = "linear",
        log_eps: float = 1e-12,
        max_voxels: Optional[int] = None,
        voxel_opacity: float = 1.0,
        voxel_depth: float = 0.5,
        threshold: bool = False,
        grid: bool = True,
        grid_opacity: float = 0.25,
        grid_color: str = "lightgray",
        # NEW:
        bbox_opacity: Optional[float] = None,
        bbox_color: Optional[str] = None,
        grid_stride: Tuple[int, int, int] = (1, 1, 1),
        integer_ticks: bool = True,
        show_ticks: bool = True,
        bounding_box_only: bool = False,
        use_magnitude_opacity: bool = False,
        opacity_gamma: float = 1.0,
        solid_color: Optional[str] = None,
        signed_values: bool = False,
        support_negative_data: Optional[bool] = None,
        cmap_vmin: float = 0.0,
        cmap_vmax: float = 1.0,
    ):
        if not isinstance(A, np.ndarray) or A.ndim != 3:
            raise ValueError("A must be a real 3D numpy array of shape (I, J, K).")
        if not np.isrealobj(A):
            raise ValueError("A must be real-valued.")

        super().__init__(
            cmap=cmap,
            normalize=normalize,
            min_fraction=min_fraction,
            min_value=min_value,
            background=background,
            transparent=transparent,
            show_colorbar=show_colorbar,
            title=title,
            aspect=aspect,
            font_color=font_color,
            color_scale=color_scale,
            log_eps=log_eps,
            cmap_vmin=cmap_vmin,
            cmap_vmax=cmap_vmax,
        )

        self.A = np.asarray(A, dtype=float)
        self.I, self.J, self.K = self.A.shape

        self.solid_color = solid_color
        self.max_voxels = int(max_voxels) if max_voxels is not None else None
        self.voxel_opacity = float(voxel_opacity)
        self.voxel_depth = float(voxel_depth)
        self.threshold = bool(threshold)
        self.grid = bool(grid)
        self.grid_opacity = float(grid_opacity)
        self.grid_color = grid_color
        self.grid_stride = tuple(int(s) for s in grid_stride)

        # NEW: if not given, fall back to grid settings for backwards compatibility
        self.bbox_opacity = float(bbox_opacity) if bbox_opacity is not None else self.grid_opacity
        self.bbox_color = bbox_color or self.grid_color
        self.integer_ticks = bool(integer_ticks)
        self.show_ticks = bool(show_ticks)
        self.bounding_box_only = bool(bounding_box_only)
        self.use_magnitude_opacity = bool(use_magnitude_opacity)
        self.use_lighting = not self.use_magnitude_opacity
        self.opacity_gamma = float(opacity_gamma)
        if support_negative_data is not None:
            signed_values = bool(support_negative_data)
        self.signed_values = bool(signed_values)

    def _build_traces(self) -> List[go.BaseTraceType | None]:
        traces: List[go.BaseTraceType | None] = []

        vox = self._make_voxel_mesh()
        if vox is not None:
            if isinstance(vox, (list, tuple)):
                traces.extend([t for t in vox if t is not None])
            else:
                traces.append(vox)

        if self.bounding_box_only:
            traces.append(self._bounding_box_lines())
        else:
            if self.grid:
                traces.append(self._grid_lines())
            traces.append(self._bounding_box_lines())

        return traces

    def _scene_axes(self) -> dict:
        # Use a common extent so that one unit on x,y,z is visually identical.
        L = max(self.I, self.J, self.K)
        return dict(
            xaxis=self._axis_style(self.I, L, "i"),
            yaxis=self._axis_style(self.J, L, "j"),
            zaxis=self._axis_style(self.K, L, "k"),
        )

    def _axis_style(self, length: int, extent: int, axis_label: str) -> dict:
        # Pad is based on the common extent, so the scene box stays cubic.
        pad = max(0.5, 0.05 * float(extent))
        cfg = dict(
            showbackground=False,
            showgrid=False,
            zeroline=False,
            title=dict(text=axis_label, font=dict(color=self.font_color)),
            tickfont=dict(color=self.font_color),
            tickmode="array",
            tickvals=list(range(1, length + 1)) if self.integer_ticks else [],
            ticktext=[str(t) for t in range(1, length + 1)] if self.integer_ticks else [],
            range=[-pad, float(extent) + pad],   # <- equalize ranges across axes
        )
        if not self.show_ticks:
            cfg["showticklabels"] = False
            cfg["ticks"] = ""  # also remove the tick marks themselves
        return cfg

    def _normalized_values(self) -> np.ndarray:
        A = self.A
        if self.normalize == "absmax":
            denom = float(np.max(np.abs(A)))
            if denom <= 0:
                return np.zeros_like(A)
            if self.signed_values:
                return A / denom
            return np.abs(A) / denom
        elif self.normalize == "none":
            if self.signed_values:
                return A
            return np.abs(A)
        else:
            raise ValueError("normalize must be 'absmax' or 'none'.")

    def _apply_threshold(self, vals: np.ndarray) -> np.ndarray:
        # Default behavior when thresholding is enabled
        # do not draw exact zeros
        abs_vals = np.abs(vals)
        if self.threshold:
            mask = abs_vals > 0
        else:
            mask = np.ones_like(vals, dtype=bool)

        if self.threshold:
            if self.min_value is None:
                thr = float(self.min_fraction)
            else:
                if self.normalize == "none":
                    thr = float(self.min_value)
                else:
                    denom = float(np.max(np.abs(self.A)))
                    thr = float(self.min_value) / (denom if denom > 0 else 1.0)

            if thr > 0:
                mask &= abs_vals >= thr

        return mask

    def _voxel_template(self) -> tuple[np.ndarray, np.ndarray]:
        if not (0 < self.voxel_depth <= 1):
            raise ValueError("voxel_depth must be in (0,1].")
        shrink = (1.0 - self.voxel_depth) / 2.0
        verts = np.array([
            [shrink,      shrink,      shrink],
            [1-shrink,    shrink,      shrink],
            [1-shrink,    1-shrink,    shrink],
            [shrink,      1-shrink,    shrink],
            [shrink,      shrink,      1-shrink],
            [1-shrink,    shrink,      1-shrink],
            [1-shrink,    1-shrink,    1-shrink],
            [shrink,      1-shrink,    1-shrink],
        ], float)
        tris = np.array([
            [0,1,2],[0,2,3],[4,6,5],[4,7,6],
            [0,5,1],[0,4,5],[3,2,6],[3,6,7],
            [0,3,7],[0,7,4],[1,5,6],[1,6,2]
        ], int)
        return verts, tris

    def _mesh_lighting(self) -> dict:
        if not self.use_lighting:
            return dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0)
        # White/light backgrounds need softer directional shading, otherwise
        # face contrast looks overly dark and muddy.
        bg = str(getattr(self, "background", "")).strip().lower()
        light_bg = bg in {
            "white",
            "#fff",
            "#ffffff",
            "rgb(255,255,255)",
            "rgba(255,255,255,1)",
            "rgba(255, 255, 255, 1)",
        }
        if light_bg and not self.transparent_mode:
            # On white backgrounds, disable directional shading so colors stay faithful.
            return dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0)
        # Keep voxels close to full brightness with only a soft face cue.
        return dict(ambient=0.92, diffuse=0.18, specular=0.02, roughness=1.0)

    def _light_position(self) -> dict:
        if not self.use_lighting:
            return dict(x=0, y=0, z=0)
        return dict(x=0, y=0, z=8)

    def _make_voxel_mesh(self) -> Optional[List[go.Mesh3d]]:
        vals = self._normalized_values()
        mask = self._apply_threshold(vals)
    
        if self.max_voxels is not None:
            nz = np.transpose(np.nonzero(mask))
            if nz.shape[0] > self.max_voxels:
                mags = vals[mask]
                idx = np.argsort(mags)[-self.max_voxels:]
                sel = nz[idx]
                ii, jj, kk = sel[:, 0], sel[:, 1], sel[:, 2]
                magnitudes = vals[ii, jj, kk]
            else:
                ii, jj, kk = np.nonzero(mask)
                magnitudes = vals[mask]
        else:
            ii, jj, kk = np.nonzero(mask)
            magnitudes = vals[mask]
    
        if ii.size == 0:
            return None
    
        VERTS, TRIS = self._voxel_template()
        built = self._build_mesh(
            ii, jj, kk, magnitudes, VERTS, TRIS,
            opacity=self.voxel_opacity,
            show_colorbar=self.show_colorbar  # ← respect the instance flag
        )
        return built if isinstance(built, list) else [built]

    def _build_mesh(
        self, ii, jj, kk, magnitudes, VERTS, TRIS, opacity, show_colorbar=None
    ):
        """
        Build Mesh3d for either solid-color fast path or colormap-based path.
    
        Parameters
        ----------
        ii, jj, kk : 1D integer arrays
            Selected voxel indices along each axis.
        magnitudes : 1D float array
            Per-voxel magnitudes (already normalized if desired).
        VERTS : (8,3) float array
            Unit-cube vertex template (possibly shrunken by voxel_depth).
        TRIS : (12,3) int array
            Triangle indices on the unit cube.
        opacity : float
            Global opacity multiplier for the mesh.
        show_colorbar : Optional[bool]
            If None, use self.show_colorbar. If True/False, override for this call.
    
        Returns
        -------
        go.Mesh3d or [go.Mesh3d, go.Mesh3d]
            The mesh, and optionally a dummy Mesh3d to host the colorbar.
        """
        nvox = ii.size
        x, y, z, c = (np.empty(8 * nvox) for _ in range(4))
        I_idx = np.empty(12 * nvox, dtype=int)
        J_idx = np.empty(12 * nvox, dtype=int)
        K_idx = np.empty(12 * nvox, dtype=int)
    
        for t, (i, j, k, m) in enumerate(zip(ii, jj, kk, magnitudes)):
            base_v, base_f = 8 * t, 12 * t
            verts = VERTS + [i, j, k]
            x[base_v : base_v + 8], y[base_v : base_v + 8], z[base_v : base_v + 8] = verts.T
            c[base_v : base_v + 8] = m
            tris = TRIS + base_v
            I_idx[base_f : base_f + 12], J_idx[base_f : base_f + 12], K_idx[base_f : base_f + 12] = tris.T
    
        # -------- solid-color fast path: never show a colorbar --------
        if self.solid_color is not None:
            rgba = self._rgba_string(self.solid_color, opacity)
            vertexcolor = [rgba] * (8 * nvox)
            mesh = go.Mesh3d(
                x=x, y=y, z=z,
                i=I_idx, j=J_idx, k=K_idx,
                vertexcolor=vertexcolor,
                flatshading=not self.use_magnitude_opacity,
                name="voxels",
                showscale=False,
                lighting=self._mesh_lighting(),
                lightposition=self._light_position(),
                opacity=opacity,
            )
            return mesh
    
        # -------- colormap path --------
        intensity, vmin, vmax, colorbar = self._color_scale(c)
        
        # map [vmin, vmax] -> [0,1]
        denom = (vmax - vmin) if vmax > vmin else 1.0
        normed = np.clip((intensity - vmin) / denom, 0.0, 1.0)

        # remap into [cmap_vmin, cmap_vmax] to truncate palette ends
        cmap_range = self.cmap_vmax - self.cmap_vmin
        palette_normed = self.cmap_vmin + normed * cmap_range

        # Try Plotly colorscales first, then fall back to Matplotlib
        try:
            import plotly.colors as pc
            # Check if it's a valid Plotly colorscale
            if self.cmap.lower() in [cs.lower() for cs in pc.named_colorscales()]:
                # Get Plotly colorscale
                colorscale = pc.get_colorscale(self.cmap)
                # Sample the colorscale at palette_normed positions
                rgb = np.array([pc.sample_colorscale(colorscale, val)[0] for val in palette_normed])
                # Convert from 'rgb(r,g,b)' string format to [r/255, g/255, b/255]
                rgb = np.array([[float(x)/255 for x in color.strip('rgb()').split(',')]
                                for color in rgb])
            else:
                raise ValueError("Not a Plotly colorscale")
        except:
            # Fall back to Matplotlib
            try:
                from matplotlib import colormaps as _mcm
                cmap = _mcm.get_cmap(self.cmap)
            except Exception:
                import matplotlib.cm as _cm
                cmap = _cm.get_cmap(self.cmap)
            rgb = cmap(palette_normed)[:, :3]
    
        if self.use_magnitude_opacity:
            vals = np.maximum(c, self.log_eps if self.color_scale == "log" else 0.0)
            fvals = np.log(vals) if self.color_scale == "log" else vals
            fmax = fvals.max() if fvals.max() > 0 else 1.0
            alpha_min = 0.05
            scaled = alpha_min + (1 - alpha_min) * ((fvals / fmax) ** self.opacity_gamma)
            opacities = scaled * opacity
            vertexcolor = [
                f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.3f})"
                for (r, g, b), a in zip(rgb, opacities)
            ]
        else:
            vertexcolor = [
                f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{opacity:.3f})"
                for (r, g, b) in rgb
            ]
    
        mesh = go.Mesh3d(
            x=x, y=y, z=z,
            i=I_idx, j=J_idx, k=K_idx,
            vertexcolor=vertexcolor,
            flatshading=not self.use_magnitude_opacity,
            name="voxels",
            showscale=False,  # colorbar hosted on dummy trace when requested
            lighting=self._mesh_lighting(),
            lightposition=self._light_position(),
            opacity=opacity,
        )
        
    
        # decide whether to show the colorbar (default to instance flag if None)
        want_bar = self.show_colorbar if show_colorbar is None else bool(show_colorbar)
        if want_bar:
            colorbar_trace = go.Mesh3d(
                x=[None], y=[None], z=[None],
                i=[], j=[], k=[],
                intensity=[vmin, vmax],
                colorscale=self.cmap,
                cmin=vmin, cmax=vmax,
                showscale=True,
                colorbar=colorbar,
                opacity=0,
            )
            return [mesh, colorbar_trace]
    
        return mesh


    # --------- wireframe helpers ---------

    def _bounding_box_lines(self) -> go.Scatter3d:
        I, J, K = self.I, self.J, self.K
        xs, ys, zs = [], [], []
    
        def seg(p0, p1):
            xs.extend([p0[0], p1[0], None])
            ys.extend([p0[1], p1[1], None])
            zs.extend([p0[2], p1[2], None])
    
        corners = [
            (0,0,0),(I,0,0),(I,J,0),(0,J,0),
            (0,0,K),(I,0,K),(I,J,K),(0,J,K)
        ]
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]
        for e in edges:
            seg(corners[e[0]], corners[e[1]])
    
        return go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=3, color=self.bbox_color),   # NEW
            opacity=self.bbox_opacity,                   # NEW
            hoverinfo="skip",
            name="",
            showlegend=False,
        )

    def _grid_lines(self) -> go.Scatter3d:
        I, J, K = self.I, self.J, self.K
        si, sj, sk = self.grid_stride
        xs, ys, zs = [], [], []

        def seg(p0, p1):
            xs.extend([p0[0], p1[0], None])
            ys.extend([p0[1], p1[1], None])
            zs.extend([p0[2], p1[2], None])

        for j in range(0, J+1, max(1, sj)):
            for k in range(0, K+1, max(1, sk)):
                seg((0, j, k), (I, j, k))
        for i in range(0, I+1, max(1, si)):
            for k in range(0, K+1, max(1, sk)):
                seg((i, 0, k), (i, J, k))
        for i in range(0, I+1, max(1, si)):
            for j in range(0, J+1, max(1, sj)):
                seg((i, j, 0), (i, j, K))

        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(width=1, color=self.grid_color),
            opacity=self.grid_opacity, name="grid", hoverinfo="skip",
            showlegend=False,
        )
        
class FourTensorView(BaseTensorView):
    def __init__(
        self,
        A: np.ndarray,
        *,
        gap_x: float = 1.5,
        pad_frac: float = 0.12,
        mini_grid: bool = True,
        mini_grid_stride: Tuple[int, int, int] = (1, 1, 1),
        mini_grid_color: str = "lightgray",
        mini_grid_opacity: float = 0.25,
        mini_grid_mode: str = "full",   # "full" or "bbox"
        share_colorbar: bool = True,
        show_colorbar: bool = False,

        camera_eye: Optional[Tuple[float, float, float]] = None,
        zoom: float = 0.75,
        lock_about_x: bool = True,
        cmap: str = "viridis",
        color_scale: str = "raw",    
        global_normalize: bool = False,
        global_max_override: Optional[float] = None,
        log_eps: float = 1e-16,
        log_floor_abs: Optional[float] = None,  
        voxel_opacity: float = 1.0,
        voxel_depth: float = 0.5,
        grid: bool = False,             
        background: str = "rgba(0,0,0,0)",
        transparent: bool = True,
        font_color: str = "white",
        index_ticks: bool = True,
        threshold: Optional[bool] = None,
        min_value: Optional[float] = None,
        min_fraction: Optional[float] = None,
        title: Optional[str] = None,
        font_size: int = 28,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap_vmin: float = 0.0,
        cmap_vmax: float = 1.0,

        **kwargs,
    ):
        if not isinstance(A, np.ndarray) or A.ndim != 4:
            raise ValueError("A must be a real 4D numpy array of shape (I, J, K, L).")
        if not np.isrealobj(A):
            raise ValueError("A must be real-valued.")
        if color_scale not in {"linear", "log", "raw"}:
            raise ValueError("color_scale must be 'linear', 'log', or 'raw'.")

        self.A = np.asarray(A, dtype=float)
        self.I, self.J, self.K, self.L = self.A.shape
        self.gap_x = float(gap_x)
        self.pad_frac = float(pad_frac)
        self.mini_grid = bool(mini_grid)
        self.mini_grid_stride = tuple(int(s) for s in mini_grid_stride)
        self.mini_grid_color = mini_grid_color
        self.mini_grid_opacity = float(mini_grid_opacity)
        self.mini_grid_mode = str(mini_grid_mode)
        self.share_colorbar = bool(share_colorbar)
        self.camera_eye = camera_eye
        self.zoom = float(zoom)
        self.lock_about_x = bool(lock_about_x)
        self.cmap = cmap
        self.color_scale = color_scale
        self.global_normalize = bool(global_normalize)
        self._gmax_override = (
            None if global_max_override is None
            else float(global_max_override) if global_max_override > 0 else None
        )
        self.log_eps = float(log_eps)
        self.log_floor_abs = None if log_floor_abs is None else float(log_floor_abs)
        self.voxel_opacity = float(voxel_opacity)
        self.voxel_depth = float(voxel_depth)
        self.slice_grid = bool(grid)  # generally False here
        self.index_ticks = bool(index_ticks)
        user_set_thresh = (threshold is not None) or (min_value is not None) or (min_fraction is not None)
        self.threshold = bool(threshold) if threshold is not None else True
        self.min_value = min_value
        self.min_fraction = None if (min_fraction is None) else float(min_fraction)
        if not user_set_thresh:
            self.min_value = 0.0
            self.min_fraction = 0.0
        self.title = title
        self.font_size = int(font_size)
        self.show_colorbar = bool(show_colorbar)  

        self.kwargs = dict(kwargs)
        self._absA = np.abs(self.A)
        self._gmax = float(self._absA.max()) if self.global_normalize else None

        super().__init__(
            cmap=cmap,
            normalize="none",
            min_fraction=0.0,
            min_value=None,
            background=background,
            transparent=transparent,
            show_colorbar=False,
            title=title,
            aspect="manual",
            font_color=font_color,
            color_scale="linear",
            log_eps=log_eps,
            vmin=vmin,
            vmax=vmax,
            cmap_vmin=cmap_vmin,
            cmap_vmax=cmap_vmax,
        )

    # ---------------- Base hooks ----------------

    def _build_traces(self) -> List[go.BaseTraceType]:
        traces: List[go.BaseTraceType] = []
    
        total_width = self._total_width()
        # Build all slices
        for ell in range(self.L):
            slice_abs = np.abs(self.A[:, :, :, ell])
            
            # Only normalize if requested
            if self.global_normalize:
                denom = self._slice_denominator(ell)
            else:
                denom = 1.0  # No normalization
            
            if self.color_scale == "log":
                base = np.clip(slice_abs / denom, self.log_eps, None)
                Lmin = float(np.log10(self.log_eps))
                A_slice = (np.log10(base) - Lmin) / (0.0 - Lmin)
                if (self.log_floor_abs is not None) and (self.log_floor_abs > 0):
                    floor_norm = self.log_floor_abs / denom
                    pos = A_slice > 0
                    A_slice[pos] = np.maximum(A_slice[pos], floor_norm)
                tv_color_scale = "raw"
            else:
                A_slice = slice_abs / denom
                tv_color_scale = self.color_scale


            tv_kwargs = dict(self.kwargs)
            tv_kwargs.update(
                dict(
                    cmap=self.cmap,
                    normalize="none",            # already normalized
                    color_scale=tv_color_scale,
                    threshold=self.threshold,
                    grid=False,                  # grid handled here (mini-grid)
                    voxel_opacity=self.voxel_opacity,
                    voxel_depth=self.voxel_depth,
                    background=self.background,
                    transparent=self.transparent_mode,  # from BaseTensorView
                    font_color=self.font_color,
                    integer_ticks=False,
                    show_ticks=False,
                    show_colorbar=self.show_colorbar and ((ell == 0) if self.share_colorbar else True),

                )
            )
            if self.min_value is not None:
                tv_kwargs["min_value"] = float(self.min_value) / denom
                tv_kwargs.pop("min_fraction", None)
            elif self.min_fraction is not None:
                tv_kwargs["min_fraction"] = float(self.min_fraction)
            tv_kwargs.pop("left_to_right", None)
            tv = ThreeTensorView(A_slice, **tv_kwargs)
            slice_fig = tv.figure()

            x_off = ell * (self.I + self.gap_x)
            keep_cb = (ell == 0) or (not self.share_colorbar)
            shifted = self._shift_traces_x_and_filter_colorbar(slice_fig.data, x_off, keep_cb)
            traces.extend(shifted)

            if self.mini_grid:
                traces.append(self._mini_grid_lines(x_off=x_off))

        return traces

    def _scene_axes(self) -> dict:
        total_width = self._total_width()
        pad_x = max(0.5, self.pad_frac * total_width)
        pad_y = max(0.5, self.pad_frac * self.J)
        pad_z = max(0.5, self.pad_frac * self.K)

        # x tick labels at slice centers
        block_spacing = self.I + self.gap_x
        x_centers = [ell * block_spacing + 0.5 * self.I for ell in range(self.L)]
        x_labels = [str(i) for i in range(1, self.L + 1)]

        # aspect ratio based on total width
        ar = dict(x=total_width, y=self.J, z=self.K)
        m = max(ar.values())
        ar = {k: v / m for k, v in ar.items()}

        return dict(
            xaxis=dict(
                showbackground=False, showgrid=False, zeroline=False,
                showticklabels=self.index_ticks,
                tickmode="array", tickvals=x_centers, ticktext=x_labels,
                tickfont=dict(size=self.font_size, color=self.font_color),
                range=[-pad_x, total_width + pad_x],
                title=dict(text=""),
            ),
            yaxis=dict(
                showbackground=False, showgrid=False, zeroline=False,
                showticklabels=False, range=[-pad_y, self.J + pad_y],
                title=dict(text="")
            ),
            zaxis=dict(
                showbackground=False, showgrid=False, zeroline=False,
                showticklabels=False, range=[-pad_z, self.K + pad_z],
                title=dict(text="")
            ),
            aspectratio=ar,
        )

    # ---------------- public API override ----------------

    def figure(self, *, force_rebuild: bool = False) -> go.Figure:
        fig = super().figure(force_rebuild=force_rebuild)

        # Apply camera and drag mode once (idempotent updates are fine)
        if self.camera_eye is not None:
            eye = tuple(map(float, self.camera_eye))
            up_vec = dict(x=1, y=0, z=0) if self.lock_about_x else dict(x=0, y=0, z=1)
            drag = "turntable" if self.lock_about_x else "orbit"
        else:
            base = float(max(self.I, self.J, self.K))
            wide = float(self._total_width()) / max(1.0, base)
            s = self.zoom * (1.6 + 1.2 * (wide ** 0.6))
            eye = (0.35 * s, 1.8 * s, 1.2 * s)
            up_vec = dict(x=1, y=0, z=0) if self.lock_about_x else dict(x=0, y=0, z=1)
            drag = "turntable" if self.lock_about_x else "orbit"

        fig.update_layout(
            scene_camera=dict(
                eye=dict(x=eye[0], y=eye[1], z=eye[2]),
                up=up_vec,
                projection=dict(type="perspective"),
            ),
            dragmode=drag,
            font=dict(size=self.font_size, color=self.font_color),
            title=dict(text=self.title or "", font=dict(size=self.font_size, color=self.font_color)),
        )
        return fig

    # ---------------- internal helpers ----------------

    def _total_width(self) -> float:
        return self.L * self.I + (self.L - 1) * self.gap_x

    def _slice_denominator(self, ell: int) -> float:
        if self.global_normalize:
            if self._gmax_override is not None:
                return self._gmax_override
            g = self._gmax if (self._gmax and self._gmax > 0) else 1.0
            return g
        smax = float(np.abs(self.A[:, :, :, ell]).max())
        return smax if smax > 0 else 1.0

    def _mini_grid_lines(self, x_off: float) -> go.Scatter3d:
        I, J, K = self.I, self.J, self.K
        xs, ys, zs = [], [], []

        def seg(p0, p1):
            xs.extend([p0[0] + x_off, p1[0] + x_off, None])
            ys.extend([p0[1], p1[1], None])
            zs.extend([p0[2], p1[2], None])

        if self.mini_grid_mode == "bbox":
            corners = [
                (0, 0, 0), (I, 0, 0), (I, J, 0), (0, J, 0),
                (0, 0, K), (I, 0, K), (I, J, K), (0, J, K),
            ]
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            for i, j in edges:
                seg(corners[i], corners[j])
        else:
            si, sj, sk = self.mini_grid_stride
            for j in range(0, J + 1, max(1, sj)):
                for k in range(0, K + 1, max(1, sk)):
                    seg((0, j, k), (I, j, k))
            for i in range(0, I + 1, max(1, si)):
                for k in range(0, K + 1, max(1, sk)):
                    seg((i, 0, k), (i, J, k))
            for i in range(0, I + 1, max(1, si)):
                for j in range(0, J + 1, max(1, sj)):
                    seg((i, j, 0), (i, j, K))

        return go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=2, color=self.mini_grid_color),
            opacity=self.mini_grid_opacity,
            hoverinfo="skip",
            name="mini-grid",
            showlegend=False,
        )

    def _log_tickinfo(self) -> Dict[str, Any]:
        Lmin = float(np.log10(self.log_eps))  # negative
        lo = int(np.ceil(Lmin))
        hi = 0
        exps = list(range(lo, hi + 1))
        # map exponent positions in log-domain to t ∈ [0,1]
        tvals = [(e - Lmin) / (0.0 - Lmin) for e in exps]
        return dict(tickvals=tvals, ticktext=[f"1e{e}" for e in exps])

    def _shift_traces_x_and_filter_colorbar(self, traces, x_off: float, keep_colorbar: bool):
        out = []
        for tr in traces:
            d = tr.to_plotly_json()
            # shift any numeric x entries by x_off
            if "x" in d:
                x = d["x"]
                if isinstance(x, (list, tuple)):
                    d["x"] = [(xx + x_off) if isinstance(xx, (int, float)) else None for xx in x]
                else:
                    d["x"] = (np.asarray(x, dtype=float) + x_off).tolist()

            is_mesh = (d.get("type", "").lower() == "mesh3d")
            is_dummy_cb = is_mesh and (len(d.get("i", [])) == 0)

            if is_dummy_cb:
                if not keep_colorbar:
                    continue
                cb = d.get("colorbar", {}) or {}
                cb.update(dict(
                    thickness=40,  # you can override by editing here or add fields to __init__ if desired
                    tickfont=dict(size=self.font_size, color=self.font_color),
                    title=dict(text=cb.get("title", {}).get("text", ""),
                               font=dict(size=self.font_size, color=self.font_color)),
                    x=0.88, xanchor="left", y=0.5, yanchor="middle", len=0.9,
                ))
                # If log mode, force range to [0,1] with log ticks at correct positions.
                if self.color_scale == "log":
                    d["cauto"] = False
                    d["cmin"] = 0.0
                    d["cmax"] = 1.0
                    cb.update(self._log_tickinfo())
                d["colorbar"] = cb
                d["showscale"] = True
                out.append(go.Mesh3d(**d))
            else:
                out.append(go.Mesh3d(**d) if is_mesh else go.Scatter3d(**d))
        return out

class MPSView(BaseTensorView):
    def __init__(
        self,
        mps_or_cores: Union[object, Sequence[np.ndarray]],
        *,
        gap_x: float = 1.5,
        pad_frac: float = 0.12,
        mini_grid: bool = True,
        mini_grid_stride: Tuple[int, int, int] = (1, 1, 1),
        mini_grid_color: str = "lightgray",
        mini_grid_opacity: float = 0.25,
        mini_grid_mode: str = "bbox",
        camera_eye: Optional[Tuple[float, float, float]] = None,
        zoom: float = 0.75,
        lock_about_x: bool = True,
        left_to_right: bool = True,
        cmap: str = "viridis",
        color_scale: str = "linear",
        global_normalize: bool = True,
        global_max_override: Optional[float] = None,
        log_eps: float = 1e-12,
        log_floor_abs: Optional[float] = None,
        voxel_opacity: float = 1.0,
        voxel_depth: float = 0.6,
        background: str = "rgba(0,0,0,0)",
        transparent: bool = True,
        font_color: str = "black",
        threshold: Optional[bool] = None,
        min_value: Optional[float] = None,
        min_fraction: Optional[float] = None,
        share_colorbar: bool = True,
        show_colorbar: bool = False,
        title: Optional[str] = None,
        font_size: int = 24,
        # NEW: expose a scene/trace-wide tick toggle
        show_ticks: bool = True,
        signed_values: bool = False,
        support_negative_data: Optional[bool] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap_vmin: float = 0.0,
        cmap_vmax: float = 1.0,
        **kwargs,
    ):
        cores = self._coerce_to_cores(mps_or_cores)
        if color_scale not in {"linear", "log", "raw"}:
            raise ValueError("color_scale must be 'linear', 'log', or 'raw'.")
        self.gap_x = float(gap_x)
        self.pad_frac = float(pad_frac)
        self.mini_grid = bool(mini_grid)
        self.mini_grid_stride = tuple(int(s) for s in mini_grid_stride)
        self.mini_grid_color = mini_grid_color
        self.mini_grid_opacity = float(mini_grid_opacity)
        self.mini_grid_mode = str(mini_grid_mode)
        self.camera_eye = camera_eye
        self.zoom = float(zoom)
        self.lock_about_x = bool(lock_about_x)
        self.left_to_right = bool(left_to_right)
        self.cmap = cmap
        self.color_scale = color_scale
        self.global_normalize = bool(global_normalize)
        self._gmax_override = None if global_max_override is None else float(global_max_override) if global_max_override > 0 else None
        self.log_eps = float(log_eps)
        self.log_floor_abs = None if log_floor_abs is None else float(log_floor_abs)
        self.voxel_opacity = float(voxel_opacity)
        self.voxel_depth = float(voxel_depth)
        self.share_colorbar = bool(share_colorbar)
        self.show_colorbar = bool(show_colorbar)
        self.title = title
        self.font_size = int(font_size)
        self.kwargs = dict(kwargs)
        # NEW: store tick toggle
        self.show_ticks = bool(show_ticks)
        if support_negative_data is not None:
            signed_values = bool(support_negative_data)
        self.signed_values = bool(signed_values)

        user_set_thresh = (threshold is not None) or (min_value is not None) or (min_fraction is not None)
        self.threshold = bool(threshold) if threshold is not None else True
        self.min_value = min_value
        self.min_fraction = None if (min_fraction is None) else float(min_fraction)
        if not user_set_thresh:
            self.min_value = 0.0
            self.min_fraction = 0.0

        self.blocks: List[np.ndarray] = []
        self.shapes: List[Tuple[int, int, int]] = []
        n = len(cores)
        if n < 2:
            raise ValueError("Need at least left and right boundary cores.")
        prev_right = None
        for idx, C in enumerate(cores):
            A = np.asarray(C, dtype=float)
            if not np.isrealobj(A):
                raise ValueError("All cores must be real-valued.")
            # if idx == 0:
            #     if A.ndim != 2:
            #         raise ValueError(f"Left boundary must be 2D; got {A.shape}")
            #     if A.shape[0] <= A.shape[1]:
            #         d, chi = A.shape
            #     else:
            #         chi, d = A.shape
            #         A = A.T
            #     B = A.T[:, None, :]
            #     prev_right = chi
            if idx == 0:
                # Left boundary: (d, bond)
                if A.ndim != 2:
                    raise ValueError(f"Left boundary must be 2D; got {A.shape}")
                d, chi = A.shape  # assume (physical, bond)
                B = A[:, None, :].T  # shape (d, 1, chi)
                prev_right = chi
            elif idx == n - 1:
                if A.ndim != 2:
                    raise ValueError(f"Right boundary must be 2D; got {A.shape}")
                chi, d = A.shape  # right boundary is chi × d
                B = A[:, None, :]  # (chi, 1, d)
                B = B.T
            else:
                if A.ndim != 3:
                    raise ValueError(f"Interior core at {idx} must be 3D; got {A.shape}")
                if prev_right is not None and A.shape[0] != prev_right:
                    # --- Handle degenerate bond-dimension=1 boundaries gracefully ---
                    if (prev_right == 1 or A.shape[0] == 1):
                        # pad or reshape to make them consistent for visualization only
                        if A.shape[0] < prev_right:
                            A = np.pad(A, ((0, prev_right - A.shape[0]), (0, 0), (0, 0)))
                        elif A.shape[0] > prev_right:
                            A = A[:prev_right, :, :]
                    else:
                        raise ValueError(
                            f"Interior core at {idx} left bond {A.shape[0]} != previous bond {prev_right}"
                        )

                B = A
                prev_right = A.shape[2]
            self.blocks.append(B)
            self.shapes.append(tuple(map(int, B.shape)))

        self._gmax = float(max((np.abs(B).max() for B in self.blocks), default=0.0)) if self.global_normalize else None

        super().__init__(
            cmap=cmap,
            normalize="none",
            min_fraction=0.0,
            min_value=None,
            background=background,
            transparent=transparent,
            show_colorbar=show_colorbar,
            title=title,
            aspect="manual",
            font_color=font_color,
            color_scale=self.color_scale,
            log_eps=log_eps,
            vmin=vmin,
            vmax=vmax,
            cmap_vmin=cmap_vmin,
            cmap_vmax=cmap_vmax,
        )

    def _coerce_to_cores(self, obj: Union[object, Sequence[np.ndarray]]) -> List[np.ndarray]:
        if isinstance(obj, (list, tuple)):
            return [np.asarray(x) for x in obj]
        tensors = getattr(obj, "tensors", None)
        if isinstance(tensors, (list, tuple)) and len(tensors) >= 2:
            return [np.asarray(x) for x in tensors]
        raise TypeError("MPSView expects an MPS-like object with .tensors or a sequence of arrays.")
    def _build_traces(self) -> List[go.BaseTraceType]:
        traces: List[go.BaseTraceType] = []
        x_cursor = 0.0
        cb_host_index = 0

        # treat values with abs <= eps as zero
        eps0 = 0.0  # exact zeros only
        # if you prefer numerical zeros too, use something like eps0 = 1e-15

        for idx, B in enumerate(self.blocks):
            I, J, K = self.shapes[idx]
            absB = np.abs(B)

            if self.color_scale == "log":
                base = np.clip(absB, self.log_eps, None)
                Lmin = float(np.log10(self.log_eps))
                A_disp = (np.log10(base) - Lmin) / (0.0 - Lmin)
                if (self.log_floor_abs is not None) and (self.log_floor_abs > 0):
                    floor_norm = self.log_floor_abs
                    pos = A_disp > 0
                    A_disp[pos] = np.maximum(A_disp[pos], floor_norm)
                tv_color_scale = "raw"
            else:
                A_disp = B if self.signed_values else absB
                tv_color_scale = "raw"

            tv_kwargs = dict(self.kwargs)
            tv_kwargs.update(
                dict(
                    cmap=self.cmap,
                    normalize="none",
                    color_scale=tv_color_scale,
                    voxel_opacity=self.voxel_opacity,
                    voxel_depth=self.voxel_depth,
                    background=self.background,
                    transparent=self.transparent_mode,
                    font_color=self.font_color,
                    integer_ticks=False,
                    show_ticks=self.show_ticks,
                    grid=False,
                    signed_values=self.signed_values,
                    show_colorbar=(
                        self.show_colorbar
                        and ((idx == cb_host_index) if self.share_colorbar else True)
                    ),
                    threshold=self.threshold,
                    cmap_vmin=self.cmap_vmin,
                    cmap_vmax=self.cmap_vmax,
                )
            )

            if self.min_value is not None:
                tv_kwargs["min_value"] = float(self.min_value)
                tv_kwargs.pop("min_fraction", None)
            elif self.min_fraction is not None:
                tv_kwargs["min_fraction"] = float(self.min_fraction)

            if self.min_value is not None:
                tv_kwargs["min_value"] = float(self.min_value)
                tv_kwargs.pop("min_fraction", None)
            elif self.min_fraction is not None:
                tv_kwargs["min_fraction"] = float(self.min_fraction)
            # if you want the default behavior to be "hide exact zeros"
            # keep eps0 = 0.0
            # if you want to hide tiny near zeros too, set eps0 to a small positive number

            tv = ThreeTensorView(A_disp, **tv_kwargs)
            f = tv.figure()
            keep_cb = self.show_colorbar and ((idx == cb_host_index) or (not self.share_colorbar))
            traces.extend(self._shift_traces_x_and_filter_colorbar(f.data, x_cursor, keep_cb))
            if self.mini_grid:
                traces.append(self._mini_grid_lines(I, J, K, x_off=x_cursor))
            x_cursor += I + self.gap_x

        return traces
    # def _build_traces(self) -> List[go.BaseTraceType]:
    #     traces: List[go.BaseTraceType] = []
    #     x_cursor = 0.0
    #     cb_host_index = 0
    #     for idx, B in enumerate(self.blocks):
    #         I, J, K = self.shapes[idx]
    #         denom = self._slice_denominator(B)
    #         absB = np.abs(B)
    #         if self.color_scale == "log":
    #             base = np.clip(absB / denom, self.log_eps, None)
    #             Lmin = float(np.log10(self.log_eps))
    #             A_disp = (np.log10(base) - Lmin) / (0.0 - Lmin)
    #             if (self.log_floor_abs is not None) and (self.log_floor_abs > 0):
    #                 floor_norm = self.log_floor_abs / denom
    #                 pos = A_disp > 0
    #                 A_disp[pos] = np.maximum(A_disp[pos], floor_norm)
    #             tv_color_scale = "raw"
    #         else:
    #             A_disp = absB / denom
    #             tv_color_scale = self.color_scale

    #         tv_kwargs = dict(self.kwargs)
    #         tv_kwargs.update(dict(
    #             cmap=self.cmap,
    #             normalize="none",
    #             color_scale=tv_color_scale,
    #             threshold=self.threshold,
    #             voxel_opacity=self.voxel_opacity,
    #             voxel_depth=self.voxel_depth,
    #             background=self.background,
    #             transparent=self.transparent_mode,
    #             font_color=self.font_color,
    #             integer_ticks=False,
    #             # CHANGED: respect the global toggle instead of forcing False
    #             show_ticks=self.show_ticks,
    #             grid=False,
    #             show_colorbar=(idx == cb_host_index) if self.share_colorbar else True,
    #         ))
    #         if self.min_value is not None:
    #             tv_kwargs["min_value"] = float(self.min_value) / denom
    #             tv_kwargs.pop("min_fraction", None)
    #         elif self.min_fraction is not None:
    #             tv_kwargs["min_fraction"] = float(self.min_fraction)

    #         tv = ThreeTensorView(A_disp, **tv_kwargs)
    #         f = tv.figure()
    #         keep_cb = (idx == cb_host_index) or (not self.share_colorbar)
    #         traces.extend(self._shift_traces_x_and_filter_colorbar(f.data, x_cursor, keep_cb))
    #         if self.mini_grid:
    #             traces.append(self._mini_grid_lines(I, J, K, x_off=x_cursor))
    #         x_cursor += I + self.gap_x
    #     return traces

    def _scene_axes(self) -> dict:
        total_width = self._total_width()
        pad_x = max(0.5, self.pad_frac * total_width)
        max_J = max(J for _, J, _ in self.shapes)
        max_K = max(K for _, _, K in self.shapes)
        pad_y = max(0.5, self.pad_frac * max_J)
        pad_z = max(0.5, self.pad_frac * max_K)
        centers = self._x_centers()
        labels = [str(i + 1) for i in range(len(self.blocks))]
        ar = dict(x=total_width, y=max_J, z=max_K)
        m = max(ar.values())
        ar = {k: v / m for k, v in ar.items()}

        # CHANGED: if ticks are disabled, also remove tick marks (ticks="")
        xaxis = dict(
            showbackground=False, showgrid=False, zeroline=False,
            showticklabels=self.show_ticks,
            ticks="outside" if self.show_ticks else "",
            tickmode="array",
            tickvals=centers,
            ticktext=labels,
            tickfont=dict(size=self.font_size, color=self.font_color),
            range=[-pad_x, total_width - self.gap_x + pad_x],
            title=dict(text="")
        )
        if not self.show_ticks:
            # With labels hidden, Plotly ignores ticktext; leaving it is harmless,
            # but we keep ticks="" above to remove the marks themselves.
            pass

        yaxis = dict(
            showbackground=False, showgrid=False, zeroline=False,
            showticklabels=False, ticks="",  # keep y ticks fully off
            range=[-pad_y, max_J + pad_y], title=dict(text="")
        )
        zaxis = dict(
            showbackground=False, showgrid=False, zeroline=False,
            showticklabels=False, ticks="",  # keep z ticks fully off
            range=[-pad_z, max_K + pad_z], title=dict(text="")
        )

        return dict(xaxis=xaxis, yaxis=yaxis, zaxis=zaxis, aspectratio=ar)

    def figure(self, *, force_rebuild: bool = False) -> go.Figure:
        fig = super().figure(force_rebuild=force_rebuild)
        if self.camera_eye is not None:
            ex, ey, ez = map(float, self.camera_eye)
        else:
            base = float(max(self._total_width(), max(J for _, J, _ in self.shapes), max(K for _, _, K in self.shapes)))
            wide = float(self._total_width()) / max(1.0, base)
            s = self.zoom * (1.6 + 1.2 * (wide ** 0.6))
            ex, ey, ez = (0.35 * s, 1.8 * s, 1.2 * s)
        ey = -abs(ey) if self.left_to_right else abs(ey)
        up_vec = dict(x=1, y=0, z=0) if self.lock_about_x else dict(x=0, y=0, z=1)
        drag = "turntable" if self.lock_about_x else "orbit"
        fig.update_layout(
            scene_camera=dict(eye=dict(x=ex, y=ey, z=ez), up=up_vec, projection=dict(type="perspective")),
            dragmode=drag,
            font=dict(size=self.font_size, color=self.font_color),
            title=dict(text=self.title or "", font=dict(size=self.font_size, color=self.font_color)),
        )
        return fig

    # def _slice_denominator(self, B: np.ndarray) -> float:
    #     if self.global_normalize:
    #         if self._gmax_override is not None:
    #             return self._gmax_override
    #         return self._gmax if (self._gmax and self._gmax > 0) else 1.0
    #     smax = float(np.abs(B).max())
    #     return smax if smax > 0 else 1.0
    def _slice_denominator(self, B: np.ndarray) -> float:
        """
        Deprecated: used to normalize each tensor block. Now returns 1.0 always,
        since all normalization is explicitly disabled unless log-scaling is used.
        """
        return 1.0
    def _total_width(self) -> float:
        widths = [I for (I, _, _) in self.shapes]
        return float(sum(widths) + self.gap_x * (len(widths) - 1))

    def _x_centers(self) -> List[float]:
        centers, x = [], 0.0
        for (I, _, _) in self.shapes:
            centers.append(x + 0.5 * I)
            x += I + self.gap_x
        return centers

    def _mini_grid_lines(self, I: int, J: int, K: int, *, x_off: float) -> go.Scatter3d:
        xs, ys, zs = [], [], []
        def seg(p0, p1):
            xs.extend([p0[0] + x_off, p1[0] + x_off, None])
            ys.extend([p0[1], p1[1], None])
            zs.extend([p0[2], p1[2], None])
        if self.mini_grid_mode == "bbox":
            corners = [(0,0,0),(I,0,0),(I,J,0),(0,J,0),(0,0,K),(I,0,K),(I,J,K),(0,J,K)]
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for i,j in edges: seg(corners[i], corners[j])
        else:
            si, sj, sk = self.mini_grid_stride
            for j in range(0, J+1, max(1, sj)):
                for k in range(0, K+1, max(1, sk)): seg((0,j,k),(I,j,k))
            for i in range(0, I+1, max(1, si)):
                for k in range(0, K+1, max(1, sk)): seg((i,0,k),(i,J,k))
            for i in range(0, I+1, max(1, si)):
                for j in range(0, J+1, max(1, sj)): seg((i,j,0),(i,j,K))
        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(width=2, color=self.mini_grid_color),
            opacity=self.mini_grid_opacity, hoverinfo="skip",
            name="mini-grid", showlegend=False
        )

    def _log_tickinfo(self) -> Dict[str, Any]:
        Lmin = float(np.log10(self.log_eps))
        lo = int(np.ceil(Lmin))
        exps = list(range(lo, 0 + 1))
        tvals = [(e - Lmin) / (0.0 - Lmin) for e in exps]
        return dict(tickvals=tvals, ticktext=[f"1e{e}" for e in exps])

    def _shift_traces_x_and_filter_colorbar(self, traces, x_off: float, keep_colorbar: bool):
        out = []
        for tr in traces:
            d = tr.to_plotly_json()

            if "x" in d:
                x = d["x"]
                if isinstance(x, (list, tuple)):
                    d["x"] = [(xx + x_off) if isinstance(xx, (int, float)) else None for xx in x]
                else:
                    d["x"] = (np.asarray(x, dtype=float) + x_off).tolist()

            is_mesh = (d.get("type", "").lower() == "mesh3d")
            is_dummy_cb = is_mesh and (len(d.get("i", [])) == 0)

            if is_dummy_cb:
                if not keep_colorbar:
                    continue

                cb = d.get("colorbar", {}) or {}
                cb.update(
                    dict(
                        thickness=40,
                        tickfont=dict(size=getattr(self, "font_size", 24), color=self.font_color),
                        title=dict(
                            text=cb.get("title", {}).get("text", ""),
                            font=dict(size=getattr(self, "font_size", 24), color=self.font_color),
                        ),
                        x=0.88,
                        xanchor="left",
                        y=0.5,
                        yanchor="middle",
                        len=0.9,
                    )
                )

                if self.color_scale == "log":
                    d["cauto"] = False
                    d["cmin"] = 0.0
                    d["cmax"] = 1.0
                    cb.update(self._log_tickinfo())

                d["colorbar"] = cb
                d["showscale"] = True
                out.append(go.Mesh3d(**d))
                continue

            out.append(go.Mesh3d(**d) if is_mesh else go.Scatter3d(**d))

        return out
        
class MPOView(BaseTensorView):
    """
    Visualize an MPO as a vertical stack of FourTensorView panels.

    Each core is rendered with FourTensorView (slices laid out internally along its x),
    then the whole panel is shifted along the global y to create a vertical column.

    Parameters mirror FourTensorView where sensible; global normalization is handled
    across all cores by computing a global max and passing it as a per-panel override.

    Expected core ranks:
      - Interior cores: 4D (bond, phys, phys, bond) or any 4D; no reordering enforced.
      - Left boundary:  4D is preferred (size, phys, bond, phys). If 3D is given, we
                        expand to 4D with a trailing singleton L-dimension.
      - Right boundary: 3D (bond, phys, phys) is allowed; expanded to 4D (bond, phys, phys, 1).

    If a core is 3D, we upgrade it to 4D by inserting a size-1 final axis.
    """
    def __init__(
        self,
        mpo_or_cores: Union[object, Sequence[np.ndarray]],
        *,
        gap_y: float = 1.5,
        pad_frac: float = 0.12,
        camera_eye: Optional[Tuple[float, float, float]] = None,
        zoom: float = 0.75,
        lock_about_x: bool = True,
        # FourTensorView-facing display knobs
        mini_grid: bool = True,
        mini_grid_stride: Tuple[int, int, int] = (1, 1, 1),
        mini_grid_color: str = "lightgray",
        mini_grid_opacity: float = 0.25,
        mini_grid_mode: str = "bbox",     # "bbox" or "full"
        cmap: str = "viridis",
        color_scale: str = "linear",      # "linear" | "log" | "raw"
        global_normalize: bool = True,    # across ALL cores
        global_max_override: Optional[float] = None,
        log_eps: float = 1e-12,
        log_floor_abs: Optional[float] = None,
        voxel_opacity: float = 1.0,
        voxel_depth: float = 0.6,
        background: str = "rgba(0,0,0,0)",
        transparent: bool = True,
        font_color: str = "black",
        threshold: Optional[bool] = None,
        min_value: Optional[float] = None,
        min_fraction: Optional[float] = None,
        share_colorbar: bool = True,
        cb_host_index: int = 0,           # which panel hosts the (shared) colorbar
        title: Optional[str] = None,
        font_size: int = 24,
        # Scene tick visibility (X/Z hidden, Y optionally labeled by block index)
        show_ticks: bool = True,
        **kwargs,
    ):
        cores = self._coerce_to_cores(mpo_or_cores)
        if color_scale not in {"linear", "log", "raw"}:
            raise ValueError("color_scale must be 'linear', 'log', or 'raw'.")

        # Promote any 3D boundary cores to 4D by appending a size-1 axis.
        self.blocks: List[np.ndarray] = []
        self.shapes4: List[Tuple[int, int, int, int]] = []
        for idx, C in enumerate(cores):
            A = np.asarray(C, dtype=float)
            if not np.isrealobj(A):
                raise ValueError(f"All MPO cores must be real-valued; core {idx} failed.")
            if A.ndim == 3:
                A = A[..., None]  # (χ, d, d) -> (χ, d, d, 1) or similar; we don't enforce ordering here.
            if A.ndim != 4:
                raise ValueError(f"MPO core {idx} must be 3D or 4D; got {A.shape}")
            self.blocks.append(A)
            self.shapes4.append(tuple(map(int, A.shape)))

        # Global max across ALL cores (if enabled); allows consistent color normalization.
        self._gmax = float(max((np.abs(B).max() for B in self.blocks), default=0.0)) if global_normalize else None
        self._gmax_override = (
            None if global_max_override is None
            else float(global_max_override) if global_max_override > 0 else None
        )

        # Threshold defaults align with your other views.
        user_set_thresh = (threshold is not None) or (min_value is not None) or (min_fraction is not None)
        self.threshold = bool(threshold) if threshold is not None else True
        self.min_value = min_value
        self.min_fraction = None if (min_fraction is None) else float(min_fraction)
        if not user_set_thresh:
            self.min_value = 0.0
            self.min_fraction = 0.0

        # Store options
        self.gap_y = float(gap_y)
        self.pad_frac = float(pad_frac)
        self.camera_eye = camera_eye
        self.zoom = float(zoom)
        self.lock_about_x = bool(lock_about_x)
        self.mini_grid = bool(mini_grid)
        self.mini_grid_stride = tuple(int(s) for s in mini_grid_stride)
        self.mini_grid_color = mini_grid_color
        self.mini_grid_opacity = float(mini_grid_opacity)
        self.mini_grid_mode = str(mini_grid_mode)
        self.cmap = cmap
        self.color_scale = color_scale
        self.global_normalize = bool(global_normalize)
        self.log_eps = float(log_eps)
        self.log_floor_abs = None if log_floor_abs is None else float(log_floor_abs)
        self.voxel_opacity = float(voxel_opacity)
        self.voxel_depth = float(voxel_depth)
        self.share_colorbar = bool(share_colorbar)
        self.cb_host_index = int(cb_host_index)
        self.title = title
        self.font_size = int(font_size)
        self.kwargs = dict(kwargs)
        self.show_ticks = bool(show_ticks)

        # For scene sizing we need per-panel extents (I, J, K, L) for each core.
        # FourTensorView lays out L slices along its local x, with x-span = I*L + gap*(L-1).
        self._panel_extents = [self._panel_extent_for_core(B) for B in self.blocks]  # (x_span, y_span, z_span)

        super().__init__(
            cmap=cmap,
            normalize="none",           # handled inside FourTensorView
            min_fraction=0.0,
            min_value=None,
            background=background,
            transparent=transparent,
            show_colorbar=False,        # FourTensorView handles the (shared) colorbar(s)
            title=title,
            aspect="manual",
            font_color=font_color,
            color_scale="linear",
            log_eps=log_eps,
        )

    # ---------------- Base hooks ----------------

    def _build_traces(self) -> List[go.BaseTraceType]:
        traces: List[go.BaseTraceType] = []
        y_cursor = 0.0

        # Choose a single global override if requested; otherwise per-core self-normalization
        gmax = None
        if self.global_normalize:
            gmax = self._gmax_override if (self._gmax_override is not None) else (self._gmax if (self._gmax and self._gmax > 0) else 1.0)

        for idx, B in enumerate(self.blocks):
            I, J, K, L = self.shapes4[idx]
            # Construct a FourTensorView for this core, with consistent normalization and display options.
            tv_kwargs = dict(self.kwargs)
            tv_kwargs.update(dict(
                gap_x=1.5,                      # per-panel spacing between L-slices (can be exposed if needed)
                mini_grid=self.mini_grid,
                mini_grid_stride=self.mini_grid_stride,
                mini_grid_color=self.mini_grid_color,
                mini_grid_opacity=self.mini_grid_opacity,
                mini_grid_mode=self.mini_grid_mode,
                share_colorbar=self.share_colorbar,
                camera_eye=None,                # camera handled at the MPO level
                zoom=1.0,
                lock_about_x=True,              # per-panel; overall drag mode set in figure()
                cmap=self.cmap,
                color_scale=self.color_scale,
                global_normalize=True,          # within-panel, but we'll override the max
                global_max_override=gmax,
                log_eps=self.log_eps,
                log_floor_abs=self.log_floor_abs,
                voxel_opacity=self.voxel_opacity,
                voxel_depth=self.voxel_depth,
                grid=False,
                background=self.background,
                transparent=True,
                font_color=self.font_color,
                index_ticks=False,              # we provide outer ticks along y only
                threshold=self.threshold,
                min_value=self.min_value,
                min_fraction=self.min_fraction,
                title=None,                     # outer title only
                font_size=self.font_size,
            ))
            tv = FourTensorView(B, **tv_kwargs)
            f = tv.figure()
            keep_cb = (idx == self.cb_host_index) or (not self.share_colorbar)

            # Shift the entire panel upward along global Y by y_cursor
            traces.extend(self._shift_traces_y_and_filter_colorbar(f.data, y_cursor, keep_cb))
            # Increment y cursor by this panel's y-span plus gap
            _, y_span, _ = self._panel_extents[idx]
            y_cursor += y_span + self.gap_y

        return traces

    def _scene_axes(self) -> dict:
        # Scene extents: x-span is the max panel width; y-span stacks; z-span is the max panel depth.
        x_spans, y_spans, z_spans = zip(*self._panel_extents) if self._panel_extents else ([1.0], [1.0], [1.0])
        total_y = sum(y_spans) + self.gap_y * (len(y_spans) - 1)
        max_x = max(x_spans)
        max_z = max(z_spans)

        pad_x = max(0.5, self.pad_frac * max_x)
        pad_y = max(0.5, self.pad_frac * total_y)
        pad_z = max(0.5, self.pad_frac * max_z)

        # y tick labels at panel centers
        centers = []
        y = 0.0
        for ys in y_spans:
            centers.append(y + 0.5 * ys)
            y += ys + self.gap_y
        labels = [str(i + 1) for i in range(len(centers))]

        # Aspect ratio normalization
        ar = dict(x=max_x, y=total_y, z=max_z)
        m = max(ar.values()) if len(ar) else 1.0
        ar = {k: (v / m if m > 0 else 1.0) for k, v in ar.items()}

        return dict(
            xaxis=dict(
                showbackground=False, showgrid=False, zeroline=False,
                showticklabels=False, ticks="",
                range=[-pad_x, max_x + pad_x],
                title=dict(text="")
            ),
            yaxis=dict(
                showbackground=False, showgrid=False, zeroline=False,
                showticklabels=self.show_ticks,
                ticks="outside" if self.show_ticks else "",
                tickmode="array", tickvals=centers, ticktext=labels,
                tickfont=dict(size=self.font_size, color=self.font_color),
                range=[-pad_y, total_y + pad_y],
                title=dict(text="")
            ),
            zaxis=dict(
                showbackground=False, showgrid=False, zeroline=False,
                showticklabels=False, ticks="",
                range=[-pad_z, max_z + pad_z],
                title=dict(text="")
            ),
            aspectratio=ar,
        )

    def figure(self, *, force_rebuild: bool = False) -> go.Figure:
        fig = super().figure(force_rebuild=force_rebuild)
        # Camera: similar philosophy to your other views; we look from +x,+y,+z quadrant.
        if self.camera_eye is not None:
            ex, ey, ez = map(float, self.camera_eye)
        else:
            # Base scale depends on the largest span
            x_spans, y_spans, z_spans = zip(*self._panel_extents) if self._panel_extents else ([1.0], [1.0], [1.0])
            max_span = float(max(max(x_spans), sum(y_spans), max(z_spans)))
            wide = float(max(x_spans)) / max(1.0, max_span)
            s = self.zoom * (1.6 + 1.2 * (wide ** 0.6))
            ex, ey, ez = (0.35 * s, 1.8 * s, 1.2 * s)

        up_vec = dict(x=1, y=0, z=0) if self.lock_about_x else dict(x=0, y=0, z=1)
        drag = "turntable" if self.lock_about_x else "orbit"
        fig.update_layout(
            scene_camera=dict(eye=dict(x=ex, y=ey, z=ez), up=up_vec, projection=dict(type="perspective")),
            dragmode=drag,
            font=dict(size=self.font_size, color=self.font_color),
            title=dict(text=self.title or "", font=dict(size=self.font_size, color=self.font_color)),
        )
        return fig

    # ---------------- helpers ----------------

    def _coerce_to_cores(self, obj: Union[object, Sequence[np.ndarray]]) -> List[np.ndarray]:
        if isinstance(obj, (list, tuple)):
            return [np.asarray(x) for x in obj]
        tensors = getattr(obj, "tensors", None)
        if isinstance(tensors, (list, tuple)) and len(tensors) >= 1:
            return [np.asarray(x) for x in tensors]
        raise TypeError("MPOView expects an MPO-like object with .tensors or a sequence of arrays.")

    def _panel_extent_for_core(self, A4: np.ndarray) -> Tuple[float, float, float]:
        """
        FourTensorView lays L slices along local x, each slice of width I, with gap_x between.
        We mirror FourTensorView's default gap_x=1.5 here for sizing (kept in tv_kwargs).
        """
        I, J, K, L = map(int, A4.shape)
        gap_x = 1.5
        x_span = L * I + (L - 1) * gap_x
        y_span = float(J)
        z_span = float(K)
        return float(x_span), y_span, z_span

    def _shift_traces_y_and_filter_colorbar(self, traces, y_off: float, keep_colorbar: bool):
        """
        Clone of your _shift_traces_x_and_filter_colorbar, but shifting Y instead of X.
        Respects shared-colorbar behavior and applies log tick correction when needed.
        """
        out = []
        for tr in traces:
            d = tr.to_plotly_json()

            # shift any numeric y entries by y_off
            if "y" in d:
                y = d["y"]
                if isinstance(y, (list, tuple)):
                    d["y"] = [(yy + y_off) if isinstance(yy, (int, float)) else None for yy in y]
                else:
                    d["y"] = (np.asarray(y, dtype=float) + y_off).tolist()

            is_mesh = (d.get("type", "").lower() == "mesh3d")
            is_dummy_cb = is_mesh and (len(d.get("i", [])) == 0)

            if is_dummy_cb:
                if not keep_colorbar:
                    continue
                cb = d.get("colorbar", {}) or {}
                cb.update(dict(
                    thickness=40,
                    tickfont=dict(size=self.font_size, color=self.font_color),
                    title=dict(text=cb.get("title", {}).get("text", ""),
                               font=dict(size=self.font_size, color=self.font_color)),
                    x=0.98, y=0.5, len=0.9,
                ))
                if self.color_scale == "log":
                    d["cauto"] = False
                    d["cmin"] = 0.0
                    d["cmax"] = 1.0
                    # reproduce FourTensorView's log ticks
                    Lmin = float(np.log10(self.log_eps))
                    lo = int(np.ceil(Lmin))
                    exps = list(range(lo, 0 + 1))
                    tvals = [(e - Lmin) / (0.0 - Lmin) for e in exps]
                    cb.update(dict(tickvals=tvals, ticktext=[f"1e{e}" for e in exps]))
                d["colorbar"] = cb
                d["showscale"] = True
                out.append(go.Mesh3d(**d))
            else:
                out.append(go.Mesh3d(**d) if is_mesh else go.Scatter3d(**d))

        return out
