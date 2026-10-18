# plotting.py: functions for plotting meta-analytic results

# package imp[orts------------------------------------------------------------------
import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import gaussian_kde

# plotting function-----------------------------------------------------------------
# import os
# import re
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import arviz as az
# from scipy.stats import gaussian_kde


def _posterior_terms(trace, var_name):
    """
    Plotting helper function to aid identifying posterior terms
    """
    da = trace.posterior[var_name]
    extra_dims = [d for d in da.dims if d not in ["chain", "draw"]]
    if len(extra_dims) == 0:
        return [(None, da.to_numpy().reshape(-1))]
    if len(extra_dims) > 1:
        raise ValueError(
            f"{var_name} has more than one non-sampling dimension: {extra_dims}"
        )
    dim = extra_dims[0]
    terms = da.coords[dim].to_numpy().tolist()
    return [
        (
            term,
            da.sel({dim: term}).to_numpy().reshape(-1),
        )
        for term in terms
    ]


def _posterior_kde(samples, gridsize=512):
    """
    Plotting helper function that allows kernel density estimation
    """
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) < 2:
        raise ValueError("Posterior distribution has fewer than 2 finite samples")
    if np.std(samples) == 0:
        return None, None
    q_low, q_high = np.quantile(samples, [0.001, 0.999])
    width = q_high - q_low
    if width == 0:
        width = np.std(samples) * 6
    pad = width * 0.10
    x = np.linspace(q_low - pad, q_high + pad, gridsize)
    kde = gaussian_kde(samples)
    y = kde(x)
    return x, y


def _safe_plot_name(value):
    """
    Plotting helper function that aids naming of plot axes
    """
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")


def _plot_posterior_density(
    ax,
    samples,
    label,
    color=None,
    hdi_prob=0.89,
    linewidth=2,
    hdi_alpha=0.20,
    show_hdi=True,
    show_mean=True,
    interval_low=None,
    interval_high=None,
    line_alpha=1.0,
    fill_alpha=None,
):
    """
    Plotting helper function to plot posterior density.
    Uses the helper for kernel density estimation
    """
    if fill_alpha is None:
        fill_alpha = hdi_alpha
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    x, y = _posterior_kde(samples)
    mean = np.mean(samples)
    hdi = np.asarray(
        az.hdi(
            samples,
            prob=hdi_prob,
        )
    )
    if interval_low is None:
        interval_low = hdi[0]
    if interval_high is None:
        interval_high = hdi[1]
    interval_low = float(interval_low)
    interval_high = float(interval_high)
    if x is None:
        ax.axvline(
            mean,
            color=color,
            linewidth=linewidth,
            label=label,
        )
    else:
        ax.plot(
            x,
            y,
            color=color,
            linewidth=linewidth,
            label=label,
            alpha=line_alpha,
        )
        if show_hdi:
            interval_idx = (x >= interval_low) & (x <= interval_high)
            ax.fill_between(
                x[interval_idx],
                0,
                y[interval_idx],
                color=color,
                alpha=fill_alpha,
            )
    if show_mean:
        ax.axvline(
            mean,
            color=color,
            linewidth=1.25,
            linestyle="--",
            alpha=0.80,
        )
    return {
        "mean": mean,
        "hdi_low": hdi[0],
        "hdi_high": hdi[1],
    }


def _plot_interval_line(
    ax,
    mean,
    hdi_low,
    hdi_high,
    y_position,
    marker="o",
    color=None,
    markerfacecolor=None,
    markeredgecolor=None,
    linewidth=3,
    markersize=6,
):
    """
    Plotting helper function to plot the interval line
    """
    mean = float(mean)
    hdi_low = float(hdi_low)
    hdi_high = float(hdi_high)
    y_position = float(y_position)
    ax.plot(
        [hdi_low, hdi_high],
        [y_position, y_position],
        color=color,
        linewidth=linewidth,
        solid_capstyle="round",
        clip_on=False,
    )
    ax.plot(
        mean,
        y_position,
        marker=marker,
        color=color,
        markerfacecolor=markerfacecolor,
        markeredgecolor=markeredgecolor,
        markersize=markersize,
        clip_on=False,
    )


def plot_bayes_meta_results(
    info,
    base_effects=("sham", "active", "contrast"),
    moderators=None,
    moderator_effects=("sham", "active", "contrast"),
    moderator_levels=None,
    moderator_plot="forest",
    forest_show_pp=True,
    forest_figsize=None,
    plot_mode="combined",
    colors=None,
    labels=None,
    hdi_prob=0.94,
    show_hdi=True,
    show_mean=False,
    show_zero=True,
    linewidth=2,
    hdi_alpha=0.20,
    figsize=(9, 6),
    xlabel="Posterior effect",
    xlabel_interpret=True,
    ylabel="Density",
    title=None,
    save_dir=None,
    file_prefix="effect_plot_bayes_meta",
    file_type="tiff",
    dpi=300,
    show=True,
    show_interval_lines=True,
    interval_start=0.04,
    interval_spacing=0.045,
    interval_linewidth=3,
    interval_markersize=6,
    best_interval_markersize=6.5,
    best_marker_shape="D",
    best_marker_face_color="snow",
    best_marker_edge_color="darkred",
    line_alpha_default=1.0,
    fill_alpha_default=0.20,
    line_alpha_map=None,
    fill_alpha_map=None,
    zero_line_color="black",
    special_level_labels={
        "site_name": {
            "dlpfc": " dlPFC",
            "ifg": "IFG",
            "ipl": "IPL",
            "mpfc": "mPFC",
            "vlpfc": "vlPFC",
            "vmpfc": "vmPFC",
            "cerebellum": "Cerebellum",
        },
        "diagnosis": {"BPD": "BPD"},
        "eeg_site": {"Fpz": "FPz"},
        "measure_scale_name": {
            "SSRT": "SSRT",
            "rt": "RT",
            "rt_neutral": "RT Neutral",
            "rt_angry": "RT Angry",
        },
    },
    special_mod_labels={"eeg_site": "EEG Site"},
    best_levels=None,
):
    """
    Plot posterior effects from Bayesian meta-analysis results.

    Display base-model effects as posterior densities and moderator effects
    as posterior densities or a grouped forest plot. Optionally save figures
    and highlight specified categorical moderator levels.

    Inputs
    ------
    info : dict
        Analysis results containing the entries required by the selected plots:

        - ``"trace_base"``: posterior trace containing ``theta_sham``,
          ``theta_active``, and/or ``theta_contrast``.
        - ``"base_summary"``: pandas DataFrame indexed by posterior variable
          name, with ``pp_improve``, ``eti89_lb``, and ``eti89_ub`` columns.
        - ``"trace_mod"``: dictionary mapping moderator names to posterior
          traces.
        - ``"moderator_summaries"``: dictionary mapping moderator names to
          summary DataFrames with the same columns as ``base_summary``.
        - ``"improvement_direction"``: direction used in the x-axis
          annotation, required when `xlabel_interpret` is True.

        Categorical moderator traces contain ``theta_{effect}_category``
        variables, with summary rows named
        ``theta_{effect}_category[level]``. Other moderator traces contain
        ``beta_moderator_{effect}`` variables, with summary rows named
        ``beta_moderator_{effect}`` or
        ``beta_moderator_{effect}[term]``.

        Each trace must expose a ``posterior`` dataset. Moderator variables
        may have at most one dimension other than ``chain`` and ``draw``.
    base_effects : sequence of str or None, optional
        Base-model effects to plot, in the supplied order. Valid entries are
        ``"sham"``, ``"active"``, and ``"contrast"``, where ``"contrast"``
        identifies the active > sham contrast. Default is all three effects.
        If None, omit base-model effects.
    moderators : str, sequence of str, or None, optional
        Moderator name or names to plot, in the supplied order. Use
        ``"all"`` to select every moderator in ``info["trace_mod"]``.
        Default is None, which omits moderator effects.
    moderator_effects : str or sequence of str, optional
        Effects to plot for each moderator. Valid entries are ``"sham"``,
        ``"active"``, and ``"contrast"``. Default is all three effects.
        Requested variables absent from a moderator trace are skipped.
    moderator_levels : dict or None, optional
        Mapping from categorical moderator names to sequences of levels.
        Supplied sequences determine both selection and plotting order.
        Levels absent from the trace are ignored. Moderators omitted from
        the mapping retain all levels in posterior coordinate order.
        Default is None.
    moderator_plot : {"density", "forest"}, optional
        Display moderator effects as posterior densities or as one grouped
        forest plot. Default is ``"forest"``. Base-model effects always use
        density plots.
    forest_show_pp : bool, optional
        Whether to annotate forest-plot rows with posterior probabilities
        of improvement, rounded to two decimal places. Default is True.
    forest_figsize : tuple of float or None, optional
        Forest-plot width and height in inches. If None, use the width from
        `figsize` and calculate the height from the number of plotted rows
        and moderator groups. Default is None.
    plot_mode : {"combined", "category_combined", "separate", "both", "all"}, optional
        Arrangement of density plots. Default is ``"combined"``.

        - ``"combined"``: overlay all selected densities in one figure.
        - ``"category_combined"``: create one figure for the base model
          and one for each selected moderator.
        - ``"separate"``: create one figure per posterior distribution.
        - ``"both"``: create category-combined and separate figures.
        - ``"all"``: create all three arrangements.

        When `moderator_plot` is ``"forest"``, these arrangements apply
        only to base-model densities. The moderator forest plot is
        generated independently of `plot_mode`.
    colors : dict or None, optional
        Mapping from plot keys to Matplotlib-compatible colors.
        Base-model keys use ``"base:{effect}"``. Categorical moderator
        keys use ``"{moderator}:{level}:{effect}"``. Other moderator keys
        use ``"{moderator}:{effect}"`` or
        ``"{moderator}:{term}:{effect}"``.

        Moderator colors are resolved using the full plot key, then
        ``"{moderator}:{effect}"``, then ``"{effect}"``. Unspecified colors
        are delegated to Matplotlib. Default is None.
    labels : dict or None, optional
        Custom density labels indexed by the full plot keys described
        under `colors`. The posterior probability of improvement is
        appended to each label. Forest-plot labels are controlled
        separately by `special_level_labels` and `special_mod_labels`.
        Default is None.
        e.g.,
              labels= {
                   "base:sham":"Sham",
                   "base:active":"Active",
                   "base:contrast":"Active > Sham",}
    hdi_prob : float, optional
        Probability mass for highest-density intervals calculated from
        posterior samples and returned in density-plot statistics.
        Default is 0.94. This does not change the displayed intervals,
        which use ``eti89_lb`` and ``eti89_ub`` from the summary tables.
    show_hdi : bool, optional
        Whether to shade the density within the supplied summary interval.
        Despite the parameter name, the shaded bounds come from
        ``eti89_lb`` and ``eti89_ub``. Default is True.
    show_mean : bool, optional
        Whether to draw a dashed vertical line at each posterior mean in
        density plots. Does not control interval or forest-plot markers.
        Default is False.
    show_zero : bool, optional
        Whether to draw a vertical reference line at zero.
        Default is True.
    linewidth : float, optional
        Width of posterior density lines in points. Default is 2.
    hdi_alpha : float, optional
        Fallback interval-fill opacity passed to the density helper.
        Overridden by the resolved `fill_alpha_map` or
        `fill_alpha_default` value unless that value is None.
        Default is 0.20.
    figsize : tuple of float, optional
        Density-plot width and height in inches. Also supplies the default
        forest-plot width. Default is (9, 6).
    xlabel : str, optional
        X-axis label for all figures. Default is ``"Posterior effect"``.
    xlabel_interpret : bool, optional
        Whether to append an annotation identifying the improvement
        direction relative to zero, using
        ``info["improvement_direction"]``. This affects labeling only.
        Default is True.
    ylabel : str, optional
        Y-axis label for density plots. Default is ``"Density"``.
        Forest plots have no y-axis title.
    title : str or None, optional
        Figure title or title prefix. If None, the combined density plot
        has no title; other figures receive automatically generated titles.
        Default is None.
    save_dir : str or path-like or None, optional
        Directory in which to save figures. Created if necessary.
        If None, figures are not saved. Default is None.
    file_prefix : str, optional
        Prefix for saved filenames. Plot-specific suffixes are appended.
        Default is ``"effect_plot_bayes_meta"``.
    file_type : str, optional
        Output filename extension, without a leading period. Must specify
        a format supported by the active Matplotlib backend.
        Default is ``"tiff"``.
    dpi : float, optional
        Resolution used when saving figures. Default is 300.
    show : bool, optional
        Whether to display figures without blocking execution. If True,
        select the QtAgg backend, which requires an available Qt binding
        and graphical display. If False, close figures after creation
        and optional saving while retaining them in the returned dictionary.
        Default is True.
    show_interval_lines : bool, optional
        Whether to draw horizontal summary intervals and posterior mean
        markers below combined and category-combined density plots.
        Does not control forest plots or separate density plots.
        Default is True.
    interval_start : float, optional
        Currently overwritten by automatic interval positioning based on
        the density-axis height; the supplied value has no effect.
        Default is 0.04.
    interval_spacing : float, optional
        Currently overwritten by automatic interval spacing based on
        the density-axis height; the supplied value has no effect.
        Default is 0.045.
    interval_linewidth : float, optional
        Width of horizontal interval lines in density and forest plots,
        in points. Default is 3.
    interval_markersize : float, optional
        Size of ordinary posterior mean markers in density and forest
        plots, in points. Default is 6.
    best_interval_markersize : float, optional
        Size of highlighted forest-plot markers in points.
        Default is 6.5.
    best_marker_shape : str, optional
        Matplotlib marker used for highlighted categorical moderator levels
        in the forest plot. Default is ``"D"`` (diamond).
    best_marker_face_color : color, optional
        Matplotlib-compatible fill color for highlighted forest-plot
        markers. Default is ``"snow"``.
    best_marker_edge_color : color, optional
        Matplotlib-compatible edge color for highlighted forest-plot
        markers. Default is ``"darkred"``.
    line_alpha_default : float, optional
        Default density-line opacity, between 0 and 1.
        Default is 1.0.
    fill_alpha_default : float, optional
        Default density interval-fill opacity, between 0 and 1.
        Default is 0.20.
    line_alpha_map : dict or None, optional
        Density-line opacity overrides. Values are resolved using the full
        plot key, then the effect name, then `line_alpha_default`.
        Default is None.
    fill_alpha_map : dict or None, optional
        Density interval-fill opacity overrides. Values are resolved using
        the full plot key, then the effect name, then `fill_alpha_default`.
        Default is None.
    zero_line_color : color, optional
        Matplotlib-compatible zero-reference-line color for
        category-combined, separate, and forest plots. The combined density
        plot does not currently use this parameter. Default is ``"black"``.
    special_level_labels : dict or None, optional
        Nested mapping from moderator names to mappings of level values
        to forest-plot labels. Defaults to the built-in label mappings
        shown in the signature. Unmapped levels have underscores replaced
        with spaces and are converted to title case. If None, use this
        automatic formatting for every level.
    special_mod_labels : dict or None, optional
        Mapping from moderator names to forest-plot group headings.
        Default is ``{"eeg_site": "EEG Site"}``. Unmapped names have
        underscores replaced with spaces and are converted to title case.
        If None, use this automatic formatting for every heading.
    best_levels : dict, optional
        Mapping from categorical moderator names to individual levels
        to highlight in the forest plot. The selected level is highlighted
        for every plotted effect. Levels must match posterior coordinates.
        No ranking or evidence threshold is calculated by this function.
        The signature default is None; the current categorical-moderator
        implementation requires a dictionary. Pass an empty dictionary
        to omit highlighting.

    Returns
    -------
    figures : dict
        Generated figures and posterior statistics with the following keys:

        - ``"combined"``: combined density Figure, or None.
        - ``"category_combined"``: dictionary of density Figures keyed by
          ``"base"`` or moderator name.
        - ``"separate"``: dictionary of density Figures keyed by full
          plot key.
        - ``"forest"``: moderator forest Figure, or None.
        - ``"statistics"``: dictionary of statistics keyed by full
          plot key.

        Density statistics contain ``"mean"``, ``"hdi_low"``, and
        ``"hdi_high"``, calculated from finite posterior samples.
        Forest statistics contain ``"mean"``, ``"eti_low"``,
        ``"eti_high"``, and ``"pp_improve"``. Forest probabilities are
        rounded to two decimal places.

    Raises
    ------
    ValueError
        If a plot mode or effect name is invalid, no posterior
        distributions are selected, a moderator variable has more than
        one non-sampling dimension, or a density has fewer than two
        finite samples.
    KeyError
        If a required results entry, base-model posterior variable,
        moderator, summary row, or summary column is missing.

    Notes
    -----
    Displayed interval bounds and probabilities of improvement are read
    from the supplied summary tables. They are not recalculated from
    `hdi_prob`, and posterior samples are not directionally recoded.

    Nonconstant posterior densities are estimated using a Gaussian kernel.
    Constant samples are represented by a vertical line.
    """
    if plot_mode not in ["combined", "category_combined", "separate", "both", "all"]:
        raise ValueError(
            "plot_mode must be 'combined', 'category_combined', 'separate', 'both', or 'all'"
        )
    if moderator_plot not in ["density", "forest"]:
        raise ValueError("moderator_plot must be 'density' or 'forest'")
    if colors is None:
        colors = {}
    if labels is None:
        labels = {}
    if moderator_levels is None:
        moderator_levels = {}
    valid_effects = {"sham", "active", "contrast"}
    if base_effects is not None:
        invalid = set(base_effects) - valid_effects
        if len(invalid) > 0:
            raise ValueError(f"Invalid base effects: {sorted(invalid)}")
    if moderator_effects is not None:
        if isinstance(moderator_effects, str):
            moderator_effects = [moderator_effects]
        invalid = set(moderator_effects) - valid_effects
        if len(invalid) > 0:
            raise ValueError(f"Invalid moderator effects: {sorted(invalid)}")
    plot_data = []
    if base_effects is not None:
        base_trace = info["trace_base"]
        base_sum = info["base_summary"]
        for effect in base_effects:
            var_name = f"theta_{effect}"
            pp_imp = round(
                base_sum.iloc[np.where(base_sum.index == var_name)[0], :].pp_improve, 2
            ).item()
            eti_low = float(base_sum.loc[var_name, "eti89_lb"])
            eti_high = float(base_sum.loc[var_name, "eti89_ub"])
            if var_name not in base_trace.posterior:
                raise KeyError(f"{var_name} not found in base trace")
            samples = base_trace.posterior[var_name].to_numpy().reshape(-1)
            key = f"base:{effect}"
            default_label = rf"$\mathbf{{{effect.capitalize()}}}$"
            display_label = labels.get(key, default_label) + f"\n P(Improve)= {pp_imp}"
            plot_data.append(
                {
                    "key": key,
                    "label": display_label,
                    "samples": samples,
                    "color": colors.get(key, None),
                    "source": "base",
                    "effect": effect,
                    "eti_low": eti_low,
                    "eti_high": eti_high,
                }
            )
    if moderators is not None:
        if moderators == "all":
            moderators = list(info["trace_mod"].keys())
        elif isinstance(moderators, str):
            moderators = [moderators]
        for mn in moderators:
            if mn not in info["trace_mod"]:
                raise KeyError(f"Moderator '{mn}' not found in info['trace_mod']")
            mod_trace = info["trace_mod"][mn]
            mod_sum = info["moderator_summaries"][mn]
            posterior_vars = set(mod_trace.posterior.data_vars)
            is_categorical = "theta_sham_category" in posterior_vars
            if is_categorical:
                best_level = None
                if best_levels is not None:
                    best_level = best_levels.get(mn, None)
                for effect in moderator_effects:
                    var_name = f"theta_{effect}_category"
                    if var_name not in posterior_vars:
                        continue
                    terms = _posterior_terms(
                        mod_trace,
                        var_name,
                    )
                    selected_levels = moderator_levels.get(mn, None)
                    # ordering levels according to moderator_levels when supplied
                    if selected_levels is not None:
                        term_lookup = {level: samples for level, samples in terms}
                        terms = [
                            (level, term_lookup[level])
                            for level in selected_levels
                            if level in term_lookup
                        ]
                    # extracting each level's matching posterior summary
                    for level, samples in terms:
                        marker = best_marker_shape if level == best_level else "o"
                        row_var_name = var_name + f"[{level}]"
                        pp_imp = round(
                            mod_sum.loc[row_var_name, "pp_improve"],
                            2,
                        )
                        eti_low = float(mod_sum.loc[row_var_name, "eti89_lb"])
                        eti_high = float(mod_sum.loc[row_var_name, "eti89_ub"])
                        key = f"{mn}:{level}:{effect}"
                        default_label = (
                            rf"$\mathbf{{{level.replace('_',' ').capitalize().replace(' ',r'\ ')}}}: "
                            rf"\mathbf{{{effect.capitalize()}}}$"
                        )
                        display_label = (
                            labels.get(key, default_label) + f"\n P(Improve)= {pp_imp}"
                        )
                        plot_data.append(
                            {
                                "key": key,
                                "label": display_label,
                                "samples": samples,
                                "color": colors.get(
                                    key,
                                    colors.get(
                                        f"{mn}:{effect}",
                                        colors.get(effect, None),
                                    ),
                                ),
                                "marker": marker,
                                "source": mn,
                                "level": level,
                                "effect": effect,
                                "eti_low": eti_low,
                                "eti_high": eti_high,
                                "pp_improve": pp_imp,
                            }
                        )
            else:
                for effect in moderator_effects:
                    var_name = f"beta_moderator_{effect}"
                    if var_name not in posterior_vars:
                        continue
                    terms = _posterior_terms(
                        mod_trace,
                        var_name,
                    )
                    for term, samples in terms:
                        if term is None:
                            row_var_name = var_name
                            key = f"{mn}:{effect}"
                            default_label = rf"$\mathbf{{{effect.capitalize()}}}$"
                        else:
                            row_var_name = f"{var_name}[{term}]"
                            key = f"{mn}:{term}:{effect}"
                            term_label = (
                                str(term)
                                .replace("_", " ")
                                .capitalize()
                                .replace(" ", r"\ ")
                            )
                            default_label = rf"$\mathbf{{{term_label}}}: \mathbf{{{effect.capitalize()}}}$"
                        pp_imp = round(
                            mod_sum.loc[row_var_name, "pp_improve"],
                            2,
                        )
                        eti_low = float(mod_sum.loc[row_var_name, "eti89_lb"])
                        eti_high = float(mod_sum.loc[row_var_name, "eti89_ub"])
                        display_label = (
                            labels.get(key, default_label) + f"\n P(Improve)= {pp_imp}"
                        )
                        plot_data.append(
                            {
                                "key": key,
                                "label": display_label,
                                "samples": samples,
                                "color": colors.get(
                                    key,
                                    colors.get(
                                        f"{mn}:{effect}",
                                        colors.get(effect, None),
                                    ),
                                ),
                                "marker": "o",
                                "source": mn,
                                "effect": effect,
                                "eti_low": eti_low,
                                "eti_high": eti_high,
                                "pp_improve": pp_imp,
                            }
                        )
    if len(plot_data) == 0:
        raise ValueError("No posterior distributions were selected for plotting")
    base_plot_data = [item for item in plot_data if item["source"] == "base"]
    moderator_plot_data = [item for item in plot_data if item["source"] != "base"]
    if moderator_plot == "forest":
        density_plot_data = base_plot_data
    else:
        density_plot_data = plot_data
    if save_dir is not None:
        os.makedirs(
            save_dir,
            exist_ok=True,
        )
    if show:
        matplotlib.use("qtagg")
    if xlabel_interpret:
        xlabel_imp = f"\nP(Improve)= {info["improvement_direction"]} 0"
    else:
        xlabel_imp = ""
    figures = {
        "combined": None,
        "category_combined": {},
        "separate": {},
        "forest": None,
        "statistics": {},
    }
    if plot_mode in ["combined", "all"] and len(density_plot_data) > 0:
        fig, ax = plt.subplots(
            figsize=figsize,
        )
        if show_zero:
            ax.axvline(
                0,
                linewidth=1,
                linestyle=":",
                alpha=0.70,
            )
        # for item in plot_data:
        interval_stats = []
        for i, item in enumerate(density_plot_data):
            line_alpha = (
                line_alpha_map.get(
                    item["key"],
                    line_alpha_map.get(
                        item.get("effect"),
                        line_alpha_default,
                    ),
                )
                if line_alpha_map is not None
                else line_alpha_default
            )
            fill_alpha = (
                fill_alpha_map.get(
                    item["key"],
                    fill_alpha_map.get(
                        item.get("effect"),
                        fill_alpha_default,
                    ),
                )
                if fill_alpha_map is not None
                else fill_alpha_default
            )
            stats = _plot_posterior_density(
                ax=ax,
                samples=item["samples"],
                label=item["label"],
                color=item["color"],
                line_alpha=line_alpha,
                fill_alpha=fill_alpha,
                hdi_prob=hdi_prob,
                linewidth=linewidth,
                hdi_alpha=hdi_alpha,
                show_hdi=show_hdi,
                show_mean=show_mean,
                interval_low=item["eti_low"],
                interval_high=item["eti_high"],
            )
            figures["statistics"][item["key"]] = stats
            interval_stats.append((item, stats))
        ax.axhline(
            0,
            linewidth=0.8,
            linestyle="-",
            alpha=0.25,
        )
        if show_interval_lines:
            density_max = ax.get_ylim()[1]
            interval_spacing = density_max * 0.035
            interval_start = -density_max * 0.025
            interval_bottom = interval_start - interval_spacing * (
                len(interval_stats) - 1
            )
            ax.set_ylim(
                interval_bottom - density_max * 0.015,
                density_max,
            )
            for i, (item, stats) in enumerate(interval_stats):
                y_position = interval_start - i * interval_spacing
                _plot_interval_line(
                    ax=ax,
                    mean=stats["mean"],
                    hdi_low=item["eti_low"],
                    hdi_high=item["eti_high"],
                    y_position=y_position,
                    color=item["color"],
                    linewidth=interval_linewidth,
                    markersize=interval_markersize,
                )
            density_ticks = [tick for tick in ax.get_yticks() if tick >= 0]
            ax.set_yticks(density_ticks)
        ax.set_xlabel(rf"$\mathbf{{{xlabel.replace(' ',r'\ ')}}}$" + f"{xlabel_imp}")
        ax.set_ylabel(ylabel, fontweight="bold")
        if title is not None:
            ax.set_title(title, fontweight="bold")
        ax.legend(
            frameon=False,
        )
        fig.tight_layout()
        if save_dir is not None:
            save_path = os.path.join(
                save_dir,
                f"{file_prefix}_combined.{file_type}",
            )
            fig.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
            )
        figures["combined"] = fig
        if show:
            plt.show(block=False)
        else:
            plt.close(fig)
    if plot_mode in ["category_combined", "both", "all"] and len(density_plot_data) > 0:
        plot_groups = {}
        # for item in plot_data:
        for item in density_plot_data:
            source = item["source"]
            if source not in plot_groups:
                plot_groups[source] = []
            plot_groups[source].append(item)
        for source, items in plot_groups.items():
            fig, ax = plt.subplots(
                figsize=figsize,
            )
            if show_zero:
                ax.axvline(
                    0,
                    linewidth=1,
                    linestyle=":",
                    color=zero_line_color,
                    alpha=0.70,
                )
            interval_stats = []
            for i, item in enumerate(items):
                line_alpha = (
                    line_alpha_map.get(
                        item["key"],
                        line_alpha_map.get(
                            item.get("effect"),
                            line_alpha_default,
                        ),
                    )
                    if line_alpha_map is not None
                    else line_alpha_default
                )
                fill_alpha = (
                    fill_alpha_map.get(
                        item["key"],
                        fill_alpha_map.get(
                            item.get("effect"),
                            fill_alpha_default,
                        ),
                    )
                    if fill_alpha_map is not None
                    else fill_alpha_default
                )
                stats = _plot_posterior_density(
                    ax=ax,
                    samples=item["samples"],
                    label=item["label"],
                    color=item["color"],
                    line_alpha=line_alpha,
                    fill_alpha=fill_alpha,
                    hdi_prob=hdi_prob,
                    linewidth=linewidth,
                    hdi_alpha=hdi_alpha,
                    show_hdi=show_hdi,
                    show_mean=show_mean,
                    interval_low=item["eti_low"],
                    interval_high=item["eti_high"],
                )
                figures["statistics"][item["key"]] = stats
                interval_stats.append((item, stats))
            ax.axhline(
                0,
                linewidth=0.8,
                linestyle="-",
                alpha=0.25,
            )
            if show_interval_lines:
                density_max = ax.get_ylim()[1]
                interval_start = -density_max * 0.025
                interval_spacing = density_max * 0.035
                interval_bottom = interval_start - interval_spacing * (
                    len(interval_stats) - 1
                )
                ax.set_ylim(
                    interval_bottom - density_max * 0.015,
                    density_max,
                )
                for i, (item, stats) in enumerate(interval_stats):
                    y_position = interval_start - i * interval_spacing
                    _plot_interval_line(
                        ax=ax,
                        mean=stats["mean"],
                        hdi_low=item["eti_low"],
                        hdi_high=item["eti_high"],
                        y_position=y_position,
                        color=item["color"],
                        linewidth=interval_linewidth,
                        markersize=interval_markersize,
                    )
                density_ticks = [tick for tick in ax.get_yticks() if tick >= 0]
                ax.set_yticks(density_ticks)
            ax.set_xlabel(
                rf"$\mathbf{{{xlabel.replace(' ',r'\ ')}}}$" + f"{xlabel_imp}"
            )
            ax.set_ylabel(
                ylabel,
                fontweight="bold",
            )
            if source == "base":
                category_title = "Base model"
            else:
                category_title = str(source).replace("_", " ")
            if title is not None:
                ax.set_title(
                    rf"$\mathbf{{{title.replace(' ',r'\ ').capitalize()}}}: $"
                    + f"{category_title.capitalize()}"
                )
            else:
                ax.set_title(
                    category_title.capitalize(),
                    fontweight="bold",
                )
            ax.legend(
                frameon=False,
            )
            fig.tight_layout()
            if save_dir is not None:
                safe_source = _safe_plot_name(source)
                save_path = os.path.join(
                    save_dir,
                    f"{file_prefix}_{safe_source}_combined.{file_type}",
                )
                fig.savefig(
                    save_path,
                    dpi=dpi,
                    bbox_inches="tight",
                )
            figures["category_combined"][source] = fig
            if show:
                plt.show(block=False)
            else:
                plt.close(fig)
    if plot_mode in ["separate", "both", "all"] and len(density_plot_data) > 0:
        for item in density_plot_data:
            fig, ax = plt.subplots(
                figsize=figsize,
            )
            if show_zero:
                ax.axvline(
                    0,
                    linewidth=1,
                    linestyle=":",
                    color=zero_line_color,
                    alpha=0.70,
                )
            line_alpha = (
                line_alpha_map.get(
                    item["key"],
                    line_alpha_map.get(
                        item.get("effect"),
                        line_alpha_default,
                    ),
                )
                if line_alpha_map is not None
                else line_alpha_default
            )
            fill_alpha = (
                fill_alpha_map.get(
                    item["key"],
                    fill_alpha_map.get(
                        item.get("effect"),
                        fill_alpha_default,
                    ),
                )
                if fill_alpha_map is not None
                else fill_alpha_default
            )
            stats = _plot_posterior_density(
                ax=ax,
                samples=item["samples"],
                label=item["label"],
                color=item["color"],
                line_alpha=line_alpha,
                fill_alpha=fill_alpha,
                hdi_prob=hdi_prob,
                linewidth=linewidth,
                hdi_alpha=hdi_alpha,
                show_hdi=show_hdi,
                show_mean=show_mean,
                interval_low=item["eti_low"],
                interval_high=item["eti_high"],
            )
            figures["statistics"][item["key"]] = stats
            interval_stats.append((item, stats))
            ax.axhline(
                0,
                linewidth=0.8,
                linestyle="-",
                alpha=0.25,
            )
            if show_interval_lines:
                density_max = ax.get_ylim()[1]
                interval_start = -density_max * 0.025
                interval_spacing = density_max * 0.035
                interval_bottom = interval_start - interval_spacing * (
                    len(interval_stats) - 1
                )
                ax.set_ylim(
                    interval_bottom - density_max * 0.015,
                    density_max,
                )
                for i, (item, stats) in enumerate(interval_stats):
                    y_position = interval_start - i * interval_spacing
                    _plot_interval_line(
                        ax=ax,
                        mean=stats["mean"],
                        hdi_low=item["eti_low"],
                        hdi_high=item["eti_high"],
                        y_position=y_position,
                        color=item["color"],
                        linewidth=interval_linewidth,
                        markersize=interval_markersize,
                    )
                density_ticks = [tick for tick in ax.get_yticks() if tick >= 0]
                ax.set_yticks(density_ticks)
            ax.set_xlabel(
                rf"$\mathbf{{{xlabel.replace(' ',r'\ ')}}}$" + f"{xlabel_imp}"
            )
            ax.set_ylabel(ylabel, fontweight="bold")
            ax.set_title(
                (
                    rf"$\mathbf{{{item["label"].replace(' ',r'\ ').capitalize()}}}: $"
                    if title is None
                    else rf"$\mathbf{{{title.replace(' ',r'\ ').capitalize()}}}: $"
                    + f"{item['label'].replace('_', ' ').capitalize().replace(" ",r"\ ")}"
                ),
            )
            fig.tight_layout()
            if save_dir is not None:
                safe_key = _safe_plot_name(item["key"])
                save_path = os.path.join(
                    save_dir,
                    f"{file_prefix}_{safe_key}.{file_type}",
                )
                fig.savefig(
                    save_path,
                    dpi=dpi,
                    bbox_inches="tight",
                )
            figures["separate"][item["key"]] = fig
            if show:
                plt.show(block=False)
            else:
                plt.close(fig)
    # Single moderator forest plot
    if moderator_plot == "forest" and len(moderator_plot_data) > 0:
        # organizing moderators into visually separated groups
        moderator_order = list(
            dict.fromkeys(item["source"] for item in moderator_plot_data)
        )
        forest_rows = []
        group_headers = []
        group_separators = []
        current_y = 0
        for group_i, moderator_name in enumerate(moderator_order):
            group_items = [
                item for item in moderator_plot_data if item["source"] == moderator_name
            ]
            group_headers.append((moderator_name, current_y))
            current_y += 0.40
            for item in group_items:
                forest_rows.append((item, current_y))
                current_y += 0.60
            if group_i < len(moderator_order) - 1:
                group_separators.append(current_y - 0.20)
                current_y += 0.30
        # sizing forest plot
        if forest_figsize is None:
            forest_size = (figsize[0], max(4, 0.30 * current_y + 1.0))
        else:
            forest_size = forest_figsize
        fig, ax = plt.subplots(figsize=forest_size)
        if show_zero:
            ax.axvline(
                0,
                linewidth=1,
                linestyle=":",
                color=zero_line_color,
                alpha=0.70,
            )
        # plotting moderator estimates
        y_positions = []
        forest_labels = []
        for item, y_position in forest_rows:
            mean = float(np.mean(item["samples"]))
            _plot_interval_line(
                ax=ax,
                mean=mean,
                hdi_low=item["eti_low"],
                hdi_high=item["eti_high"],
                y_position=y_position,
                color=item["color"],
                markerfacecolor=(
                    best_marker_face_color
                    if item["marker"] == best_marker_shape
                    else item["color"]
                ),
                markeredgecolor=(
                    best_marker_edge_color
                    if item["marker"] == best_marker_shape
                    else item["color"]
                ),
                marker=item["marker"],
                linewidth=interval_linewidth,
                markersize=(
                    best_interval_markersize
                    if item["marker"] == best_marker_shape
                    else interval_markersize
                ),
            )
            y_positions.append(y_position)
            if "level" in item:
                moderator = str(item["source"])
                level = str(item["level"])
                default_level = level.replace("_", " ").title()
                if (
                    special_level_labels is not None
                    and moderator in special_level_labels
                ):
                    level_name = special_level_labels[moderator].get(
                        level,
                        default_level,
                    )
                else:
                    level_name = default_level
                forest_label = f"{level_name} ({item['effect'].capitalize()})"
            else:
                forest_label = f"{item['effect'].capitalize()}"
            forest_labels.append(forest_label)
            if forest_show_pp:
                ax.text(
                    1.01,
                    y_position,
                    f"P(Improve)={item['pp_improve']:.2f}",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    ha="left",
                )
            figures["statistics"][item["key"]] = {
                "mean": mean,
                "eti_low": item["eti_low"],
                "eti_high": item["eti_high"],
                "pp_improve": item["pp_improve"],
            }
        # adding bold moderator headings
        for moderator_name, header_y in group_headers:
            default_moderator_title = str(moderator_name).replace("_", " ").title()
            if special_mod_labels is not None and moderator_name in special_mod_labels:
                moderator_title = special_mod_labels.get(
                    moderator_name, default_moderator_title
                )
            else:
                moderator_title = default_moderator_title
            ax.text(
                -0.02,
                header_y,
                moderator_title,
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="right",
                fontweight="bold",
            )
        # adding separator lines between moderator groups
        for separator_y in group_separators:
            ax.axhline(
                separator_y,
                linewidth=1.5,
                color="black",
                alpha=0.75,
            )
        # applying row labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(forest_labels)
        ax.set_ylim(current_y - 0.10, -0.25)
        ax.set_xlabel(rf"$\mathbf{{{xlabel.replace(' ',r'\ ')}}}$" + f"{xlabel_imp}")
        ax.set_ylabel("")
        if title is not None:
            ax.set_title(
                rf"$\mathbf{{{title.replace(' ',r'\ ').capitalize()}}}:$ Moderator effects"
            )
        else:
            ax.set_title(
                "Moderator effects",
                fontweight="bold",
            )
        if forest_show_pp:
            fig.tight_layout(rect=(0, 0, 0.84, 1))
        else:
            fig.tight_layout()
        if save_dir is not None:
            save_path = os.path.join(
                save_dir,
                f"{file_prefix}_moderator_forest.{file_type}",
            )
            fig.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
            )
        figures["forest"] = fig
        if show:
            plt.show(block=False)
        else:
            plt.close(fig)
    return figures


#
