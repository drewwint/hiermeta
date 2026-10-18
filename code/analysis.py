# analysis.py: functions for hierarchical random-effect bayseian meta-analysis


# package imports--------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import os
import operator
import warnings

# Study-random-effect and pre/post-correlation hierachael bayesean model----------------------
# # Requires the research project's PyMC, PyTensor, ArviZ, pandas, numpy, and matplotlib
# # installation.
#
# import operator
# import os
# import warnings
# import arviz as az
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import pymc as pm
# import pytensor.tensor as pt


# Helpers__________________________________________
## blocks
def _build_sampling_blocks(
    df,
    prepost_rows,
    centered_means,
    contrast_pairs,
    se,
    study_idx,
    measure_col,
    measure_name_col,
    prepost_corr,
    outcome_corr,
):
    """
    Build study-specific observations and sampling covariance factors.

    Combine pre-post means and post-only contrasts into joint blocks
    while retaining covariance induced by correlated outcomes, repeated
    assessments, and shared observations.

    Inputs
    ------
    df : pandas.DataFrame
        Prepared analysis dataset containing "ID", "group", "time",
        measure_col, and measure_name_col. Row order must match the
        positional indices and row-level arrays supplied below.
    prepost_rows : numpy.ndarray of int
        Positions in df of observations entering the pre-post likelihood.
    centered_means : numpy.ndarray of float
        Observed means aligned with prepost_rows. These may be centered
        or uncentered, according to the calling function.
    contrast_pairs : pandas.DataFrame
        Post-only contrast definitions containing "active_row_idx",
        "sham_row_idx", "mean_active", and "mean_sham".
        Row indices identify positions in df. Observed contrasts are
        calculated by subtracting sham means from active means.
    se : numpy.ndarray of float
        Finite, positive sampling standard errors for all rows in df,
        in matching order.
    study_idx : numpy.ndarray of int
        Nonnegative, zero-based study/sample codes aligned with df.
        Rows from the same independent sample share a code.
    measure_col : str
        Column identifying the measure.
    measure_name_col : str
        Column identifying the scale/subscale. Combined with measure_col
        to identify outcomes.
    prepost_corr : float
        Fixed sampling correlation for the same outcome at different
        times within an arm. Must be finite and in [0, 1); validation
        is performed by bayes_meta_hre().
    outcome_corr : float
        Fixed sampling correlation for different outcomes at the same
        time within an arm. Must be finite and in [0, 1); validation
        is performed by bayes_meta_hre().

    Returns
    -------
    blocks : list of dict
        One dictionary per study with usable pre-post observations or
        post-only contrasts. Each contains:
        - "study_idx": integer study/sample code.
        - "raw_rows": pre-post row positions in df.
        - "contrast_rows": row positions in contrast_pairs.
        - "observed": pre-post means followed by observed contrasts.
        - "chol": lower-triangular Cholesky factor of the covariance
          matrix in the same order as "observed".

    Raises
    ------
    ValueError
        If a transformed study covariance cannot be Cholesky-factorized,
        including when supplied contrasts introduce linear redundancy.

    Notes
    -----
    Within an arm, observations of different outcomes at different times
    use prepost_corr * outcome_corr. Distinct rows sharing both outcome
    and time use prepost_corr + outcome_corr - their product.
    Each observation has unit correlation with itself.

    Sampling errors across arms are independent. Row-level covariance
    is transformed into the covariance of the combined means and
    contrasts, accounting for observations reused across contrasts.

    This function prepares numerical likelihood inputs; it does not
    create PyMC variables.
    """
    centered_by_row = np.full(len(df), np.nan)
    centered_by_row[prepost_rows] = centered_means
    row_outcomes = (
        df[measure_col].astype(str) + "::" + df[measure_name_col].astype(str)
    ).to_numpy()
    post_active = contrast_pairs["active_row_idx"].to_numpy(dtype=int)
    post_sham = contrast_pairs["sham_row_idx"].to_numpy(dtype=int)
    post_differences = contrast_pairs["mean_active"].to_numpy(
        dtype=float
    ) - contrast_pairs["mean_sham"].to_numpy(dtype=float)
    blocks = []
    for study in range(int(study_idx.max()) + 1):
        raw = prepost_rows[study_idx[prepost_rows] == study]
        contrasts = np.flatnonzero(study_idx[post_active] == study)
        if len(raw) + len(contrasts) == 0:
            continue
        primitive = np.unique(
            np.concatenate([raw, post_active[contrasts], post_sham[contrasts]])
        )
        loc = {int(row): j for j, row in enumerate(primitive)}
        times = df.iloc[primitive]["time"].to_numpy(dtype=int)
        groups = df.iloc[primitive]["group"].to_numpy()
        outcomes = row_outcomes[primitive]
        same_arm = groups[:, None] == groups[None, :]
        same_time = times[:, None] == times[None, :]
        same_outcome = outcomes[:, None] == outcomes[None, :]
        r_t, r_o = prepost_corr, outcome_corr
        residual_corr = (
            r_t * r_o
            + r_o * (1 - r_t) * same_time
            + r_t * (1 - r_o) * same_outcome
            + (1 - r_o) * (1 - r_t) * np.eye(len(primitive))
        ) * same_arm
        primitive_cov = np.outer(se[primitive], se[primitive]) * residual_corr
        transformation = np.zeros((len(raw) + len(contrasts), len(primitive)))
        for j, row in enumerate(raw):
            transformation[j, loc[int(row)]] = 1
        for j, contrast in enumerate(contrasts, start=len(raw)):
            transformation[j, loc[int(post_active[contrast])]] = 1
            transformation[j, loc[int(post_sham[contrast])]] = -1
        covariance = transformation @ primitive_cov @ transformation.T
        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"Study ID {df.iloc[primitive[0]]['ID']!r} has redundant or "
                "unidentifiable reported contrasts; inspect this study's outputs."
            ) from exc
        blocks.append(
            {
                "study_idx": study,
                "raw_rows": raw,
                "contrast_rows": contrasts,
                "observed": np.concatenate(
                    [centered_by_row[raw], post_differences[contrasts]]
                ),
                "chol": chol,
            }
        )
    return blocks


## joint liklihood
def _add_joint_likelihood(
    blocks,
    arm_baseline,
    arm_idx,
    post_idx,
    treatment_effect_row,
    postonly_direction,
    gamma_measure,
    postonly_measure_idx,
    postonly_contrast_row,
):
    """
    Add study-specific observed likelihoods to the active PyMC model.

    Inputs
    ------
    blocks : list of dict
        Sampling blocks returned by _build_sampling_blocks(), containing
        row positions, observed values, and covariance Cholesky factors.
    arm_baseline : tensor-like
        Baseline parameters indexed by study-arm-measure combination,
        in the same order used by arm_idx.
    arm_idx : numpy.ndarray of int
        Baseline-parameter indices aligned with all analysis rows.
    post_idx : numpy.ndarray
        Indicators aligned with analysis rows, with 1 for post
        observations and 0 for pre observations.
    treatment_effect_row : tensor-like
        Row-specific effects already converted to the observed scale
        using direction coding and measure-specific scaling.
        Applied to post observations in the pre-post likelihood.
    postonly_direction : numpy.ndarray
        Direction codes aligned with the post-only contrast table.
    gamma_measure : tensor-like
        Positive scaling parameters indexed by measure.
    postonly_measure_idx : numpy.ndarray of int
        Measure indices aligned with the post-only contrast table.
    postonly_contrast_row : tensor-like
        Latent active > sham effects aligned with the post-only contrast
        table, including study deviations and any moderator effects.
        Direction coding and measure scaling are applied within this
        function.

    Returns
    -------
    None
        Observed likelihood variables are registered in the active
        PyMC model.

    Notes
    -----
    Call this function within a PyMC model context.

    Pre-post expectations combine the corresponding baseline with the
    row-specific effect for post observations. Post-only expectations
    multiply the latent contrast by its direction code and measure
    scaling parameter.

    Each block creates an observed variable named "observed_study_{j}",
    where j is its zero-based position in blocks. A one-observation block
    uses pm.Normal; a larger block uses pm.MvNormal with the supplied
    Cholesky factor.

    Reported change-score likelihoods are added separately by
    bayes_meta_hre().
    """
    for j, block in enumerate(blocks):
        raw = block["raw_rows"]
        contrasts = block["contrast_rows"]
        mu_parts = []
        if len(raw):
            mu_parts.append(
                arm_baseline[arm_idx[raw]] + post_idx[raw] * treatment_effect_row[raw]
            )
        if len(contrasts):
            mu_parts.append(
                postonly_direction[contrasts]
                * gamma_measure[postonly_measure_idx[contrasts]]
                * postonly_contrast_row[contrasts]
            )
        mu = mu_parts[0] if len(mu_parts) == 1 else pt.concatenate(mu_parts)
        observed = block["observed"]
        if len(observed) == 1:
            pm.Normal(
                f"observed_study_{j}",
                mu=mu[0],
                sigma=block["chol"][0, 0],
                observed=observed[0],
            )
        else:
            pm.MvNormal(
                f"observed_study_{j}",
                mu=mu,
                chol=block["chol"],
                observed=observed,
            )


# Function ______________________________________________
## bayes_meta_hre (bayesian meta-analysis hierarcheal random effect)


def bayes_meta_hre(
    df,
    moderator_names,
    continuous_expected_mod,
    dichotomous_mod,
    vnames=["theta_sham", "theta_active", "theta_contrast"],
    tacc=0.98,
    gamma_sigma=0.5,
    prepost_corr=0.5,
    outcome_corr=0.3,
    study_tau_sigma=1.0,
    baseline_cent=True,
    plt_show=False,
    qc_plt_save_dir=None,
    r_seed=1979,
    verbose=True,
    improvement_direction=">",
    measure_col="measure_name",
    measure_name_col="measure_scale_name",
    direction_col="domain_direction",
    moderation=True,
):
    """
    Fit a hierarchical Bayesian meta-analysis with study-level random effects.

    Estimate sham, active, and active > sham effects using pre-post
    summaries, reported change scores, and available post-only contrasts.
    Include partially pooled baselines, positive measure-specific scaling,
    and study-level variation in the active > sham effect.

    Optionally fit a separate model for each eligible moderator.

    Inputs
    ------
    df : pandas.DataFrame
        Summary-level dataset containing:
        - "ID": nonmissing identifier for each independent study/sample.
          Outcomes from the same sample must share an identifier.
        - "group": treatment assignment, coded 0 for sham and 1 for active.
        - "time": assessment type, coded 0 for pre, 1 for post, and
          3 for a reported post-minus-pre change score.
        - "mean": observed mean or mean change.
        - "sd": corresponding standard deviation, not a standard error
          or variance. Change-score rows require the SD of change scores.
        - "n": sample size corresponding to each reported summary.
        - The columns specified by measure_col, measure_name_col,
          and direction_col.
        Grouping columns must identify the study, arm, measure, and
        scale/subscale associated with each observation. Retained rows
        must yield finite, positive standard errors calculated as
        sd / sqrt(n).
        The "measure_category" column is additionally required when
        saving QC plots. The input DataFrame is not modified.
    moderator_names : sequence of str or None
        Moderator columns to evaluate. Each eligible, recognized moderator
        is fitted in a separate model. Ignored when moderation is False.
    continuous_expected_mod : sequence of str or None
        Moderators eligible for continuous treatment. Those with more
        than two distinct values are standardized using the mean and
        sample SD across retained rows. Those with exactly two distinct
        values are treated as categorical.
        A moderator also listed in dichotomous_mod is treated as
        categorical regardless of its number of distinct values.
    dichotomous_mod : sequence of str or None
        Moderators to treat as categorical. Despite the name, variables
        with more than two categories are supported. Category effects
        use sum-to-zero deviations without a reference category.
    vnames : sequence of str, optional
        Scalar base-model posterior variables to plot, summarize, and
        evaluate for directional probability. "tau_study" is always
        included in the summary.
        Default is ["theta_sham", "theta_active", "theta_contrast"].
    tacc : float, optional
        Target acceptance probability passed to pm.sample().
        Default is 0.98.
    gamma_sigma : float, optional
        Standard deviation of the zero-centered Normal prior on
        "log_gamma_raw". Log scaling parameters are centered across
        measures before exponentiation, constraining the geometric
        mean of "gamma_measure" to 1. Default is 0.5.
    prepost_corr : float, optional
        Fixed sampling-error correlation between observations of the
        same outcome at different times within an arm. Must be finite
        and in [0, 1). Default is 0.5.
    outcome_corr : float, optional
        Fixed sampling-error correlation between different outcomes
        measured at the same time within an arm. Must be finite and
        in [0, 1). Different outcomes at different times use the product
        of outcome_corr and prepost_corr. Default is 0.3.
    study_tau_sigma : float, optional
        Scale of the HalfNormal prior on "tau_study", the standard
        deviation of study-level active > sham deviations.
        Must be finite and positive. Default is 1.0.
    baseline_cent : bool, optional
        Whether to subtract each measure's unweighted mean of baseline
        means from its pre-post observations. Centering is performed
        within measure_col and does not standardize observations.
        Default is True.
    plt_show : bool, optional
        Whether to call plt.show(block=False) for the base-model QC plot.
        The current implementation selects the "agg" backend before
        this call. Moderator QC plots are not displayed by this option.
        Default is False.
    qc_plt_save_dir : str or path-like or None, optional
        Directory for QC plots, saved as TIFF files at 300 dpi.
        Created if necessary. If None, plots are not saved.
        QC plots are generated and closed regardless of this setting.
        Default is None.
    r_seed : int, optional
        Random seed passed to sampling for the base model and each
        moderator model. Default is 1979.
    verbose : bool, optional
        Whether to return an additional information dictionary.
        This does not suppress progress messages or sampler output.
        Default is True.
    improvement_direction : {"<", ">"}, optional
        Direction used to calculate "pp_improve" from posterior draws.
        "<" calculates the probability below zero; ">" calculates the
        probability above zero. This affects summary interpretation,
        not model fitting or direction coding. Default is ">".
    measure_col : str, optional
        Column defining measure-specific scaling, baseline distributions,
        and centering groups. Default is "measure_name".
    measure_name_col : str, optional
        Column identifying the scale/subscale within a measure.
        Combined with measure_col to identify outcomes for design
        classification, post-only pairing, and sampling correlations.
        Default is "measure_scale_name".
    direction_col : str, optional
        Column containing +1 or -1 to align raw-score effects with the
        modeled domain. Use +1 when increasing raw scores indicate
        increasing domain values and -1 when they indicate decreasing
        domain values. Coding should be consistent within each outcome.
        Default is "domain_direction".
    moderation : bool, optional
        Whether to screen and fit the requested moderators.
        Default is True.

    Returns
    -------
    base_summary : pandas.DataFrame
        Base-model summary for vnames and "tau_study", with "pp_improve"
        added for variables in vnames. Summary statistics and interval
        settings follow az.summary().
    moderator_summaries : dict of pandas.DataFrame
        Returned only when moderation is True. Maps fitted moderator
        names to summaries containing category-specific effects or
        continuous slopes for sham, active, and active > sham, plus
        "tau_study". Includes "pp_improve" for the moderator effects
        and a "moderator" column.
        Empty when no moderator models are fitted.
    info : dict
        Returned only when verbose is True. Contains:
        - "final_df": filtered dataset with a reset integer index.
          Includes unpaired post-only rows that do not enter the
          likelihood. Reported means remain uncentered in this table.
        - "base_summary": base-model summary.
        - "trace_base": base-model posterior sampling output.
        - "improvement_direction": supplied interpretation direction.
        - "prepost_corr": supplied temporal sampling correlation.
        - "outcome_corr": supplied outcome sampling correlation.
        - "study_tau_sigma": supplied study-heterogeneity prior scale.
        - "unpaired_postonly_row_indices": positions in final_df of
          post-only rows excluded from the likelihood.
        - "sampling_block_sizes": observation counts in the joint
          study blocks, excluding reported change-score rows.
        - "n_postonly_contrasts": number of constructed post-only
          contrasts, stored as an integer.

        When moderation is True, also contains:
        - "moderator_summaries": fitted moderator summaries.
        - "valid_mod": moderators passing initial eligibility checks.
          Moderators without a recognized type can still be skipped.
        - "invalid_mod": dictionaries containing moderator names and
          eligibility flags: "exists", "var_overall", "gp_levels",
          "sd_n0", "complete", and "mod_type".
        - "trace_mod": posterior sampling outputs keyed by fitted
          moderator name.
        Return combinations are:
          (base_summary, moderator_summaries, info) when both options are True;
          (base_summary, moderator_summaries) when only moderation is True;
          (base_summary, info) when only verbose is True; and base_summary alone when both are False.

    Raises
    ------
    ValueError
        If correlation or study-prior settings are invalid, time-code
        validation fails, improvement_direction is invalid, a retained
        study identifier is missing, baseline scales or standard errors
        are invalid, or a joint sampling covariance cannot be
        Cholesky-factorized.
    KeyError
        If a required data column is missing.

    Warns
    -----
    RuntimeWarning
        If post-only rows lack a matching opposite-arm outcome and
        therefore cannot enter a treatment contrast.

    Notes
    -----
    Pre-post observations require both times within the same study,
    arm, measure, and scale/subscale. Reported change scores are retained
    when that group lacks a complete pre-post pair. Post observations
    without a baseline are eligible for post-only contrasts.

    Post-only contrasts match opposite-arm post observations within
    the same study and outcome. Shared observations are accounted for
    in the joint covariance. Sampling errors are assumed independent
    across treatment arms and across independent study identifiers.

    Reported change scores use conditionally independent Normal
    likelihoods. Their sampling errors are not included in the joint
    pre-post and post-only covariance blocks.

    Baseline prior scales use the median baseline SD for each measure,
    falling back to the median SD across its retained observations.
    Baseline parameters are indexed by study, arm, and measure;
    scale/subscale names do not create separate baseline parameters.

    Moderator screening requires complete values, at least two distinct
    values overall and within each treatment group, and positive sample
    SD for numeric variables.

    Sampling requests four chains, 2,000 tuning iterations, and 2,000
    posterior draws per chain for every fitted model. The function does
    not explicitly request an equal-tailed interval or an interval mass.
    """
    if not np.isfinite(prepost_corr) or not 0 <= prepost_corr < 1:
        raise ValueError("prepost_corr must be finite and in [0, 1)")
    if not np.isfinite(outcome_corr) or not 0 <= outcome_corr < 1:
        raise ValueError("outcome_corr must be finite and in [0, 1)")
    if not np.isfinite(study_tau_sigma) or study_tau_sigma <= 0:
        raise ValueError("study_tau_sigma must be finite and positive")
    # copying dataframe
    df = df.copy()
    # identifying pre-post, post-only, and difference-score observations
    time_values = set(df["time"].dropna().astype(int).unique())
    if not time_values.issubset({0, 1, 3}):
        raise ValueError("time must be coded 0=pre, 1=post, or 3=difference")
    design_keys = ["ID", "group", measure_col, measure_name_col]
    has_pre = df.groupby(design_keys)["time"].transform(lambda x: (x == 0).any())
    has_post = df.groupby(design_keys)["time"].transform(lambda x: (x == 1).any())
    is_prepost = df["time"].isin([0, 1]) & has_pre & has_post
    is_diff = (df["time"] == 3) & ~(has_pre & has_post)
    is_postonly = (df["time"] == 1) & (~has_pre) & has_post
    df["_obs_type"] = np.select(
        [is_diff, is_prepost, is_postonly],
        ["difference", "prepost", "postonly"],
        default="drop",
    )
    df = df.loc[df["_obs_type"] != "drop"].reset_index(drop=True).copy()
    if any(df["_obs_type"] == "postonly") and any(df["_obs_type"] == "prepost"):
        fn_post = "primary_"
    else:
        fn_post = "sensitivity_"
    obs_type = df.pop("_obs_type").to_numpy()
    # establishing measure direction
    direction = df[direction_col].to_numpy()
    # establishing posterior improvement direction
    if improvement_direction == ">":
        posterior_dir = operator.gt
    elif improvement_direction == "<":
        posterior_dir = operator.lt
    else:
        raise ValueError("improvement_direction must be '>' or '<'")
    # checking which moderators can be used
    moderator_yes = []
    moderator_no = []
    if moderation:
        for mn in moderator_names:
            ## exists
            exists = mn in df.columns
            if not exists:
                moderator_no.append(
                    {
                        "moderator": mn,
                        "exists": False,
                        "var_overall": False,
                        "gp_levels": False,
                        "sd_n0": False,
                        "complete": False,
                        "mod_type": None,
                    }
                )
                continue
            ## nonmissing varaition across studies
            moderator_nonmissing = df[mn].dropna()
            var_overall = bool(moderator_nonmissing.nunique() >= 2)
            ## represented in at least 2 levels by group
            gp_levels = bool(
                df.dropna(subset=[mn])
                .groupby("group")[mn]
                .nunique()
                .reindex([0, 1], fill_value=0)
                .ge(2)
                .all()
            )
            ## complete varaible
            complete = bool(df[mn].notna().all())
            ## sd non 0
            if pd.api.types.is_numeric_dtype(moderator_nonmissing):
                sd_n0 = bool(moderator_nonmissing.std() > 0)
                mod_type = "continuous"
            else:
                sd_n0 = True
                mod_type = "categorical"
            if exists and var_overall and gp_levels and sd_n0 and complete:
                moderator_yes.append(mn)
            else:
                moderator_no.append(
                    {
                        "moderator": mn,
                        "exists": exists,
                        "var_overall": var_overall,
                        "gp_levels": gp_levels,
                        "sd_n0": sd_n0,
                        "complete": complete,
                        "mod_type": mod_type,
                    }
                )
        print(f"\n# of valid moderators: {len(moderator_yes)}\n")
    # creating measure index
    measure_idx, measure_name = pd.factorize(df[measure_col])
    # ID must uniquely identify an independent study/sample across its outcomes.
    if df["ID"].isna().any():
        raise ValueError("Every row must have a nonmissing independent study/sample ID")
    study_idx, study_name = pd.factorize(df["ID"], sort=False)
    # creating unique arm index
    arm_label = (
        df["ID"].astype(str)
        + "__"
        + df["group"].astype(str)
        + "__"
        + df[measure_col].astype(str)
    )
    arm_idx, arm_name = pd.factorize(arm_label)
    # identifying measure associated with each arm
    arm_measure_idx = (
        pd.DataFrame(
            {
                "arm_idx": arm_idx,
                "measure_idx": measure_idx,
            }
        )
        .drop_duplicates("arm_idx")
        .sort_values("arm_idx")["measure_idx"]
        .to_numpy(dtype=int)
    )
    # creating active index
    active_idx = np.where(df.group == 1, 1, 0)
    post_idx = np.where(df.time == 1, 1, 0)
    # identifying observation types
    prepost_row_idx = np.flatnonzero(obs_type == "prepost")
    diff_row_idx = np.flatnonzero(obs_type == "difference")
    postonly_row_idx = np.flatnonzero(obs_type == "postonly")
    # extracting actual baseline data
    baseline_df = df.loc[df.time == 0]
    # centering pre-post means by measure baseline
    if baseline_cent:
        baseline_center = (
            baseline_df.groupby(measure_col)["mean"]
            .mean()
            .reindex(measure_name)
            .fillna(0)
            .to_numpy()
        )
        means_prepost = (
            df.iloc[prepost_row_idx]["mean"].to_numpy(dtype=float)
            - baseline_center[measure_idx[prepost_row_idx]]
        )
    else:
        means_prepost = df.iloc[prepost_row_idx]["mean"].to_numpy(dtype=float)
    # extracting measure-specific baseline scale
    baseline_scale_pre = (
        baseline_df.groupby(measure_col)["sd"].median().reindex(measure_name).to_numpy()
    )
    baseline_scale_all = (
        df.groupby(measure_col)["sd"].median().reindex(measure_name).to_numpy()
    )
    baseline_scale = np.where(
        np.isfinite(baseline_scale_pre) & (baseline_scale_pre > 0),
        baseline_scale_pre,
        baseline_scale_all,
    )
    # checking baseline scaling
    if bool((~np.isfinite(baseline_scale) | (baseline_scale <= 0)).any()):
        bad_scale = np.asarray(measure_name)[
            ~np.isfinite(baseline_scale) | (baseline_scale <= 0)
        ]
        raise ValueError(f"Invalid baseline scale for measures: {bad_scale.tolist()}")
    # calculating row-specific standard errors
    se = df["sd"].to_numpy(dtype=float) / np.sqrt(df["n"].to_numpy(dtype=float))
    if not np.isfinite(se).all() or (se <= 0).any():
        raise ValueError("Every retained row needs a finite, positive SD and n")
    means_diff = df.iloc[diff_row_idx]["mean"].to_numpy(dtype=float)
    se_diff = se[diff_row_idx]
    # Construct independent contrasts from every usable post-only arm row.
    # - With no baseline in either arm, m + n - 1 contrasts retain all relative
    # - information from m active and n sham post means, without matching rows
    # - by order. A shared reference mean is accounted for in the joint
    # - covariance below. Pre/post rows already enter the raw likelihood.
    postonly_df = df.loc[
        df.time.eq(1), ["ID", measure_col, measure_name_col, "group"]
    ].copy()
    postonly_df["obs_type"] = obs_type[postonly_df.index.to_numpy(dtype=int)]
    pair_keys = ["ID", measure_col, measure_name_col]
    contrast_rows = []
    for _, outcome_rows in postonly_df.groupby(pair_keys, dropna=False, sort=False):
        sham = outcome_rows.loc[outcome_rows.group == 0]
        active = outcome_rows.loc[outcome_rows.group == 1]
        if sham.empty or active.empty:
            continue
        raw_sham = sham.loc[sham.obs_type == "prepost"].index.to_numpy(dtype=int)
        raw_active = active.loc[active.obs_type == "prepost"].index.to_numpy(dtype=int)
        missing_sham = sham.loc[sham.obs_type == "postonly"].index.to_numpy(dtype=int)
        missing_active = active.loc[active.obs_type == "postonly"].index.to_numpy(
            dtype=int
        )
        if len(raw_sham) and len(raw_active):
            continue
        if len(raw_sham):
            contrast_rows.extend((int(row), int(raw_sham[0])) for row in missing_active)
        elif len(raw_active):
            contrast_rows.extend((int(raw_active[0]), int(row)) for row in missing_sham)
        else:
            # All post means have no baseline. These differences span every
            # mean except the unidentifiable common post-treatment intercept.
            sham_anchor = int(missing_sham[0])
            active_anchor = int(missing_active[0])
            contrast_rows.extend((int(row), sham_anchor) for row in missing_active)
            contrast_rows.extend((active_anchor, int(row)) for row in missing_sham[1:])
    postonly_pairs = pd.DataFrame(
        contrast_rows, columns=["active_row_idx", "sham_row_idx"]
    )
    if postonly_pairs.empty:
        postonly_pairs = postonly_pairs.astype(int)
    postonly_pairs["mean_active"] = df.iloc[
        postonly_pairs["active_row_idx"].to_numpy(dtype=int)
    ]["mean"].to_numpy(dtype=float)
    postonly_pairs["mean_sham"] = df.iloc[
        postonly_pairs["sham_row_idx"].to_numpy(dtype=int)
    ]["mean"].to_numpy(dtype=float)
    paired_postonly = set(postonly_pairs["active_row_idx"]) | set(
        postonly_pairs["sham_row_idx"]
    )
    unpaired_postonly = sorted(set(postonly_row_idx) - paired_postonly)
    if unpaired_postonly:
        warnings.warn(
            f"{len(unpaired_postonly)} post-only arm rows lack a matching "
            "active/sham outcome; these rows cannot identify a treatment "
            "contrast and were not used in the likelihood.",
            RuntimeWarning,
            stacklevel=2,
        )
    postonly_sham_row_idx = postonly_pairs["sham_row_idx"].to_numpy(dtype=int)
    postonly_active_row_idx = postonly_pairs["active_row_idx"].to_numpy(dtype=int)
    postonly_measure_idx = measure_idx[postonly_active_row_idx]
    postonly_study_idx = study_idx[postonly_active_row_idx]
    postonly_direction = direction[postonly_active_row_idx]
    sampling_blocks = _build_sampling_blocks(
        df=df,
        prepost_rows=prepost_row_idx,
        centered_means=means_prepost,
        contrast_pairs=postonly_pairs,
        se=se,
        study_idx=study_idx,
        measure_col=measure_col,
        measure_name_col=measure_name_col,
        prepost_corr=prepost_corr,
        outcome_corr=outcome_corr,
    )
    # establishing model coordinates
    coords = {
        "n": np.arange(len(df)),
        "n_diff": np.arange(len(diff_row_idx)),
        "measure": measure_name,
        "arm": arm_name,
        "study": study_name,
    }
    # running base model
    with pm.Model(coords=coords) as base_model:
        # latent sham effect
        theta_sham = pm.Normal(
            "theta_sham",
            mu=0,
            sigma=1,
        )
        # latent active effect
        theta_active = pm.Normal(
            "theta_active",
            mu=0,
            sigma=1,
        )
        # latent active minus sham contrast
        theta_contrast = pm.Deterministic(
            "theta_contrast",
            theta_active - theta_sham,
        )
        # Non-centered heterogeneity in the *active-minus-sham* effect.
        tau_study = pm.HalfNormal("tau_study", sigma=study_tau_sigma)
        study_offset = pm.Normal("study_offset", 0, 1, dims="study")
        study_contrast_deviation = pm.Deterministic(
            "study_contrast_deviation",
            tau_study * study_offset,
            dims="study",
        )
        pm.Deterministic(
            "theta_contrast_study",
            theta_contrast + study_contrast_deviation,
            dims="study",
        )
        # measure-specific scaling
        log_gamma_raw = pm.Normal(
            "log_gamma_raw",
            mu=0,
            sigma=gamma_sigma,
            dims="measure",
        )
        log_gamma = pm.Deterministic(
            "log_gamma",
            log_gamma_raw - pt.mean(log_gamma_raw),
            dims="measure",
        )
        gamma_measure = pm.Deterministic(
            "gamma_measure",
            pt.exp(log_gamma),
            dims="measure",
        )
        # measure-specific baseline means
        baseline_mean_measure = pm.Normal(
            "baseline_mean_measure",
            mu=0,
            sigma=2 * baseline_scale,
            dims="measure",
        )
        # between-arm baseline variability
        baseline_sd_measure = pm.HalfNormal(
            "baseline_sd_measure",
            sigma=baseline_scale,
            dims="measure",
        )
        # standardized arm-specific baseline deviations
        arm_baseline_offset = pm.Normal(
            "arm_baseline_offset",
            mu=0,
            sigma=1,
            dims="arm",
        )
        # partially pooled arm baselines
        arm_baseline = pm.Deterministic(
            "arm_baseline",
            baseline_mean_measure[arm_measure_idx]
            + baseline_sd_measure[arm_measure_idx] * arm_baseline_offset,
            dims="arm",
        )
        # assigning sham or active latent effect by row
        theta_row = (1 - active_idx) * theta_sham + active_idx * (
            theta_active + study_contrast_deviation[study_idx]
        )
        # converting latent effect into each measure's scale
        treatment_effect_row = direction * gamma_measure[measure_idx] * theta_row
        # Joint pre/post and post-only likelihood, grouped by study. This
        # accounts for assumed residual correlations even with missing rows.
        postonly_contrast_row = (
            theta_contrast + study_contrast_deviation[postonly_study_idx]
        )
        _add_joint_likelihood(
            sampling_blocks,
            arm_baseline,
            arm_idx,
            post_idx,
            treatment_effect_row,
            postonly_direction,
            gamma_measure,
            postonly_measure_idx,
            postonly_contrast_row,
        )
        # difference-score likelihood
        if len(diff_row_idx) > 0:
            pm.Normal(
                "observed_difference",
                mu=treatment_effect_row[diff_row_idx],
                sigma=se_diff,
                observed=means_diff,
                dims="n_diff",
            )
        # sampling
        base_trace = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=tacc,
            random_seed=r_seed,
        )
    # plotting base posterior
    az.plot_trace_dist(
        base_trace,
        var_names=vnames,
    )
    # saving base posterior plot
    if qc_plt_save_dir is not None:
        save_dir = os.path.abspath(qc_plt_save_dir)
        if os.path.exists(save_dir):
            plt.savefig(
                os.path.join(
                    save_dir, f"base_{str(df.measure_category[0])}_{fn_post}qc.tiff"
                ),
                dpi=300,
            )
        else:
            os.makedirs(save_dir, exist_ok=True)
            print(f"created qc plot directory: {save_dir}")
            plt.savefig(
                os.path.join(
                    save_dir, f"base_{str(df.measure_category[0])}_{fn_post}qc.tiff"
                ),
                dpi=300,
            )
    # showing or closing plot
    if plt_show:
        matplotlib.use("agg")
        plt.show(block=False)
    plt.close()
    # summarizing base model
    base_summary = az.summary(
        base_trace,
        var_names=list(dict.fromkeys([*vnames, "tau_study"])),
    )
    # calculating posterior probability of improvement
    for nn in vnames:
        base_summary.loc[nn, "pp_improve"] = (
            posterior_dir(
                base_trace.posterior[nn],
                0,
            )
            .mean(dim=("chain", "draw"))
            .item()
        )
    # creating moderator output containers
    moderator_traces = {}
    moderator_summaries = {}
    dichotomous_coding = {}
    # running moderator models
    if moderation and len(moderator_yes) > 0:
        mn_n = 1
        for mn in moderator_yes:
            print(f"\n Estimating Moderator {mn_n} / {len(moderator_yes)}: {mn}\n")
            mn_n += 1
            # extracting moderator values
            moderator_values = df[mn]
            n_unique = moderator_values.nunique()
            # determining categorical versus continuous moderator treatment
            is_categorical = mn in dichotomous_mod or (
                mn in continuous_expected_mod and n_unique == 2
            )
            is_continuous = mn in continuous_expected_mod and n_unique > 2
            # preparing categorical moderator
            if is_categorical:
                categorical_values = moderator_values.astype("category")
                category_levels = categorical_values.cat.categories.tolist()
                category_idx = categorical_values.cat.codes.to_numpy(dtype=int)
                moderator_terms = category_levels
                dichotomous_coding[mn] = {
                    "coding": "sum_to_zero",
                    "category_levels": category_levels,
                }
            # preparing continuous moderator
            elif is_continuous:
                moderator_mean = moderator_values.mean()
                moderator_sd = moderator_values.std()
                X_mod = (
                    (moderator_values.to_numpy(dtype=float) - moderator_mean)
                    / moderator_sd
                )[:, None]
                moderator_terms = [f"{mn}_z"]
            # skipping unrecognized moderator type
            else:
                continue
            # establishing moderator model coordinates
            moderator_coords = {
                **coords,
                "moderator_term": moderator_terms,
            }
            # running moderator model
            with pm.Model(coords=moderator_coords) as mod_model:
                # latent sham effect
                theta_sham = pm.Normal(
                    "theta_sham",
                    mu=0,
                    sigma=1,
                )
                # latent active effect
                theta_active = pm.Normal(
                    "theta_active",
                    mu=0,
                    sigma=1,
                )
                # latent active minus sham contrast
                theta_contrast = pm.Deterministic(
                    "theta_contrast",
                    theta_active - theta_sham,
                )
                tau_study = pm.HalfNormal("tau_study", sigma=study_tau_sigma)
                study_offset = pm.Normal("study_offset", 0, 1, dims="study")
                study_contrast_deviation = pm.Deterministic(
                    "study_contrast_deviation",
                    tau_study * study_offset,
                    dims="study",
                )
                # categorical moderator model
                if is_categorical:
                    beta_moderator_sham = pm.ZeroSumNormal(
                        "beta_moderator_sham",
                        sigma=1,
                        dims="moderator_term",
                    )
                    beta_moderator_active = pm.ZeroSumNormal(
                        "beta_moderator_active",
                        sigma=1,
                        dims="moderator_term",
                    )
                    beta_moderator_contrast = pm.Deterministic(
                        "beta_moderator_contrast",
                        beta_moderator_active - beta_moderator_sham,
                        dims="moderator_term",
                    )
                    theta_sham_category = pm.Deterministic(
                        "theta_sham_category",
                        theta_sham + beta_moderator_sham,
                        dims="moderator_term",
                    )
                    theta_active_category = pm.Deterministic(
                        "theta_active_category",
                        theta_active + beta_moderator_active,
                        dims="moderator_term",
                    )
                    theta_contrast_category = pm.Deterministic(
                        "theta_contrast_category",
                        theta_active_category - theta_sham_category,
                        dims="moderator_term",
                    )
                    theta_sham_row = theta_sham_category[category_idx]
                    theta_active_row = theta_active_category[category_idx]
                # continuous moderator model
                else:
                    X_moderator = pm.Data(
                        "X_moderator",
                        X_mod,
                        dims=("n", "moderator_term"),
                    )
                    beta_moderator_sham = pm.Normal(
                        "beta_moderator_sham",
                        mu=0,
                        sigma=1,
                        dims="moderator_term",
                    )
                    beta_moderator_active = pm.Normal(
                        "beta_moderator_active",
                        mu=0,
                        sigma=1,
                        dims="moderator_term",
                    )
                    beta_moderator_contrast = pm.Deterministic(
                        "beta_moderator_contrast",
                        beta_moderator_active - beta_moderator_sham,
                        dims="moderator_term",
                    )
                    moderator_shift_sham = pm.math.dot(
                        X_moderator,
                        beta_moderator_sham,
                    )
                    moderator_shift_active = pm.math.dot(
                        X_moderator,
                        beta_moderator_active,
                    )
                    theta_sham_row = theta_sham + moderator_shift_sham
                    theta_active_row = theta_active + moderator_shift_active
                # A study-specific contrast shift applies to the active arm.
                theta_active_row = (
                    theta_active_row + study_contrast_deviation[study_idx]
                )
                # measure-specific scaling
                log_gamma_raw = pm.Normal(
                    "log_gamma_raw",
                    mu=0,
                    sigma=gamma_sigma,
                    dims="measure",
                )
                log_gamma = pm.Deterministic(
                    "log_gamma",
                    log_gamma_raw - pt.mean(log_gamma_raw),
                    dims="measure",
                )
                gamma_measure = pm.Deterministic(
                    "gamma_measure",
                    pt.exp(log_gamma),
                    dims="measure",
                )
                # measure-specific baseline means
                baseline_mean_measure = pm.Normal(
                    "baseline_mean_measure",
                    mu=0,
                    sigma=2 * baseline_scale,
                    dims="measure",
                )
                # between-arm baseline variability
                baseline_sd_measure = pm.HalfNormal(
                    "baseline_sd_measure",
                    sigma=baseline_scale,
                    dims="measure",
                )
                # standardized arm-specific baseline deviations
                arm_baseline_offset = pm.Normal(
                    "arm_baseline_offset",
                    mu=0,
                    sigma=1,
                    dims="arm",
                )
                # partially pooled arm baselines
                arm_baseline = pm.Deterministic(
                    "arm_baseline",
                    baseline_mean_measure[arm_measure_idx]
                    + baseline_sd_measure[arm_measure_idx] * arm_baseline_offset,
                    dims="arm",
                )
                # assigning moderator-adjusted sham or active effect by row
                theta_row = (
                    1 - active_idx
                ) * theta_sham_row + active_idx * theta_active_row
                # converting latent effect into each measure's scale
                treatment_effect_row = (
                    direction * gamma_measure[measure_idx] * theta_row
                )
                # Same correlated likelihood as the base model.
                theta_postonly_contrast = (
                    theta_active_row[postonly_active_row_idx]
                    - theta_sham_row[postonly_sham_row_idx]
                )
                _add_joint_likelihood(
                    sampling_blocks,
                    arm_baseline,
                    arm_idx,
                    post_idx,
                    treatment_effect_row,
                    postonly_direction,
                    gamma_measure,
                    postonly_measure_idx,
                    theta_postonly_contrast,
                )
                # difference-score likelihood
                if len(diff_row_idx) > 0:
                    pm.Normal(
                        "observed_difference",
                        mu=treatment_effect_row[diff_row_idx],
                        sigma=se_diff,
                        observed=means_diff,
                        dims="n_diff",
                    )
                # sampling
                mod_trace = pm.sample(
                    draws=2000,
                    tune=2000,
                    chains=4,
                    target_accept=tacc,
                    random_seed=r_seed,
                )
            # storing moderator trace
            moderator_traces[mn] = mod_trace
            # selecting categorical effects for summary
            if is_categorical:
                mod_vnames = [
                    "theta_sham_category",
                    "theta_active_category",
                    "theta_contrast_category",
                ]
            # selecting continuous slopes for summary
            else:
                mod_vnames = [
                    "beta_moderator_sham",
                    "beta_moderator_active",
                    "beta_moderator_contrast",
                ]
            # plotting base posterior
            az.plot_trace_dist(
                mod_trace,
                var_names=mod_vnames,
            )
            # saving base posterior plot
            if qc_plt_save_dir is not None:
                save_dir = os.path.abspath(qc_plt_save_dir)
                if os.path.exists(save_dir):
                    plt.savefig(
                        os.path.join(save_dir, f"moderator-{str(mn)}_qc.tiff"), dpi=300
                    )
            plt.close()
            # summarizing moderator model
            ms = az.summary(
                mod_trace,
                var_names=mod_vnames + ["tau_study"],
            )
            # calculating posterior probabilities
            for name in mod_vnames:
                probabilities = (
                    posterior_dir(
                        mod_trace.posterior[name],
                        0,
                    )
                    .mean(dim=("chain", "draw"))
                    .to_numpy()
                    .reshape(-1)
                )
                rows = [idx for idx in ms.index if idx.startswith(name)]
                ms.loc[rows, "pp_improve"] = probabilities
            # adding moderator name
            ms["moderator"] = mn
            # storing moderator summary
            moderator_summaries[mn] = ms
    # returning verbose outputs
    if verbose:
        if moderation:
            info = {
                "final_df": df,
                "base_summary": base_summary,
                "moderator_summaries": moderator_summaries,
                "valid_mod": moderator_yes,
                "invalid_mod": moderator_no,
                "trace_base": base_trace,
                "trace_mod": moderator_traces,
                "improvement_direction": improvement_direction,
                "prepost_corr": prepost_corr,
                "outcome_corr": outcome_corr,
                "study_tau_sigma": study_tau_sigma,
                "unpaired_postonly_row_indices": unpaired_postonly,
                "sampling_block_sizes": [len(b["observed"]) for b in sampling_blocks],
                "n_postonly_contrasts": len(postonly_pairs),
            }
            return base_summary, moderator_summaries, info
        else:
            info = {
                "final_df": df,
                "base_summary": base_summary,
                "trace_base": base_trace,
                "improvement_direction": improvement_direction,
                "prepost_corr": prepost_corr,
                "outcome_corr": outcome_corr,
                "study_tau_sigma": study_tau_sigma,
                "unpaired_postonly_row_indices": unpaired_postonly,
                "sampling_block_sizes": [len(b["observed"]) for b in sampling_blocks],
                "n_postonly_contrasts": len(postonly_pairs),
            }
            return base_summary, info
    # returning concise outputs
    else:
        if moderation:
            return base_summary, moderator_summaries
        else:
            return base_summary
