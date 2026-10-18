# Moderation.py: functions for evaluating meta-analytic moderator analyses

# Package imports--------------------------------------------------------------
from itertools import combinations
import numpy as np
import pandas as pd
import os

# moderation results: evaluate, indicate, and save ----------------------------
# import os
# import pandas as pd


def mod_eval(
    mod_dict_primary,
    mod_dict_sensitivity,
    save_dir,
    save_folder="moderator_eval",
    show="none",
    p_primary=0.95,
    p_sensitivity=0.80,
    summary_cols=[
        "mean",
        "sd",
        "eti89_lb",
        "eti89_ub",
        "pp_improve",
        "moderator",
    ],
    robust_est_col=True,
):
    """
    Evaluate moderator robustness across primary and sensitivity analyses.

    Retain a moderator when at least one shared contrast estimate has the
    same sign in both analyses and its probability of improvement exceeds
    both user-specified thresholds. Retained summaries include all shared
    estimates with matching signs.

    Inputs
    ------
    mod_dict_primary : dict of pandas.DataFrame
      Dictionary mapping moderator names to primary analysis summaries.
      Each summary must contain "mean", "pp_improve", and the requested
      summary_cols. For moderators with sensitivity results, the index
      must be unique and include "tau_study".
      Contrast estimates are identified by index labels containing
      "contrast".
    mod_dict_sensitivity : dict of pandas.DataFrame
      Dictionary mapping moderator names to sensitivity summaries.
      Each matched summary must have a unique index and contain "mean"
      and "pp_improve". Estimates are aligned using shared index labels.
      Moderators absent from this dictionary are saved with a
      "no_sensitivity" suffix and excluded from the returned results.
    save_dir : str or path-like
      Parent directory in which to save the results.
    save_folder : str, optional
      Subdirectory created within save_dir for the output tables.
      Default is "moderator_eval".
    show : {"all", "important", "none"}, optional
      Tables to print. "all" displays complete primary summaries,
      including robustness flags when robust_est_col is True.
      "important" displays filtered summaries for retained moderators.
      "none" suppresses table displays. Saved-status messages are
      printed independently of this setting. Default is "none".
    p_primary : float
      Primary probability-of-improvement threshold. An estimate must
      have "pp_improve" strictly greater than this value to qualify
      as robust. Default is 0.95.
    p_sensitivity : float
      Sensitivity probability-of-improvement threshold. An estimate
      must have "pp_improve" strictly greater than this value, as well
      as meeting the primary threshold and matching signs, to qualify
      as robust. Default is 0.80.
    summary_cols : list of str
      Columns selected for returned summaries and printed tables.
      Default is ["mean", "sd", "eti89_lb", "eti89_ub", "pp_improve",
      "moderator"].
    robust_est_col : bool
      Whether to include "robust_est" in returned summaries and printed
      tables. The flag is 1 for estimates meeting both probability
      thresholds with matching signs, and 0 for estimates that do not
      meet those criteria. The "tau_study" row has no robustness flag.
      When show is "all", moderators without sensitivity results have
      NaN flags. Original input DataFrames are not modified.
      The flag is included in saved filtered tables regardless of
      this setting. Default is True.

    Returns
    -------
    included_mods : list of str
      Names of retained moderators, with "concordant" moderators first,
      followed by "mixed" moderators. Both groups have at least one
      contrast estimate meeting the full robustness criteria.
    mod_sum : dict of pandas.DataFrame
      Primary summaries for retained moderators, restricted to
      summary_cols and optionally "robust_est". Each summary contains
      shared estimates with matching signs and the primary "tau_study"
      row. Estimates with matching signs may be retained even when
      they do not meet the probability thresholds.
    robust_levels : dict of list of str
      Dictionary mapping retained moderator names to all directionally
      consistent contrast levels available for subsequent comparisons.
      Individual levels need not meet both probability thresholds.
      Labels of the form "theta_contrast_category[level]" are reduced
      to "level". Other contrast index labels are returned as strings.

    Raises
    ------
    ValueError
      If show is invalid, a matched primary or sensitivity summary has
      duplicate index labels, or evaluated "mean" or "pp_improve"
      values cannot be converted to numeric.
    KeyError
      If a required column or the primary "tau_study" row is missing.
      Also raised if requested display or summary columns are absent.

    Notes
    -----
    Directional consistency requires nonmissing posterior means with
    identical signs. Two zero means count as matching signs.
    Moderator status is assigned as follows:
    - "concordant": at least one contrast is robust, every primary contrast
      exceeding p_primary is also robust, and all shared contrasts have
      matching signs.
    - "mixed": at least one contrast is robust, but the remaining
      concordance requirements are not all satisfied.
    - "discordant": no contrast meets the full robustness criteria.
    For each matched moderator, save the complete primary summary,
    complete sensitivity summary, and filtered primary summary as
    tab-separated files with the corresponding status suffix.
    Discordant moderators are saved but excluded from returned results.
    """
    # input check
    if show not in {"all", "important", "none"}:
        raise ValueError("show must be 'all', 'important', or 'none'")
    # save path formalization
    save_loc = os.path.join(save_dir, save_folder)
    os.makedirs(save_loc, exist_ok=True)
    # initiating lists and dicts
    valid_mods = []
    partial_valid_mods = []
    robust_est_mod_sum = {}
    mod_sum = {}
    robust_levels = {}
    for ii, primary_df in mod_dict_primary.items():
        # Sensitivity analysis unavailable
        if ii not in mod_dict_sensitivity:
            primary_df.to_csv(
                os.path.join(
                    save_loc,
                    f"table_moderator-{ii}_no_sensitivity.tsv",
                ),
                sep="\t",
            )
            continue
        sensitivity_df = mod_dict_sensitivity[ii]
        # Duplicate labels would create many-to-many merge matches
        if not primary_df.index.is_unique:
            duplicates = primary_df.index[
                primary_df.index.duplicated(keep=False)
            ].unique()
            raise ValueError(
                f"Duplicate primary indices for {ii}: " f"{duplicates.tolist()}"
            )
        if not sensitivity_df.index.is_unique:
            duplicates = sensitivity_df.index[
                sensitivity_df.index.duplicated(keep=False)
            ].unique()
            raise ValueError(
                f"Duplicate sensitivity indices for {ii}: " f"{duplicates.tolist()}"
            )
        # Compare all estimates using aligned index labels
        common_idx = primary_df.index.intersection(
            sensitivity_df.index,
            sort=False,
        )
        # removing tau row
        common_idx = common_idx[common_idx != "tau_study"]
        # identifying directional cocordant rows
        full_directional_df = pd.DataFrame(
            {
                "primary_mean": pd.to_numeric(primary_df.loc[common_idx, "mean"]),
                "sensitivity_mean": pd.to_numeric(
                    sensitivity_df.loc[common_idx, "mean"]
                ),
                "primary_pp": pd.to_numeric(primary_df.loc[common_idx, "pp_improve"]),
                "sensitivity_pp": pd.to_numeric(
                    sensitivity_df.loc[common_idx, "pp_improve"]
                ),
            },
            index=common_idx,
        )
        same_direction = (
            full_directional_df["primary_mean"].notna()
            & full_directional_df["sensitivity_mean"].notna()
            & (
                np.sign(full_directional_df["primary_mean"])
                == np.sign(full_directional_df["sensitivity_mean"])
            )
        )
        # identifying concordant posterior probablities pp
        concordant_pp = (
            full_directional_df["primary_pp"].notna()
            & full_directional_df["sensitivity_pp"].notna()
            & (full_directional_df["primary_pp"] > p_primary)
            & (full_directional_df["sensitivity_pp"] > p_sensitivity)
        )
        stable_est_idx = full_directional_df.index[same_direction]
        robust_est_idx = full_directional_df.index[same_direction & concordant_pp]
        # This is the genuinely filtered primary table
        robust_est = primary_df.loc[stable_est_idx].copy()
        robust_est["robust_est"] = robust_est.index.isin(robust_est_idx).astype(int)
        tau = pd.DataFrame(primary_df.loc["tau_study", :]).T
        robust_est = pd.concat([robust_est, tau], axis=0)
        robust_est_mod_sum[ii] = robust_est
        # Qualifying moderator using existing robustness flags.
        contrast_idx = primary_df.index[
            primary_df.index.astype(str).str.contains("contrast", na=False)
        ]
        contrasts = robust_est.loc[
            robust_est.index.intersection(contrast_idx, sort=False)
        ]
        qualifying_idx = contrasts.index[contrasts["robust_est"].eq(1)]
        # establishing base status unless following criteria are met
        status = "discordant"
        if len(qualifying_idx) > 0:
            # Compare all directionally consistent levels.
            robust_levels[ii] = [
                (
                    str(level)[len("theta_contrast_category[") : -1]
                    if str(level).startswith("theta_contrast_category[")
                    else str(level)
                )
                for level in contrasts.index
            ]
            # Check whether every qualifying primary contrast was retained
            # as robust, and whether all shared contrasts agree in direction.
            primary_important_idx = contrast_idx[
                pd.to_numeric(primary_df.loc[contrast_idx, "pp_improve"]) > p_primary
            ]
            concordant = (
                primary_important_idx.isin(qualifying_idx).all()
                and same_direction.loc[
                    same_direction.index.intersection(contrast_idx, sort=False)
                ].all()
            )
            if concordant:
                status = "concordant"
                valid_mods.append(ii)
            else:
                status = "mixed"
                partial_valid_mods.append(ii)
        fn_sig = f"_{status}"
        # Save complete and robust tables
        primary_df.to_csv(
            os.path.join(
                save_loc,
                f"table_moderator-{ii}{fn_sig}.tsv",
            ),
            sep="\t",
        )
        sensitivity_df.to_csv(
            os.path.join(
                save_loc,
                f"table_moderator-{ii}_sensitivity{fn_sig}.tsv",
            ),
            sep="\t",
        )
        robust_est.to_csv(
            os.path.join(
                save_loc,
                f"table_moderator-{ii}_robust_est{fn_sig}.tsv",
            ),
            sep="\t",
        )
        print(f"Saved {ii}: {status}")
    # Full and partially valid moderators are both retained
    included_mods = valid_mods + partial_valid_mods
    # Construct returned summaries
    if robust_est_col:
        summary_cols = summary_cols + ["robust_est"]
    mod_sum = {ii: robust_est_mod_sum[ii][summary_cols] for ii in included_mods}
    if show == "all":
        for ii, df in mod_dict_primary.items():
            display_df = df.copy()
            if robust_est_col:
                display_df["robust_est"] = (
                    robust_est_mod_sum[ii]["robust_est"].reindex(
                        display_df.index, fill_value=0
                    )
                    if ii in robust_est_mod_sum
                    else np.nan
                )
            print(f"\n{ii}\n")
            print(display_df[summary_cols].to_string())
    elif show == "important":
        for ii in included_mods:
            print(f"\n{ii}\n")
            print(robust_est_mod_sum[ii][summary_cols].to_string())
    return included_mods, mod_sum, robust_levels


# comparing moderators levels --------------------------------
# from itertools import combinations
# import numpy as np
# import pandas as pd


def compare_moderator_levels(
    moderator_traces,
    moderator,
    save_dir,
    save_folder="moderator_comparisons",
    improvement_direction="<",
    eti=0.89,
    robust_levels=None,
):
    """
    Conduct all pairwise comparisons for all levels within a moderator category
    and estimate their probabilities of being best performing level.

    Inputs
    ------
    moderator traces : dict of pymc traces
        Dictionary mapping moderator names to posterior traces. Each trace
        must expose a posterior dataset containing "theta_contrast_category"
        for categorical moderators or "beta_moderator_contrast" for
        continuous moderators. These variables represent active > sham
        effects and moderator coefficients, respectively.
    moderator : string
      Name of moderator to be tested
    save_dir : filepath
      Parent directory in which to save the results
    save_folder : string
      Name of folder to create and save insice the save_dir
    improvement_direction : literal[">","<"]
        Direction indicating better outcomes. Use "<" when smaller posterior
        effects are better and ">" when larger effects are better.
        This should match the direction specified for the analysis.
        Default is "<".
    eti : float
      The equal-tailed interval from which to report. Must be finite between 0-1
    Robust_levels: dict[moderator][level]
      dictionary containing the moderator and the levels that are robust.
      Intended here to use robust_levels output from the mod_eval function.
      Only the supplied levels are compared. Categorical labels are
      matched to posterior coordinates using their string representations.
      If the moderator is absent or its collection is empty, return
      results with status "no_robust_levels".

    Returns
    -------
    pairwise_df : pandas.DataFrame
      One row per unique pair of selected categorical levels. Contains
      "moderator", "level_a", "level_b", "mean_difference",
      "sd_difference", "lower", "upper", "p_a_better", "p_b_better",
      "better_level", and "p_better".
      Differences are calculated as level_a minus level_b. The "lower"
      and "upper" columns contain equal-tailed interval bounds.
      Directional probabilities follow improvement_direction.
      The "better_level" column identifies the level with the larger
      directional probability, and "p_better" contains that probability.
      No probability threshold is applied.
      Empty when fewer than two robust categorical levels are available
      or when the moderator is continuous.

    best_df : pandas.DataFrame
      Contains "moderator", "level", "mean", "sd", "lower", "upper",
      "p_best", and "comparison_status". Summary statistics describe
      each level's posterior effect, or the continuous moderator
      coefficient.
      For multiple robust categorical levels, "p_best" is the fraction
      of posterior draws in which each level is best among the selected
      levels. Rows are sorted by decreasing "p_best".
      The "comparison_status" column identifies the result:
      - "compared": two or more robust categorical levels were compared.
      - "single_robust_level": one robust categorical level was available.
      - "continuous_moderator": the continuous coefficient was summarized.
      - "no_robust_levels": no robust levels were supplied.
      The "p_best" column is NaN unless the status is "compared".
      With no robust levels, return one row with level None and NaN
      summary statistics.


    Raises
    ------
    ValueError
      If improvement_direction is invalid, eti is nonfinite or outside
      the open interval (0, 1), or robust_levels is None.
      Also raised if neither supported posterior variable is available,
      the categorical posterior does not have exactly one non-sampling
      dimension, category labels have ambiguous string representations,
      or requested levels are duplicated or absent from the posterior.
    TypeError
      If robust_levels is not a dictionary.
    KeyError
      If robust levels are supplied but moderator is absent from
      moderator_traces.

    Notes
    -----
    Both tables are saved as tab-separated files within
    save_dir/save_folder, including when no comparisons are performed:
    - table-moderator_comparisons-pairwise_{moderator}.tsv
    - table-moderator_comparisons-best_{moderator}.tsv

    Exact posterior draw ties count toward neither pairwise direction.
    For p_best, exact ties are assigned to the first level in the supplied
    order. If pairwise directional probabilities are equal, better_level
    is assigned to level_a.

    """
    save_loc = os.path.join(save_dir, save_folder)
    os.makedirs(save_loc, exist_ok=True)
    if improvement_direction not in {"<", ">"}:
        raise ValueError("improvement_direction must be '<' or '>'")
    if not np.isfinite(eti) or not 0 < eti < 1:
        raise ValueError("eti must be finite and between 0 and 1")
    if robust_levels is None:
        raise ValueError("Provide the robust_levels dictionary returned by mod_eval().")
    if not isinstance(robust_levels, dict):
        raise TypeError("robust_levels must be the dictionary returned by mod_eval().")
    if isinstance(robust_levels, (str, bytes)):
        raise TypeError("levels must be a list of labels, not a single string")
    levels = list(robust_levels.get(moderator, []))
    # Handle this before the continuous/categorical branches.
    if not levels:
        pairwise_df = pd.DataFrame()
        best_df = pd.DataFrame(
            {
                "moderator": [moderator],
                "level": [None],
                "mean": [np.nan],
                "sd": [np.nan],
                "lower": [np.nan],
                "upper": [np.nan],
                "p_best": [np.nan],
                "comparison_status": ["no_robust_levels"],
            }
        )
        pairwise_df.to_csv(
            os.path.join(
                save_loc,
                f"table-moderator_comparisons-pairwise_{moderator}.tsv",
            ),
            sep="\t",
            index=False,
        )
        best_df.to_csv(
            os.path.join(
                save_loc,
                f"table-moderator_comparisons-best_{moderator}.tsv",
            ),
            sep="\t",
            index=False,
        )
        return pairwise_df, best_df
    trace = moderator_traces[moderator]
    if "theta_contrast_category" not in trace.posterior.data_vars:
        if "beta_moderator_contrast" in trace.posterior.data_vars:
            beta = trace.posterior["beta_moderator_contrast"].values.ravel()
            alpha = (1 - eti) / 2
            pairwise_df = pd.DataFrame()
            best_df = pd.DataFrame(
                {
                    "moderator": [moderator],
                    "level": [moderator],
                    "mean": [np.mean(beta)],
                    "sd": [np.std(beta, ddof=1)],
                    "lower": [np.quantile(beta, alpha)],
                    "upper": [np.quantile(beta, 1 - alpha)],
                    "p_best": [np.nan],
                    "comparison_status": ["continuous_moderator"],
                }
            )
            pairwise_df.to_csv(
                os.path.join(
                    save_loc,
                    f"table-moderator_comparisons-pairwise_{moderator}.tsv",
                ),
                sep="\t",
                index=False,
            )
            best_df.to_csv(
                os.path.join(
                    save_loc,
                    f"table-moderator_comparisons-best_{moderator}.tsv",
                ),
                sep="\t",
                index=False,
            )
            return pairwise_df, best_df
        raise ValueError(
            f"No categorical or continuous moderator posterior found for {moderator}"
        )
    post = trace.posterior["theta_contrast_category"]
    level_dims = [dim for dim in post.dims if dim not in ["chain", "draw"]]
    if len(level_dims) != 1:
        raise ValueError(f"Expected one moderator dimension, found: {level_dims}")
    level_dim = level_dims[0]
    all_levels = list(post.coords[level_dim].values)
    # Summary indices encode labels as text; restore native coordinate types.
    level_lookup = {}
    for native_level in all_levels:
        key = str(native_level)
        if key in level_lookup:
            raise ValueError(f"Ambiguous category labels for {moderator}: {key!r}")
        level_lookup[key] = native_level
    requested_keys = [str(level) for level in levels]
    if len(requested_keys) != len(set(requested_keys)):
        raise ValueError(f"Duplicate requested levels for {moderator}")
    missing = [key for key in requested_keys if key not in level_lookup]
    if missing:
        raise ValueError(f"Levels not found in posterior for {moderator}: {missing}")
    levels = [level_lookup[key] for key in requested_keys]
    alpha = (1 - eti) / 2
    lower_q = alpha
    upper_q = 1 - alpha
    if len(levels) == 0:
        pairwise_df = pd.DataFrame()
        best_df = pd.DataFrame(
            {
                "moderator": [moderator],
                "level": [None],
                "mean": [np.nan],
                "sd": [np.nan],
                "lower": [np.nan],
                "upper": [np.nan],
                "p_best": [np.nan],
                "comparison_status": ["no_robust_levels"],
            }
        )
    elif len(levels) == 1:
        level = levels[0]
        level_samples = (
            post.sel({level_dim: level}).stack(sample=("chain", "draw")).values
        )
        pairwise_df = pd.DataFrame()
        best_df = pd.DataFrame(
            {
                "moderator": [moderator],
                "level": [level],
                "mean": [np.mean(level_samples)],
                "sd": [np.std(level_samples, ddof=1)],
                "lower": [np.quantile(level_samples, lower_q)],
                "upper": [np.quantile(level_samples, upper_q)],
                "p_best": [np.nan],
                "comparison_status": ["single_robust_level"],
            }
        )
    else:
        samples = (
            post.sel({level_dim: levels})
            .stack(sample=("chain", "draw"))
            .transpose(level_dim, "sample")
        )
        pairwise_results = []
        for level_a, level_b in combinations(levels, 2):
            a = samples.sel({level_dim: level_a}).values
            b = samples.sel({level_dim: level_b}).values
            diff = a - b
            if improvement_direction == "<":
                p_a_better = np.mean(diff < 0)
                p_b_better = np.mean(diff > 0)
            elif improvement_direction == ">":
                p_a_better = np.mean(diff > 0)
                p_b_better = np.mean(diff < 0)
            else:
                raise ValueError("improvement_direction must be '<' or '>'")
            if p_a_better >= p_b_better:
                better_level = level_a
                p_better = p_a_better
            else:
                better_level = level_b
                p_better = p_b_better
            pairwise_results.append(
                {
                    "moderator": moderator,
                    "level_a": level_a,
                    "level_b": level_b,
                    "mean_difference": np.mean(diff),
                    "sd_difference": np.std(diff, ddof=1),
                    "lower": np.quantile(diff, lower_q),
                    "upper": np.quantile(diff, upper_q),
                    "p_a_better": p_a_better,
                    "p_b_better": p_b_better,
                    "better_level": better_level,
                    "p_better": p_better,
                }
            )
        pairwise_df = pd.DataFrame(pairwise_results)
        values = samples.values
        if improvement_direction == "<":
            best_idx = np.argmin(values, axis=0)
        else:
            best_idx = np.argmax(values, axis=0)
        best_results = []
        for ii, level in enumerate(levels):
            level_samples = values[ii]
            best_results.append(
                {
                    "moderator": moderator,
                    "level": level,
                    "mean": np.mean(level_samples),
                    "sd": np.std(level_samples, ddof=1),
                    "lower": np.quantile(level_samples, lower_q),
                    "upper": np.quantile(level_samples, upper_q),
                    "p_best": np.mean(best_idx == ii),
                    "comparison_status": "compared",
                }
            )
        best_df = (
            pd.DataFrame(best_results)
            .sort_values("p_best", ascending=False)
            .reset_index(drop=True)
        )
    pairwise_df.to_csv(
        os.path.join(
            save_loc,
            f"table-moderator_comparisons-pairwise_{moderator}.tsv",
        ),
        sep="\t",
        index=False,
    )
    best_df.to_csv(
        os.path.join(
            save_loc,
            f"table-moderator_comparisons-best_{moderator}.tsv",
        ),
        sep="\t",
        index=False,
    )
    return pairwise_df, best_df


# Identifying best levels for plotting purposes----------------------------------------
def get_best_level(best_df, best_th=0.90):
    """
    Identify from best_df dataframe which meets criteria for best usign a user specified threshold

    Inputs
    ------
    best_df: dataframe
      Dataframe testing which level performs best from compare_moderator_levels()
    best_th: float
      Uset set threshold for posterior interpretation for which moderator performs best.
      Default=0.90

    Returns
    -------
    dataframe with added column indicating which level in that moderator dataframe performs the best.
    indicated by 1 for best

    best_df : pandas.DataFrame
        Results from compare_moderator_levels(). A nonempty DataFrame must
        contain "comparison_status", "level", and "p_best".
        For multiple rows, consider only those marked "compared" with
        nonmissing levels and numeric probabilities between 0 and 1.
        Probabilities are converted to numeric; invalid values are excluded.
    best_th : float, optional
        Minimum posterior probability required to select the level with
        the highest "p_best". Must be finite and between 0 and 1,
        inclusive. Default is 0.90.

    Returns
    -------
    best_level : object or None
        Level with the unique highest valid "p_best" at or above
        `best_th`. Return None if no eligible level meets the threshold
        or multiple eligible rows share the highest probability.

        If `best_df` contains exactly one row, return its nonmissing level
        only when "comparison_status" is "single_robust_level",
        regardless of its probability or `best_th`. Otherwise, return None.

        Also return None when `best_df` is None or empty.

    Raises
    ------
    ValueError
        If `best_th` is nonfinite or outside [0, 1], or a nonempty
        `best_df` is missing required columns.
    """
    if not np.isfinite(best_th) or not 0 <= best_th <= 1:
        raise ValueError("best_th must be finite and between 0 and 1")
    if best_df is None or best_df.empty:
        return None
    required = {"comparison_status", "level", "p_best"}
    missing = required.difference(best_df.columns)
    if missing:
        raise ValueError(f"best_df is missing columns: {sorted(missing)}")
    if len(best_df) == 1:
        row = best_df.iloc[0]
        if row["comparison_status"] == "single_robust_level" and pd.notna(row["level"]):
            return row["level"]
        return None
    compared = best_df.loc[best_df["comparison_status"].eq("compared")].copy()
    compared["p_best"] = pd.to_numeric(compared["p_best"], errors="coerce")
    compared = compared.loc[
        compared["level"].notna() & compared["p_best"].between(0, 1)
    ]
    if compared.empty:
        return None
    highest_probability = compared["p_best"].max()
    if highest_probability < best_th:
        return None
    winners = compared.loc[compared["p_best"].eq(highest_probability)]
    # Do not arbitrarily mark one category when probabilities tie.
    if len(winners) != 1:
        return None
    return winners.iloc[0]["level"]
