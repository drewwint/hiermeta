# descriptives.py: functions for descriptive information from meta analysis dataframes

# package imports-----------------------------------------------------
import numpy as np
import pandas as pd
import os


# Pre - Post dataframe-------------------------------------------------
def _clean_id(value):
    """
    Normalize a scalar identifier to a string or missing value.

    Inputs
    ------
    value : scalar
        Identifier to normalize. Missing values recognized by pandas
        are returned as pd.NA. Other values are converted to strings
        and stripped of leading and trailing whitespace.

    Returns
    -------
    cleaned_id : str or pandas.NA
        Normalized identifier. Numeric values representing integers
        are returned without a decimal suffix. Other numeric values
        are formatted with up to 15 significant digits. Nonnumeric
        strings retain their content after whitespace removal.

    Notes
    -----
    Numeric identifiers are parsed as floats. Leading zeros and original
    numeric formatting are not preserved, and large or highly precise
    numeric identifiers may lose precision.
    Missing-value detection occurs before string conversion. A string
    such as "nan" therefore remains a string rather than becoming pd.NA.
    """
    if pd.isna(value):
        return pd.NA
    value = str(value).strip()
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return format(number, ".15g")
    except ValueError:
        return value


def df_pre_post(df, id="ID", group="group", measure_col="measure_name"):
    """
    Extract pre-post records and count study-level data formats.

    Classify records within each study-arm-measure group using assessment
    times. Return the pre-post subset and counts of studies represented
    in pre-post, post-only, difference-only, mixed, or unclassified formats.

    Inputs
    ------
    df : pandas.DataFrame
        Meta-analysis dataset containing the specified identifier,
        group, and measure columns, plus a "time" column.

        Assessment times are coded 0 for pre, 1 for post, and 3 for
        difference scores. Times are converted to numeric, with invalid
        values converted to NaN. The input DataFrame is not modified.
    id : str, optional
        Study identifier column. Values are normalized using _clean_id()
        and stored with pandas string dtype. Study counts use unique,
        nonmissing identifiers after normalization. Default is "ID".
    group : str, optional
        Study-arm column used to group records within each study.
        Default is "group".
    measure_col : str, optional
        Outcome-measure column used to group records within each
        study arm. Default is "measure_name".

    Returns
    -------
    df_pp : pandas.DataFrame
        Records from study-arm-measure groups containing both pre
        and post assessments. All rows belonging to qualifying groups
        are retained, including rows with difference scores, other
        assessment times, or missing times.

        Includes all original columns, with normalized identifiers,
        numeric assessment times, and a reset integer index.
    n_vals : dict of str to int
        Counts of unique, nonmissing study identifiers:
        - "n_total": all studies in the normalized input.
        - "n_prepost_only": studies represented in the pre-post subset
          but neither the post-only nor difference-only subset.
        - "n_post_only": studies represented in the post-only subset
          but neither of the other two subsets.
        - "n_diff_only": studies represented in the difference-only
          subset but neither of the other two subsets.
        - "n_mixed": studies represented in at least two subsets.
        - "n_unclassified": studies represented in none of the subsets.

        The five classification counts sum to "n_total".

    Raises
    ------
    KeyError
        If a specified grouping column or the "time" column is missing.

    Notes
    -----
    A pre-post group contains both time 0 and time 1. A post-only group
    contains time 1 without time 0 or time 3; only its time-1 rows enter
    the post-only subset. A difference-only group contains time 3
    without time 0 or time 1; only its time-3 rows enter the
    difference-only subset.

    Post-only and difference-only subsets are used internally for
    classification and are not returned.

    Missing grouping values are retained during grouping. Records with
    missing study identifiers can appear in df_pp but do not contribute
    to study counts.

    Study classifications reflect subset membership. Unclassified
    records within an otherwise classified study do not make that
    study mixed or unclassified.
    """
    df = df.copy()
    df[id] = df[id].map(_clean_id).astype("string")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    group_keys = [id, group, measure_col]
    grouped_time = df.groupby(group_keys, dropna=False)["time"]
    has_pre = grouped_time.transform(lambda x: x.eq(0).any())
    has_post = grouped_time.transform(lambda x: x.eq(1).any())
    has_diff = grouped_time.transform(lambda x: x.eq(3).any())
    is_postonly = df["time"].eq(1) & ~has_pre & ~has_diff & has_post
    is_diffonly = df["time"].eq(3) & has_diff & ~has_pre & ~has_post
    df_pp = df.loc[has_pre & has_post].reset_index(drop=True).copy()
    df_pt = df.loc[is_postonly].reset_index(drop=True).copy()
    df_df = df.loc[is_diffonly].reset_index(drop=True).copy()
    pp_ids = set(df_pp[id].dropna().unique())
    pt_ids = set(df_pt[id].dropna().unique())
    df_ids = set(df_df[id].dropna().unique())
    prepost_only = pp_ids - pt_ids - df_ids
    postonly_only = pt_ids - pp_ids - df_ids
    diffonly_only = df_ids - pp_ids - pt_ids
    mixed_ids = (pp_ids & pt_ids) | (pp_ids & df_ids) | (pt_ids & df_ids)
    classified_ids = pp_ids | pt_ids | df_ids
    all_ids = set(df[id].dropna().unique())
    unclassified_ids = all_ids - classified_ids
    n_vals = {
        "n_total": len(all_ids),
        "n_prepost_only": len(prepost_only),
        "n_post_only": len(postonly_only),
        "n_diff_only": len(diffonly_only),
        "n_mixed": len(mixed_ids),
        "n_unclassified": len(unclassified_ids),
    }
    return df_pp, n_vals


# descriptives for the final dataframe-------------------------------


def describe_meta_df(
    df,
    id="ID",
    group="group",
    measure_col="measure_name",
    sample_size_col="n",
    save_dir=None,
    save_name=None,
    continuous_vars=[
        "mean_age",
        "p_male",
        "p_female",
    ],
    categorical_vars=[
        "diagnosis",
        "sample_type",
        "study_type",
        "blinding",
        "randomization",
    ],
):
    """
    Summarize study characteristics and outcome composition of meta-analysis data.

    Inputs
    ------
    df : pandas.DataFrame
        Nonempty meta-analysis dataset containing study identifiers, group
        assignments, measure names, sample sizes, and assessment times.
        The "time" column uses 0 for pre, 1 for post, and 3 for difference
        scores. Additional requirements for study-level data-format
        classification follow df_pre_post().
    id : str
        Study identifier column used for overview counts, sample-size
        aggregation, and df_pre_post(). Default is "ID".
        Other descriptive sections currently use the literal column "ID".
    group : str
        Study-arm column used for sample-size aggregation and
        df_pre_post(). Sham is coded 0 and active is coded 1.
        Default is "group". Arm-specific totals currently access the
        literal column "group".
    measure_col : str
        Column identifying outcome measures. Default is "measure_name".
    sample_size_col : str
        Column containing numeric sample sizes. The maximum reported
        sample size within each study arm is used to calculate nominal
        participant totals. Default is "n".
    save_dir : str or path-like or None
        Directory in which to save the descriptive tables. Created if
        necessary. If None, tables are returned without saving.
        Default is None.
    save_name : str or None
        Name included in saved filenames:
        "descriptives_{save_name}_{table_name}.tsv".
        If None or empty, use df["measure_category"][0] as the name.
        Ignored when save_dir is None. Default is None.
    continuous_vars : list of str
        Columns to summarize using continuous descriptive statistics.
        Missing columns and columns without complete study-value pairs
        are skipped. Values must be convertible to float.
        Summaries use unique, nonmissing combinations of "ID" and the
        selected variable. A study can therefore contribute multiple
        values when that characteristic varies within the study.
        Values are not weighted by sample size.
        Defaults to the column names listed in the function signature.
    categorical_vars : list of str
        Columns to summarize using category counts and percentages.
        Missing columns and incomplete study-category pairs are skipped.
        Each study is counted once per distinct category. Percentages
        use the total number of studies in df as the denominator,
        including studies missing that characteristic.
        Defaults to the column names listed in the function signature.

    Returns
    -------
    describe_dict : dict of pandas.DataFrame
        Dictionary containing six descriptive tables:
        - "overview": study count, number of unique study-measure
          combinations, total nominal sample size, sham and active
          sample sizes, and mean, median, minimum, and maximum nominal
          sample sizes per study.
        - "data_format_by_study": study-level classifications obtained
          from df_pre_post(), indexed by "pre-post", "post-only",
          "diff-only", "mixed", and "unclassified". Contains "n_studies"
          and "perc_studies". Despite its name, "perc_studies" contains
          proportions on a 0-to-1 scale.
        - "data_format_by_measure": counts of study-measure combinations
          classified as "prepost", "postonly", or "difference".
          The count column uses sample_size_col as its name but counts
          study-measure combinations. The "percent" column uses a
          0-to-100 scale.
        - "continuous": "variable", "n", "mean", "sd", "median", "min",
          and "max". The "n" column counts unique nonmissing study-value
          pairs, and "sd" is the sample standard deviation.
        - "categorical": "variable", "level", "n_studies", and
          "percent_studies". A study may appear in multiple levels,
          so percentages within a variable can sum to more than 100.
        - "outcomes": composition of available outcome descriptors:
          "measure_type", "measure_category", "measure_specific",
          "measure_domain", "measure_dimension", and "measure_scale_name".
          For each variable and level, reports unique study counts
          ("n_studies"), unique measure-name counts ("n_measures"),
          and study-measure combination counts ("n_study_measure").

    Notes
    -----
    Nominal sample sizes sum the maximum reported sample size for each
    study arm. Repeated measures and assessment times within an arm do
    not add to this total. Participants shared across arms or study
    identifiers are not identified or deduplicated.
    Study-measure data formats are classified using assessment times
    pooled across arms. Combinations containing both pre and post times
    are classified as "prepost"; those containing post without pre or
    difference times are classified as "postonly". All remaining
    patterns are classified as "difference".
    Several sections currently reference "ID" and "group" directly,
    so alternative id and group column names are not fully supported.
    When saving is enabled, each returned table is written to a separate
    tab-separated file, including its DataFrame index.
    """
    df = df.copy()
    # basic study counts
    studies = df[id].nunique()
    study_measure = df[[id, measure_col]].drop_duplicates()
    # nominal sample size by study arm without double-counting measures/time
    arm_n = df.groupby([id, group])[sample_size_col].max().reset_index()
    total_n = int(arm_n[sample_size_col].sum())
    sham_n = int(arm_n.loc[arm_n.group == 0, sample_size_col].sum())
    active_n = int(arm_n.loc[arm_n.group == 1, sample_size_col].sum())
    study_n = arm_n.groupby(id)[sample_size_col].sum()
    overview = pd.DataFrame(
        {
            "value": [
                studies,
                len(study_measure),
                total_n,
                sham_n,
                active_n,
                float(study_n.mean()),
                float(study_n.median()),
                int(study_n.min()),
                int(study_n.max()),
            ]
        },
        index=[
            "studies",
            "study_measure_combinations",
            "total_nominal_n",
            "sham_n",
            "active_n",
            "mean_n_per_study",
            "median_n_per_study",
            "min_n_per_study",
            "max_n_per_study",
        ],
    )
    # identifying analysis format at study level
    _, v_vals = df_pre_post(df, id=id, group=group, measure_col=measure_col)
    n_vals = [
        v_vals["n_prepost_only"],
        v_vals["n_post_only"],
        v_vals["n_diff_only"],
        v_vals["n_mixed"],
        v_vals["n_unclassified"],
    ]
    p_vals = np.array(n_vals) / v_vals["n_total"]
    study_data_format_summary = pd.DataFrame(
        {
            "n_studies": n_vals,
            "perc_studies": p_vals,
        },
        index=["pre-post", "post-only", "diff-only", "mixed", "unclassified"],
    )
    # identifying analysis format at study-measure level
    design_rows = []
    for (sid, measure), d in df.groupby(["ID", measure_col]):
        times = set(d["time"].dropna().astype(int))
        if 3 in times and 0 not in times and 1 not in times:
            design = "difference"
        elif 0 in times and 1 in times:
            design = "prepost"
        elif 0 in times and 1 in times and 3 in times:
            design = "prepost"
        elif 0 not in times and 3 not in times and 1 in times:
            design = "postonly"
        else:
            design = "difference"
        design_rows.append({"ID": sid, measure_col: measure, "data_format": design})
    data_format = pd.DataFrame(design_rows)
    data_format_summary = (
        data_format["data_format"]
        .value_counts()
        .rename_axis("data_format")
        .reset_index(name=sample_size_col)
    )
    data_format_summary["percent"] = (
        100 * data_format_summary[sample_size_col] / len(data_format)
    )
    # continuous study characteristics
    continuous_rows = []
    for var in continuous_vars:
        if var not in df.columns:
            continue
        x = df[["ID", var]].dropna().drop_duplicates()
        if len(x) == 0:
            continue
        values = x[var].astype(float)
        continuous_rows.append(
            {
                "variable": var,
                "n": len(values),
                "mean": values.mean(),
                "sd": values.std(),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
            }
        )
    continuous = pd.DataFrame(continuous_rows)
    # categorical study characteristics
    categorical_rows = []
    for var in categorical_vars:
        if var not in df.columns:
            continue
        x = df[["ID", var]].dropna().drop_duplicates()
        for level, n in x[var].value_counts().items():
            categorical_rows.append(
                {
                    "variable": var,
                    "level": level,
                    "n_studies": int(n),
                    "percent_studies": 100 * n / studies,
                }
            )
    categorical = pd.DataFrame(categorical_rows)
    # outcome composition
    outcome_vars = [
        "measure_type",
        "measure_category",
        "measure_specific",
        "measure_domain",
        "measure_dimension",
        "measure_scale_name",
    ]
    outcome_rows = []
    for var in outcome_vars:
        if var not in df.columns:
            continue
        x = df[["ID", measure_col, var]].dropna().drop_duplicates()
        for level, g in x.groupby(var):
            outcome_rows.append(
                {
                    "variable": var,
                    "level": level,
                    "n_studies": g["ID"].nunique(),
                    "n_measures": g[measure_col].nunique(),
                    "n_study_measure": len(g),
                }
            )
    outcomes = pd.DataFrame(outcome_rows)
    describe_dict = {
        "overview": overview,
        "data_format_by_study": study_data_format_summary,
        "data_format_by_measure": data_format_summary,
        "continuous": continuous,
        "categorical": categorical,
        "outcomes": outcomes,
    }
    if save_dir is not None and not save_name:
        os.makedirs(save_dir, exist_ok=True)
        for jj in describe_dict.keys():
            describe_dict[jj].to_csv(
                os.path.join(
                    save_dir, f"descriptives_{df.measure_category[0]}_{jj}.tsv"
                ),
                sep="\t",
            )
    elif save_dir is not None and save_name is not None:
        os.makedirs(save_dir, exist_ok=True)
        for jj in describe_dict.keys():
            describe_dict[jj].to_csv(
                os.path.join(save_dir, f"descriptives_{save_name}_{jj}.tsv"),
                sep="\t",
            )
    return describe_dict
