# hiermeta: Hierarchical Bayesian meta-analysis for Python, with descriptive summaries, moderator evaluation, and plotting.


[![PyPI Downloads](https://static.pepy.tech/personalized-badge/hiermeta?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/hiermeta)

## Installation

```bash
pip install hiermeta
```

Supports python versions >=3.12

## Worked example

This example loads the bundled dataset, fits the overall model, and reads the active > sham contrast. Detailed input requirements, options, and return values are available in the function docstrings.

### Load and select the data

```python
import hiermeta as hm

grp_map = {
    "active": 1,
    "sham": 0,
}

time_map = {
    "post": 1,
    "pre": 0,
    "diff": 3,
}

df = hm.load_example_data()

df["group"] = df.group.map(grp_map)
df["time"] = df.time.map(time_map)

print(df[["ID", "group", "time", "measure_name", "mean", "sd", "n"]].head())
print(df["measure_category"].dropna().unique())
```

Each row contains summary data for a study arm, measure, and time point. The example shows the layout expected by the analysis; when using your own data, check the column requirements and coding in `help(hm.bayes_meta_hre)`.

Choose one outcome category for the analysis. Here, the first category is selected for demonstration; replace it with the category you intend to study.

```python
analysis_df = df.loc[df["measure_category"].eq("externalizing")].copy()

print(f"Selected outcome category: {"externalizing}")
```

### Fit the overall model

The call below uses `improvement_direction="<"` to illustrate an analysis where lower values on the aligned outcome direction indicate improvement. Check that this matches your selected category and its direction coding. Use `">"` when improvement corresponds to higher aligned values.

```python
base_summary, info = hm.bayes_meta_hre(
    df = analysis_df,
    moderator_names=None,
    continuous_expected_mod=None,
    dichotomous_mod=None,
    improvement_direction="<",
    tacc=0.99,
    baseline_cent=True,
    measure_col="measure_name",
    direction_col="category_direction",
    moderation=False)


```

This step runs posterior sampling. Runtime depends on the dataset and hardware.

### Read the results

```python
print(base_summary)

contrast = base_summary.loc["theta_contrast"]

print(f"Posterior mean (active > sham): {contrast['mean']:.3f}")
print(f"Posterior probability of improvement: {contrast['pp_improve']:.1%}")

trace = info["trace_base"]
```

For this example, a negative contrast favors active over sham in the specified improvement direction. With `improvement_direction="<"`, `pp_improve` is the proportion of posterior draws below zero. For example, a value of 0.97 would indicate 97% posterior probability that the overall contrast lies in that direction, conditional on the data and model. This is an illustration of interpretation, not a result claimed for the bundled dataset.

Read directional probability alongside the estimated magnitude and posterior interval. The contrast is expressed on the model's shared latent scale; it is not automatically a standardized mean difference such as Cohen's d. The `tau_study` row summarizes the standard deviation of study-level variation in the contrast.

Review sampling diagnostics, including divergences, R-hat, effective sample sizes, and trace plots, before drawing conclusions. The returned `trace` provides access to posterior draws and sampling diagnostics.


## Main functions

Use Python's `help()` to view the full documentation for any function:

```python
help(hm.bayes_meta_hre)
help(hm.mod_eval)
```

| Function | Purpose |
|----------|---------|
| `load_example_data()` | Load the bundled example dataframe |
| `bayes_meta_hre()` | Fit hierarchical Bayesian meta-analysis models |
| `describe_meta_df()` | Summarize the input dataset |
| `df_pre_post()` | Select pre/post data and count study formats |
| `mod_eval()` | Evaluate moderators across primary and sensitivity analyses |
| `compare_moderator_levels()` | Compare retained moderator levels |
| `get_best_level()` | Identify a best level from comparison results |
| `plot_bayes_meta_results()` | Plot analysis results |


## Citation

Will add formal citation details when available. Whe reporting an anlaysis, include the package name and version used.


## Support and contributions

Any bugs or inquiery for additional features please make a request on the github site. Will add github when available

## License

Released under the [MIT License](LICENSE).
