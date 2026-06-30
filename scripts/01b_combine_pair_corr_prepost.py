#%%
import os
from glob import glob
from pathlib import Path
import pandas as pd
import config as C
from scripts.utils.io import read_table
from scripts.utils.noise_corr_utils import combine_corr_results

def combine_pair_tables(kind, subfoldername):
    files = sorted(glob(str(C.RESULTSPATH / subfoldername / f"pair_{kind}_*.parquet")))

    if not files:
        print(f"[No files] {C.RESULTSPATH / subfoldername / f'pair_{kind}_*.parquet'}")
        return pd.DataFrame()

    df_all = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)

    if kind == "noise":
        key_cols = [
            "session_pid",
            "cluster_region",
            "cluster_id1",
            "cluster_id2",
            "window",
            "nc_t0",
            "nc_t1",
        ]
    elif kind == "signal":
        key_cols = [
            "session_pid",
            "cluster_region",
            "cluster_id1",
            "cluster_id2",
            "window",
            "sc_t0",
            "sc_t1",
        ]
    else:
        raise ValueError("kind must be 'noise' or 'signal'")

    key_cols = [c for c in key_cols if c in df_all.columns]

    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=key_cols)
    print(f"[{subfoldername} | {kind}] Dropped duplicated rows: {before - len(df_all)}")

    out = C.DATAPATH / f"all_pair_neural_metrics_r_{kind}_{subfoldername}.parquet"
    df_all.to_parquet(out, index=False)

    print(f"[Saved] {out}")
    print(f"Rows: {len(df_all)}")
    print(f"Files combined: {len(files)}")

    return df_all


def check_pair_file_coverage(kind, subfoldername):
    result_dir = C.RESULTSPATH / subfoldername

    recordings_filtered = read_table(C.DATAPATH / "BWM_LL_release_afterQC_df.csv")
    all_pids = set(recordings_filtered["pid"].astype(str))

    excluded_pids = set(map(str, C.PIDS_WITHOUT_ILBLSORTOR))
    excluded_pids_in_table = all_pids & excluded_pids
    expected_pids = all_pids - excluded_pids_in_table

    pair_files = sorted(glob(str(result_dir / f"pair_{kind}_*.parquet")))
    done_files = sorted(glob(str(result_dir / "*.noise_signal.done")))

    file_pids = {
        Path(f).name.replace(f"pair_{kind}_", "").replace(".parquet", "")
        for f in pair_files
    }

    done_pids = {
        Path(f).name.replace(".noise_signal.done", "")
        for f in done_files
    }

    missing_pair_file = sorted(expected_pids - file_pids)
    missing_done_file = sorted(expected_pids - done_pids)
    done_but_no_pair_file = sorted(done_pids - file_pids)

    print(f"\n[Coverage] {subfoldername} | {kind}")
    print(f"All pids in recording table: {len(all_pids)}")
    print(f"Excluded pids listed in config: {len(excluded_pids)}")
    print(f"Excluded pids actually in table: {len(excluded_pids_in_table)}")
    print(f"Expected pids after exclusion: {len(expected_pids)}")
    print(f"Pair files found: {len(file_pids)}")
    print(f"Done files found: {len(done_pids)}")
    print(f"Missing pair files: {len(missing_pair_file)}")
    print(f"Missing done files: {len(missing_done_file)}")
    print(f"Done but no pair file: {len(done_but_no_pair_file)}")

    return {
        "kind": kind,
        "subfoldername": subfoldername,
        "missing_pair_file": missing_pair_file,
        "missing_done_file": missing_done_file,
        "done_but_no_pair_file": done_but_no_pair_file,
        "excluded_pids_in_table": sorted(excluded_pids_in_table),
    }


def combine_noise_signal_for_window(window_name, subfoldername):
    noise = combine_pair_tables("noise", subfoldername)
    signal = combine_pair_tables("signal", subfoldername)

    if noise.empty or signal.empty:
        print(f"[Skip merge] {window_name}: empty noise or signal table.")
        return pd.DataFrame()

    merged = combine_corr_results(noise, signal)

    out = C.DATAPATH / f"all_pair_neural_metrics_merged_noise_signal_{subfoldername}.parquet"
    merged.to_parquet(out, index=False)

    print(f"[Saved merged] {out}")
    print(f"Merged rows: {len(merged)}")

    return merged


def main():
    coverage_records = []
    merged_tables = []

    for window_name, window_cfg in C.NC_WINDOWS.items():
        subfoldername = window_cfg["subfoldername"]

        coverage_records.append(check_pair_file_coverage("noise", subfoldername))
        coverage_records.append(check_pair_file_coverage("signal", subfoldername))

        merged = combine_noise_signal_for_window(window_name, subfoldername)
        if not merged.empty:
            merged_tables.append(merged)

    if coverage_records:
        coverage_summary = pd.DataFrame(coverage_records)
        out_coverage = C.DATAPATH / "proj2_noise_pair_corr_file_coverage_prepost.csv"


        coverage_summary.to_csv(out_coverage, index=False)
        print(f"[Saved coverage] {out_coverage}")

    if merged_tables:
        merged_prepost = pd.concat(merged_tables, ignore_index=True)
        out_prepost = C.DATAPATH / "proj2_noise_pair_corr_merged_prepost_500ms.parquet"
        merged_prepost.to_parquet(out_prepost, index=False)

        print(f"[Saved prepost merged] {out_prepost}")
        print(f"Rows: {len(merged_prepost)}")
        print(merged_prepost["window"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
