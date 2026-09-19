from __future__ import annotations

import pandas as pd

from pr_pilot.data.manifest import assert_no_test_leakage


def main() -> None:
    manifests = r"I:\PR_PILOT_SCIENTIFIC\20260917\manifest_length_v1"
    annotated = r"I:\PR_PILOT_SCIENTIFIC\20260919\rna_entropy_balance\blind_freeze\annotated\complex_annotated.tsv"
    all_rows = pd.read_csv(annotated, sep="\t", dtype=str)
    dev_ids = set(pd.concat([pd.read_csv(f"{manifests}\\complex_train.tsv", sep="\t", dtype=str), pd.read_csv(f"{manifests}\\complex_val.tsv", sep="\t", dtype=str)], ignore_index=True)["sample_id"])
    test_ids = set(pd.read_csv(f"{manifests}\\complex_test.tsv", sep="\t", dtype=str)["sample_id"])
    dev = all_rows[all_rows["sample_id"].isin(dev_ids)]
    test = all_rows[all_rows["sample_id"].isin(test_ids)]
    known = dev_ids | test_ids
    candidates = all_rows[~all_rows["sample_id"].astype(str).isin(known)]
    print(candidates[["sample_id", "protein_cluster_p30", "rna_cluster_r80", "rfam_family"]].to_string(index=False))
    for _, row in candidates.iterrows():
        try:
            assert_no_test_leakage(dev, test, pd.DataFrame([row]), strict_cluster_check=True)
        except AssertionError as exc:
            print(f"REJECT\t{row['sample_id']}\t{exc}")
        else:
            print(f"SAFE\t{row['sample_id']}")


if __name__ == "__main__":
    main()
