from pr_pilot.data.clustering import parse_rfam_tbl_fmt2


def test_cmscan_fmt2_parser_accounts_for_prepended_index_column():
    lines = [
        "# comment",
        "1 tRNA RF00005 singleR::abc - CL00001 cm 1 73 1 73 + no 1 0.50 0.0 42.0 1e-10 ! * * * * * * 0.98 tRNA",
        "2 U2 RF00004 complexR::xyz::B - CL00002 cm 1 100 1 100 + no 1 0.50 0.0 50.0 1e-20 ! * * * * * * 0.99 U2",
    ]
    hits = parse_rfam_tbl_fmt2(lines)
    assert hits["singleR::abc"] == {"RF00005"}
    assert hits["complexR::xyz::B"] == {"RF00004"}
    assert "1" not in hits
