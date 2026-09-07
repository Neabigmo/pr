from pr_pilot.data.rcsb_io import rcsb_query


def test_rcsb_query_uses_numeric_operators_for_numeric_counts():
    for kind in ["protein", "rna", "complex"]:
        query = rcsb_query(kind)
        nodes = query["query"]["nodes"]
        by_attribute = {node["parameters"]["attribute"]: node["parameters"] for node in nodes}
        assert by_attribute["rcsb_entry_info.polymer_entity_count_DNA"]["operator"] == "equals"
        assert by_attribute["rcsb_entry_info.polymer_entity_count_nucleic_acid_hybrid"]["operator"] == "equals"
        for attribute, parameters in by_attribute.items():
            if attribute.startswith("rcsb_entry_info.polymer_entity_count_"):
                assert parameters["operator"] in {"equals", "greater"}
