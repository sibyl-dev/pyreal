from pyreal.transformers.aggregator import Mappings
import pandas as pd


def test_generate_mappings():
    one_to_many = {
        "parent1": ["child1", "child2"],
        "parent2": ["child3"],
        "parent3": ["child4", "child5", "child6"],
    }
    many_to_one = {
        "child1": "parent1",
        "child2": "parent1",
        "child3": "parent2",
        "child4": "parent3",
        "child5": "parent3",
        "child6": "parent3",
    }

    dataframe = pd.DataFrame(
        [
            ["parent1", "child1"],
            ["parent1", "child2"],
            ["parent2", "child3"],
            ["parent3", "child4"],
            ["parent3", "child5"],
            ["parent3", "child6"],
        ],
        columns=["parent", "child"],
    )

    mappings = Mappings.generate_mappings(one_to_many=one_to_many)
    assert mappings.one_to_many == one_to_many
    assert mappings.many_to_one == many_to_one

    mappings = Mappings.generate_mappings(many_to_one=many_to_one)
    assert mappings.one_to_many == one_to_many
    assert mappings.many_to_one == many_to_one

    mappings = Mappings.generate_mappings(dataframe=dataframe)
    assert mappings.one_to_many == one_to_many
    assert mappings.many_to_one == many_to_one
