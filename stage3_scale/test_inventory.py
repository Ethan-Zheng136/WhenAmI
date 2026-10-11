from stage3_scale.inventory import balanced_assignment, role_for_key


def test_pinned_holdout_roles_cannot_be_overwritten():
    registry = {("bridge_orig", "episode"): "test", ("fractal20220817_data", "dev"): "dev"}
    assert role_for_key("bridge_orig", "episode", registry, split_seed=999) == "test"
    assert role_for_key("fractal20220817_data", "dev", registry, split_seed=999) == "dev"


def test_new_episode_role_agrees_with_original_reader_study():
    from stage3.reader_study_data import assign_partition
    for i in range(100):
        key = f"content-key-{i}"
        assert role_for_key("bridge_orig", key, {}) == assign_partition(key, 20261002)


def test_source_shards_have_exactly_one_owner_deterministically():
    rows = [{"id": str(i), "source_episodes": i + 1} for i in range(33)]
    assigned, loads = balanced_assignment(rows, 8)
    assert len(assigned) == 33
    assert len({r["id"] for r in assigned}) == 33
    assert sum(loads) == sum(range(1, 34))
    assert all(0 <= r["extraction_worker"] < 8 for r in assigned)
    assert (assigned, loads) == balanced_assignment(rows, 8)
    assert all("extraction_worker" not in r for r in rows)
