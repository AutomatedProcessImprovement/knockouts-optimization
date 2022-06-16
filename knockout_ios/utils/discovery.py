import collections
from itertools import chain

import pm4py
from pm4py import filter_eventually_follows_relation, filter_directly_follows_relation

from knockout_ios.utils.constants import globalColumnNames


def get_sorted_variants(df):
    # Find variants & sort by prefix length

    df = df.sort_values(
        by=[globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME, globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME])

    variants = pm4py.get_variants_as_tuples(df)
    entries = []
    for item in variants.items():
        entry = {'prefix': list(item[0]), 'case_count': item[1], 'prefix_len': len(item[0])}
        entries.append(entry)

    variants = sorted(entries, key=lambda e: e['case_count'], reverse=True)
    variants = sorted(variants, key=lambda e: e['prefix_len'])

    return variants


def find_most_frequent_tuple(tuples_dict):
    flat_values = list(chain(*tuples_dict.values()))

    common = collections.Counter(flat_values).most_common()

    if len(common) > 0:
        return collections.Counter(flat_values).most_common()[0]

    return ()


def get_possible_ko_sequences(variants, limit):
    ko_ac = {}

    for i in range(0, len(variants)):
        variant = variants[i]
        ko_info = variant['most_frequent_differentiating_transition']

        if len(ko_info) > 0:
            key = ko_info[0][0]
            if not (key in ko_ac.keys()):
                ko_ac[key] = (ko_info[0][1], variant['prefix'][-2], variant['prefix_len'])

            # Store also second part of sequence as possible KO
            sequence_end = ko_info[0][1]

            if not (sequence_end in ko_ac.keys()):
                # Find activity after second part of sequence
                idx = next(i for i, v in enumerate(variant['prefix']) if v == sequence_end)

                # Don't consider this activity if it is next-to-last to the End of variant
                if idx == (len(variant['prefix']) - 2):
                    continue

                if (idx + 1) < len(variant['prefix']):
                    next_activity = variant['prefix'][idx + 1]
                else:
                    next_activity = variant['prefix'][-2]

                ko_ac[sequence_end] = (next_activity, variant['prefix'][-2], variant['prefix_len'])

        if len(ko_ac.keys()) >= limit:
            break

    return ko_ac


def discover_ko_sequences_known_post_kos(df, post_knockout_activities):
    # if negative outcome(s) are known, simply get all the activities that directly-follow them

    activities = df[globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME].unique()
    activities = [a for a in activities if a not in post_knockout_activities]

    relations = []

    for activity in activities:
        for post_ko_activity in post_knockout_activities:
            relations.append((activity, post_ko_activity))

    df = df.sort_values(
        by=[globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME,
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME])
    df = filter_directly_follows_relation(df, relations)
    df = df.sort_values(
        by=[globalColumnNames.SIMOD_LOG_READER_CASE_ID_COLUMN_NAME,
            globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME])

    # add a column to df with the value of the next activity
    df["next_activity"] = (df[globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME].shift(-1))

    # keep only rows where next activity is in the list of post-ko activities
    df = df[df["next_activity"].isin(post_knockout_activities)]

    ko_activities = df[globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME].unique()

    return ko_activities, post_knockout_activities, None


def discover_ko_sequences(df, config_file_name, cache_dir, limit=3, post_knockout_activities=None,
                          success_activities=None,
                          known_ko_activities=None,
                          quiet=False, start_activity_name="Start", force_recompute=False):
    if success_activities is None:
        success_activities = []

    if post_knockout_activities is None:
        post_knockout_activities = []

    if known_ko_activities is None:
        known_ko_activities = []

    if not quiet:
        print(f"Cache for {config_file_name} variants not found")

    if len(post_knockout_activities) > 0:
        return discover_ko_sequences_known_post_kos(df, post_knockout_activities)

    if len(success_activities) > 0:
        relations = list(map(lambda ca: (start_activity_name, ca), success_activities))
        df = filter_eventually_follows_relation(df, relations, retain=False)

    # Find variants & sort by prefix length (less ko_activities start to end: possible indicator of knockout)
    variants = get_sorted_variants(df)

    # Ideas:
    # 1) most frequent differentiating transition, can be indicator of activity that triggered KO?
    # 2) last activity / outcome of shortest variants indicates negative outcome/cancellation?
    for v_i in range(0, len(variants)):
        p1 = variants[v_i]['prefix']
        diffs = {}
        for v_j in range(0, len(variants)):
            p2 = variants[v_j]['prefix']
            diffs[f"{v_j}"] = []

            # for every transition in current variant, check: is this transition present in the other variant?
            for e1, e2 in zip(p1, p1[1:]):
                present = False
                for x, y in zip(p2, p2[1:]):
                    if (e1, e2) == (x, y):
                        present = True
                        break
                if not present:
                    diffs[f"{v_j}"].append((e1, e2))  # transition was not present, add to detected differences

        variants[v_i]['diffs'] = diffs
        variants[v_i]['most_frequent_differentiating_transition'] = find_most_frequent_tuple(diffs)

    ko_sequences = get_possible_ko_sequences(variants, limit)
    # sort by prefix length ASC (shorter variants = more likely to feature knock-out)
    ko_sequences = dict(sorted(ko_sequences.items(), key=lambda item: item[1][2]))

    ko_outcomes = {}

    # Report results
    if not quiet:
        # print_characteristic_transitions(variants)
        print(f"\nPossible knockout sequences:\n")

    for key in ko_sequences.keys():
        if not quiet:
            print(f"{key:>40} -> {ko_sequences[key][0]:<30} | outcome: {ko_sequences[key][1]:<35}")

        outcome_key = ko_sequences[key][1]
        ko_outcomes[outcome_key] = ko_outcomes.get(outcome_key, 0) + 1

    ko_activities = list(ko_sequences.keys())

    if len(known_ko_activities) > 0:
        ko_activities.extend(known_ko_activities)

    # remove possible duplicates
    ko_activities = list(set(ko_activities))

    if len(post_knockout_activities) > 0:
        return ko_activities, post_knockout_activities, ko_sequences
    else:
        return ko_activities, list(ko_outcomes.keys()), ko_sequences
