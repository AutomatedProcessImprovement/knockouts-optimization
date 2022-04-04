import collections
import pickle
from itertools import chain

import pandas as pd
from pm4py import filter_eventually_follows_relation
from pm4py.objects.log.obj import EventLog, EventStream
from pm4py.statistics.traces.generic.pandas import case_statistics

from knockout_ios.utils.constants import *


def extract_ko_config_fields(config):
    return [config.log_path,
            config.start_activity,
            ",".join(config.negative_outcomes),
            ",".join(config.known_ko_activities),
            config.ko_count_threshold
            ]


def config_hash_changed(config_1, config_2):
    h1 = hash(frozenset(extract_ko_config_fields(config_1)))
    h2 = hash(frozenset(extract_ko_config_fields(config_2)))
    return h1 != h2


def read_config_cache(config_file_name, cache_dir):
    binary_file = open(f'{cache_dir}/{config_file_name}_config', 'rb')
    config_cache = pickle.load(binary_file)
    binary_file.close()
    return config_cache


def dump_config_cache(config_file_name, config, cache_dir):
    binary_file = open(f'{cache_dir}/{config_file_name}_config', 'wb')
    pickle.dump(config, binary_file)
    binary_file.close()


def read_variants_cache(config_file_name, cache_dir):
    binary_file = open(f'{cache_dir}/{config_file_name}_variants', 'rb')
    variants = pickle.load(binary_file)
    binary_file.close()
    return variants


def dump_variants_cache(config_file_name, variants, cache_dir):
    binary_file = open(f'{cache_dir}/{config_file_name}_variants', 'wb')
    pickle.dump(variants, binary_file)
    binary_file.close()


def get_sorted_variants(df, caseid_col=PM4PY_CASE_ID_COLUMN_NAME, activity_col=PM4PY_ACTIVITY_COLUMN_NAME,
                        end_timestamp_col=PM4PY_END_TIMESTAMP_COLUMN_NAME):
    # Find variants & sort by prefix length

    variants_count = case_statistics.get_variant_statistics(df,
                                                            parameters=
                                                            {
                                                                case_statistics.Parameters.CASE_ID_KEY: caseid_col,
                                                                case_statistics.Parameters.ACTIVITY_KEY: activity_col,
                                                                case_statistics.Parameters.TIMESTAMP_KEY: end_timestamp_col
                                                            })

    variants_count = sorted(variants_count, key=lambda x: x[caseid_col], reverse=True)

    variants = list(map(lambda entry: {'prefix': entry['variant'].split(","), 'case_count': entry[caseid_col],
                                       'prefix_len': len(entry['variant'].split(","))}, variants_count))

    variants = sorted(variants, key=lambda e: e['prefix_len'])

    return variants


def find_most_frequent_tuple(tuples_dict):
    flat_values = list(chain(*tuples_dict.values()))

    common = collections.Counter(flat_values).most_common()

    if len(common) > 0:
        return collections.Counter(flat_values).most_common()[0]

    return ()


def print_characteristic_transitions(variants):
    for i in range(0, len(variants)):
        variant = variants[i]
        ko_info = variant['most_frequent_differentiating_transition']

        if len(ko_info) > 0:
            print(f"Variant {i} - prefix len: {variant['prefix_len']:>2} - cases: {variant['case_count']:>5} "
                  f"- characteristic: {'(' + ko_info[0][0] + ' -> ' + ko_info[0][1] + ')':75} "
                  f"- outcome: {variant['prefix'][-2]:>25}")


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


def discover_ko_sequences(df, config_file_name, cache_dir, limit=3, negative_outcomes=[], positive_outcomes=[],
                          known_ko_activities=[],
                          quiet=False, start_activity_name="Start", force_recompute=False):
    try:

        if force_recompute:
            raise FileNotFoundError

        if not quiet:
            print(f"Found cache for {config_file_name} variants")

        variants = read_variants_cache(config_file_name, cache_dir=cache_dir)
        variants = sorted(variants, key=lambda e: e['prefix_len'])

    except FileNotFoundError:

        if not quiet:
            print(f"Cache for {config_file_name} variants not found")

        if not isinstance(df, (pd.DataFrame, EventLog, EventStream)):
            raise Exception("the method can be applied only to a traditional event log!")

        if len(negative_outcomes) > 0:
            relations = list(map(lambda ca: (start_activity_name, ca), negative_outcomes))
            df = filter_eventually_follows_relation(df, relations)

        if len(positive_outcomes) > 0:
            relations = list(map(lambda ca: (start_activity_name, ca), positive_outcomes))
            df = filter_eventually_follows_relation(df, relations, retain=False)

        # Find variants & sort by prefix length (less activities start to end: possible indicator of knockout)
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
                        diffs[f"{v_j}"].append((e1, e2))  # transition was nor present, add to detected differences

            variants[v_i]['diffs'] = diffs
            variants[v_i]['most_frequent_differentiating_transition'] = find_most_frequent_tuple(diffs)

        dump_variants_cache(config_file_name, variants, cache_dir=cache_dir)

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

    if len(negative_outcomes) > 0:
        return ko_activities, negative_outcomes, ko_sequences
    else:
        return ko_activities, list(ko_outcomes.keys()), ko_sequences
