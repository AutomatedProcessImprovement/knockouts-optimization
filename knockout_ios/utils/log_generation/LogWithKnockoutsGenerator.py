import itertools
import random
from datetime import datetime
from random import randint

import numpy as np
from dateutil.relativedelta import relativedelta

import uuid

from knockout_ios.utils.log_generation import xes


def get_random_minutes():
    return int(random.uniform(30, 60))


class LogWithKnockoutsGenerator:

    def __init__(self, filename):
        self.filename = filename

    def generate_log(self, events=5000):

        demographic_values = ['demographic_type_1', 'demographic_type_2', 'demographic_type_3']
        vehicle_values = ['vehicle_type_1', 'vehicle_type_2', 'vehicle_type_3', 'vehicle_type_4']
        resources = ['Resource 1', 'Resource 2', 'Resource 3']
        case_attributes = ['Total Debt', 'Monthly Income', 'Loan Ammount', 'Demographic', 'Vehicle Owned']

        # Triplets: ko name, ko rule, rejection rate
        ko_checks = [('Check Liability', lambda trace: trace['Total Debt'] > 5000, 0.5),
                     ('Check Risk', lambda trace: trace['Loan Ammount'] > 10_000, 0.3),
                     ('Check Monthly Income', lambda trace: trace['Monthly Income'] < 1000, 0.2)]

        knocked_out_cases = []

        for check in ko_checks:
            activity_name = check[0]
            rate = check[2]
            number_of_kos = int(events * rate)

            events = max(0, events - number_of_kos)  # update "population"

            for _ in range(0, number_of_kos):
                attributes = {'Total Debt': str(randint(0, 4999)),
                              'Monthly Income': str(randint(1000, 5000)),
                              'Loan Ammount': str(randint(100, 9999)),
                              'Demographic': np.random.choice(demographic_values),
                              'Vehicle Owned': np.random.choice(vehicle_values),
                              }

                # Overwrite field according to KO rule we need to activate and build trace
                event_start = datetime.now()
                duration = get_random_minutes()
                trace = []

                if activity_name == 'Check Liability':
                    if random.random() < 0.5:
                        attributes['Total Debt'] = str(randint(5000, 30_000))
                    else:
                        # Only possible way to encode "Non existent vehicle", because None cannot be serialized
                        attributes['Vehicle Owned'] = ''

                    trace = [
                        {"concept:name": "Credit application received", "org:resource": np.random.choice(resources),
                         "start_timestamp": event_start.isoformat(),
                         "time:timestamp": (event_start + relativedelta(minutes=duration)).isoformat(), **attributes},
                        {"concept:name": "Check Liability", "org:resource": np.random.choice(resources),
                         "start_timestamp": (event_start + relativedelta(minutes=duration)).isoformat(),
                         "time:timestamp": (event_start + relativedelta(minutes=duration * 3)).isoformat(),
                         **attributes},
                        {"concept:name": "Notify Rejection", "org:resource": np.random.choice(resources),
                         "start_timestamp": (event_start + relativedelta(minutes=duration * 3)).isoformat(),
                         "time:timestamp": (event_start + relativedelta(minutes=duration * 4)).isoformat(),
                         **attributes},
                        {"concept:name": "Credit application processed", "org:resource": np.random.choice(resources),
                         "start_timestamp": (event_start + relativedelta(minutes=duration * 4)).isoformat(),
                         "time:timestamp": (event_start + relativedelta(minutes=duration * 5)).isoformat(),
                         **attributes}
                    ]

                elif activity_name == 'Check Risk':
                    attributes['Loan Ammount'] = str(randint(10_000, 30_000))

                    trace = [
                        {"concept:name": "Credit application received", "org:resource": np.random.choice(resources),
                         "start_timestamp": event_start.isoformat(),
                         "time:timestamp": (event_start + relativedelta(minutes=duration)).isoformat(), **attributes},
                        {"concept:name": "Check Liability", "org:resource": np.random.choice(resources),
                         "start_timestamp": (event_start + relativedelta(minutes=duration)).isoformat(),
                         "time:timestamp": (event_start + relativedelta(minutes=duration * 3)).isoformat(),
                         **attributes},
                        {"concept:name": "Check Risk", "org:resource": np.random.choice(resources),
                         "start_timestamp": (event_start + relativedelta(minutes=duration * 3)).isoformat(),
                         "time:timestamp": (event_start + relativedelta(minutes=duration * 4)).isoformat(),
                         **attributes},
                        {"concept:name": "Notify Rejection", "org:resource": np.random.choice(resources),
                         "start_timestamp": (event_start + relativedelta(minutes=duration * 4)).isoformat(),
                         "time:timestamp": (event_start + relativedelta(minutes=duration * 5)).isoformat(),
                         **attributes},
                        {"concept:name": "Credit application processed", "org:resource": np.random.choice(resources),
                         "start_timestamp": (event_start + relativedelta(minutes=duration * 5)).isoformat(),
                         "time:timestamp": (event_start + relativedelta(minutes=duration * 6)).isoformat(),
                         **attributes}
                    ]
                elif activity_name == 'Check Monthly Income':
                    attributes['Monthly Income'] = str(randint(0, 800))

                    trace = [
                        {"concept:name": "Credit application received", "org:resource": np.random.choice(resources),
                         "start_timestamp": event_start.isoformat(),
                         "time:timestamp": (event_start + relativedelta(minutes=duration)).isoformat(), **attributes},
                        {"concept:name": "Check Liability", "org:resource": np.random.choice(resources),
                         "start_timestamp": (event_start + relativedelta(minutes=duration)).isoformat(),
                         "time:timestamp": (event_start + relativedelta(minutes=duration * 3)).isoformat(),
                         **attributes},
                        {"concept:name": "Check Risk", "org:resource": np.random.choice(resources),
                         "start_timestamp": (event_start + relativedelta(minutes=duration * 3)).isoformat(),
                         "time:timestamp": (event_start + relativedelta(minutes=duration * 4)).isoformat(),
                         **attributes},
                        {"concept:name": "Check Monthly Income", "org:resource": np.random.choice(resources),
                         "start_timestamp": (event_start + relativedelta(minutes=duration * 4)).isoformat(),
                         "time:timestamp": (event_start + relativedelta(minutes=duration * 5)).isoformat(),
                         **attributes},
                        {"concept:name": "Notify Rejection", "org:resource": np.random.choice(resources),
                         "start_timestamp": (event_start + relativedelta(minutes=duration * 5)).isoformat(),
                         "time:timestamp": (event_start + relativedelta(minutes=duration * 6)).isoformat(),
                         **attributes},
                        {"concept:name": "Credit application processed", "org:resource": np.random.choice(resources),
                         "start_timestamp": (event_start + relativedelta(minutes=duration * 6)).isoformat(),
                         "time:timestamp": (event_start + relativedelta(minutes=duration * 7)).isoformat(),
                         **attributes}
                    ]

                knocked_out_cases.append(trace)

        # Regular cases
        regular_cases = []
        for _ in range(0, events):
            attributes = {'Total Debt': str(randint(0, 4000)),
                          'Monthly Income': str(1000 + randint(500, 2000)),
                          'Loan Ammount': str(randint(1000, 9000)),
                          'Demographic': np.random.choice(demographic_values),
                          'Vehicle Owned': np.random.choice(vehicle_values)}

            event_start = datetime.now()
            duration = get_random_minutes()

            regular_cases.append(
                [{"concept:name": "Credit application received", "org:resource": np.random.choice(resources),
                  "start_timestamp": event_start.isoformat(),
                  "time:timestamp": (event_start + relativedelta(minutes=duration)).isoformat(), **attributes},
                 {"concept:name": "Check Liability", "org:resource": np.random.choice(resources),
                  "start_timestamp": (event_start + relativedelta(minutes=duration)).isoformat(),
                  "time:timestamp": (event_start + relativedelta(minutes=duration * 3)).isoformat(),
                  **attributes},
                 {"concept:name": "Check Risk", "org:resource": np.random.choice(resources),
                  "start_timestamp": (event_start + relativedelta(minutes=duration * 3)).isoformat(),
                  "time:timestamp": (event_start + relativedelta(minutes=duration * 4)).isoformat(),
                  **attributes},
                 {"concept:name": "Check Monthly Income", "org:resource": np.random.choice(resources),
                  "start_timestamp": (event_start + relativedelta(minutes=duration * 4)).isoformat(),
                  "time:timestamp": (event_start + relativedelta(minutes=duration * 5)).isoformat(),
                  **attributes},
                 {"concept:name": "Make credit offer", "org:resource": np.random.choice(resources),
                  "start_timestamp": (event_start + relativedelta(minutes=duration * 5)).isoformat(),
                  "time:timestamp": (event_start + relativedelta(minutes=duration * 7)).isoformat(),
                  **attributes},
                 {"concept:name": "Credit application processed", "org:resource": np.random.choice(resources),
                  "start_timestamp": (event_start + relativedelta(minutes=duration * 7)).isoformat(),
                  "time:timestamp": (event_start + relativedelta(minutes=duration * 8)).isoformat(),
                  **attributes}
                 ]
            )

        traces = itertools.chain(knocked_out_cases, regular_cases)

        # Build Log & export XES
        log = xes.Log()
        for trace in traces:
            t = xes.Trace()
            t.add_attribute(xes.Attribute(type="string", key="concept:name", value=str(uuid.uuid1())))

            for event in trace:
                e = xes.Event()
                e.attributes = [
                    xes.Attribute(type="string", key="concept:name", value=event["concept:name"]),
                    xes.Attribute(type="string", key="org:resource", value=event["org:resource"]),
                    xes.Attribute(type="date", key="start_timestamp", value=event["start_timestamp"]),
                    xes.Attribute(type="date", key="time:timestamp", value=event["time:timestamp"]),
                    xes.Attribute(type="string", key='Total Debt', value=event['Total Debt']),
                    xes.Attribute(type="string", key='Monthly Income', value=event['Monthly Income']),
                    xes.Attribute(type="string", key='Loan Ammount', value=event['Loan Ammount']),
                    xes.Attribute(type="string", key='Demographic', value=event['Demographic']),
                    xes.Attribute(type="string", key='Vehicle Owned', value=event['Vehicle Owned']),
                ]
                t.add_event(e)
            log.add_trace(t)
        log.classifiers = [
            xes.Classifier(name="org:resource", keys="org:resource"),
            xes.Classifier(name="concept:name", keys="concept:name")
        ]

        try:
            open(self.filename, "w").write(str(log))
        except Exception as e:
            print(e)


if __name__ == "__main__":
    gen = LogWithKnockoutsGenerator("../../inputs/synthetic_example_raw.xes")
    gen.generate_log(1000)
