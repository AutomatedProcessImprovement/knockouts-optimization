import logging
import traceback

import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from knockout_ios.pipeline_wrapper import Pipeline
from knockout_ios.utils.constants import globalColumnNames
from knockout_ios.utils.custom_exceptions import KnockoutsDiscoveryException, InvalidFileExtensionException, \
    KnockoutRuleDiscoveryException
from knockout_ios.utils.ui import get_rectangle_from_rule

SMALL_FONT = 6
plt.rc('font', size=SMALL_FONT)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_FONT)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_FONT)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_FONT)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_FONT)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_FONT)  # legend fontsize
plt.rc('figure', titlesize=SMALL_FONT)  # fontsize of the figure title

st.set_page_config(
    page_title="Knockouts Redesign Tool",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/AutomatedProcessImprovement/knockouts-redesign/issues',
        'Report a bug': "https://github.com/AutomatedProcessImprovement/knockouts-redesign/issues",
        'About': "This project is about discovering improvement opportunities in Knock-out checks performed within Business Processes."
    }
)

initial_state = [("config_dir", "config"), ("config_file_name", "synthetic_example.json"),
                 ("cache_dir", "cache/synthetic_example"),
                 ("pipeline", None), ('log_activities', []), ('log_attributes', []), ('ko_redesign_adviser', None),
                 ('attributes_to_consider', []), ('known_ko_activities', []), ('post_ko_activities', []),
                 ]

for v in initial_state:
    if v[0] not in st.session_state:
        st.session_state[v[0]] = v[1]


def reset_state(fields_to_keep=None):
    if fields_to_keep is None:
        fields_to_keep = []

    for v in initial_state:
        if v[0] in fields_to_keep:
            continue
        st.session_state[v[0]] = v[1]


def load_log(config_dir, config_file_name, cache_dir):
    reset_state(fields_to_keep=["config_file_name"])

    pipeline = Pipeline(config_dir=config_dir,
                        config_file_name=config_file_name,
                        cache_dir=cache_dir)
    try:
        pipeline.read_log_and_config()

        log_activities = pipeline.log_df[globalColumnNames.SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME].unique()
        log_attributes = pipeline.log_df.columns.values.tolist()

        # clean up attributes to be shown, only keep attributes that are not in globalColumnNames
        log_attributes = [x for x in log_attributes if
                          not x.startswith("@@")]
        cols = [getattr(globalColumnNames, attr) for attr in dir(globalColumnNames)]
        log_attributes = [x for x in log_attributes if
                          x not in cols]

        return pipeline, log_activities, log_attributes

    except FileNotFoundError:
        logging.error(traceback.format_exc())
        st.error("Config file not found")


def load_log_wrapper():
    try:
        with st.spinner('Reading Log and Config...'):
            st.session_state['pipeline'], st.session_state['log_activities'], st.session_state[
                'log_attributes'] = load_log(config_dir=st.session_state['config_dir'],
                                             config_file_name=st.session_state['config_file_name'],
                                             cache_dir=st.session_state['cache_dir'])

            # Initially assume user wants to consider all attributes in the analysis
            st.session_state['attributes_to_consider'] = st.session_state['log_attributes']

            # Reflect any info already provided via the config file
            if not (st.session_state['pipeline'].config.attributes_to_ignore is None):
                st.session_state['attributes_to_consider'] = [x for x in st.session_state['attributes_to_consider'] if
                                                              x not in st.session_state[
                                                                  'pipeline'].config.attributes_to_ignore]

            st.session_state['known_ko_activities'] = st.session_state['pipeline'].config.known_ko_activities
            st.session_state['post_ko_activities'] = st.session_state['pipeline'].config.post_knockout_activities
            st.session_state['success_activities'] = st.session_state['pipeline'].config.success_activities

            fields = {'attributes_to_consider': st.session_state['attributes_to_consider'],
                      'known_ko_activities': st.session_state['known_ko_activities'],
                      'post_ko_activities': st.session_state['post_ko_activities']}
            logging.info(f"Read from config: {fields}")

    except InvalidFileExtensionException:
        logging.error(traceback.format_exc())
        st.error("Invalid file extension")
    except Exception:
        logging.error(traceback.format_exc())
        st.error("Error running analysis. Check the console for details.")


def run_analysis():
    if st.session_state['pipeline'] is None:
        return

    _ko_redesign_adviser, ko_analysis_report, ko_redesign_reports, ko_rule_discovery_stats, redesign_options, low_confidence_warnings = \
        st.session_state[
            'pipeline'].run_analysis()

    return _ko_redesign_adviser, ko_analysis_report, ko_redesign_reports, ko_rule_discovery_stats, redesign_options, low_confidence_warnings


def run_analysis_wrapper():
    with st.spinner('Running pipeline...'):
        try:
            st.session_state[
                'ko_redesign_adviser'], ko_analysis_report, ko_redesign_report, ko_rule_discovery_stats, redesign_options, low_confidence_warnings = run_analysis()

            if not (st.session_state["ko_redesign_adviser"] is None):
                st.markdown("### Knockouts Analysis")
                st.table(ko_analysis_report)

                for warning in low_confidence_warnings:
                    st.warning(warning)

                # Show rule discovery metrics in a container collapsed by default
                with st.expander("See rule discovery stats", expanded=False):
                    st.json(ko_rule_discovery_stats)

                st.markdown("### Reordering Options")

                st.markdown(ko_redesign_report['dependencies'].to_markdown())
                st.text(" \n")
                st.write("Optimal Order of Knock-out checks (taking into account attribute dependencies):")
                st.write(ko_redesign_report['reordering'])

                st.text(" \n")
                st.markdown("### Relocation Options")
                st.markdown(ko_redesign_report['relocation'].to_markdown(), unsafe_allow_html=True)

                st.text(" \n")
                st.markdown("### Rule-change Options (numerical attributes)")
                # Plot distributions of numerical attributes appearing in each rule,
                # and overlay the range captured by rules

                rule_change = redesign_options["rule_change"]
                aggregated_df = st.session_state['ko_redesign_adviser'].knockout_analyzer.rule_discovery_log_df
                classifiers = st.session_state['ko_redesign_adviser'].knockout_analyzer.RIPPER_rulesets

                if classifiers is None:
                    classifiers = st.session_state['ko_redesign_adviser'].knockout_analyzer.IREP_rulesets

                for ko_activity in redesign_options["rule_change"]:
                    st.markdown(f"#### {ko_activity}")
                    try:

                        fig, ax = plt.subplots(len(rule_change[ko_activity]), 1, squeeze=True)
                        ruleset = classifiers[ko_activity][0].ruleset_

                        for attribute in rule_change[ko_activity]:
                            hist = sns.histplot(aggregated_df, x=attribute, ax=ax)
                            ax.set(ylabel='Cases')

                            # Overlay the range of current attribute captured by the rules of current ko activity
                            for rule in ruleset.rules:
                                rectangles = get_rectangle_from_rule(rule, hist.dataLim, hist.viewLim, attribute)
                                for rect in rectangles:
                                    hist.add_patch(rect)

                            legend_elements = [
                                Patch(facecolor="#5799C6",
                                      label=f'Values of \"{attribute}\" across cases'),
                                Patch(facecolor='red', alpha=0.1, label='Knocked-out cases')]

                            ax.legend(handles=legend_elements, loc='upper right')

                        fig.tight_layout(rect=(0, 0, 1, 0.35))
                        st.pyplot(fig)

                    except:
                        st.write("No numerical attributes found in the rules of this activity")

        except KnockoutRuleDiscoveryException:
            logging.error(traceback.format_exc())
            st.error("Error discovering Knockout Rules. Check the console for details.")
        except KnockoutsDiscoveryException:
            logging.error(traceback.format_exc())
            st.error("Error discovering Knockout Activities. Check the console for details.")
        except Exception:
            logging.error(traceback.format_exc())
            st.error("Error running analysis. Check the console for details.")


with st.sidebar:
    st.title('Knockouts Redesign Tool')

    config_file_name = st.sidebar.text_input("Config file", key="config_file_name", on_change=load_log_wrapper)

    if len(st.session_state['log_activities']) > 0:
        known_ko_activities = st.multiselect(
            'Known knockout activities',
            options=st.session_state["log_activities"],
            key="known_ko_activities",
            on_change=lambda: st.session_state["pipeline"].update_known_ko_activities(
                st.session_state['known_ko_activities']))

        post_ko_activities = st.multiselect(
            'Post-knockout activities',
            options=st.session_state["log_activities"],
            key="post_ko_activities",
            on_change=lambda: st.session_state["pipeline"].update_post_ko_activities(
                st.session_state['post_ko_activities']))

        success_activities = st.multiselect(
            'Success activities',
            options=st.session_state["log_activities"],
            key="success_activities",
            on_change=lambda: st.session_state["pipeline"].update_success_activities(
                st.session_state['success_activities']))

    if len(st.session_state['log_attributes']) > 0:
        log_attributes = st.multiselect(
            'Case attributes to consider',
            options=st.session_state["log_attributes"],
            key="attributes_to_consider",
            on_change=lambda: st.session_state["pipeline"].update_log_attributes(
                st.session_state['attributes_to_consider']))

    col1, col2 = st.columns(2)

    with col1:
        st.button('Load Log', on_click=load_log_wrapper)

    with col2:
        st.button('Run Pipeline', disabled=(st.session_state["pipeline"] is None), on_click=run_analysis_wrapper)
