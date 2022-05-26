import logging

import pandas as pd
import streamlit as st

from knockout_ios.pipeline_wrapper import Pipeline
from knockout_ios.utils.constants import globalColumnNames

initial_state = [("config_dir", "config_examples"), ("config_file_name", "envpermit.json"),
                 ("cache_dir", "cache"),
                 ("pipeline", None), ('log_activities', []), ('log_attributes', []), ('ko_redesign_adviser', None)]

for v in initial_state:
    if v[0] not in st.session_state:
        st.session_state[v[0]] = v[1]


def load_log(config_dir, config_file_name, cache_dir):
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

    except FileNotFoundError as e:
        logging.error(e)
        st.error("Config file not found")


def load_log_wrapper():
    try:
        st.session_state['pipeline'], st.session_state['log_activities'], st.session_state[
            'log_attributes'] = load_log(config_dir=st.session_state['config_dir'],
                                         config_file_name=st.session_state['config_file_name'],
                                         cache_dir=st.session_state['cache_dir'])

        # assume user wants to consider all attributes in the analysis
        st.session_state['attributes_to_consider'] = st.session_state['log_attributes']
        st.session_state['known_ko_activities'] = st.session_state['pipeline'].config.known_ko_activities
        st.session_state['post_ko_activities'] = st.session_state['pipeline'].config.post_knockout_activities
        st.session_state['success_activities'] = st.session_state['pipeline'].config.success_activities

    except TypeError:
        pass


# @st.cache(allow_output_mutation=False)
def run_analysis(pipeline):
    if pipeline is None:
        return

    _ko_redesign_adviser = pipeline.run_analysis()

    return _ko_redesign_adviser


def run_analysis_wrapper():
    try:
        st.session_state['ko_redesign_adviser'] = run_analysis(st.session_state['pipeline'])

        if not (st.session_state["ko_redesign_adviser"] is None):
            data = pd.read_csv(st.session_state["ko_redesign_adviser"].knockout_analyzer.report_file_name)
            st.table(data)
    except Exception as e:
        logging.error(e)
        st.error("Error running analysis. Check the console for details.")


with st.sidebar:
    config_file_name = st.sidebar.text_input("Config file", key="config_file_name")

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
