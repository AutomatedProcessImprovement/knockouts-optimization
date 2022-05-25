import pandas as pd
import streamlit as st

from knockout_ios.knockout_redesign_adviser import KnockoutRedesignAdviser
from knockout_ios.pipeline_wrapper import run_pipeline

config_dir = "config_examples"
config_file_name = "synthetic_example.json"
cache_dir = "cache/synthetic_example"
ko_redesign_adviser: KnockoutRedesignAdviser = None


@st.cache
def run_analysis():
    _ko_redesign_adviser = run_pipeline(config_dir=config_dir, config_file_name=config_file_name, cache_dir=cache_dir)
    return _ko_redesign_adviser


def load_log():
    pass


def preview():
    pass


with st.sidebar:
    # config_dir = st.sidebar.text_input("Config directory", config_dir)
    config_file_name = st.sidebar.text_input("Config file", config_file_name)
    # cache_dir = st.sidebar.text_input("Cache directory", cache_dir)

    known_ko_activities = st.multiselect(
        'Known knockout activities',
        ['Activity 1', 'Activity 2', 'Activity 3'],
        [])

    post_ko_activities = st.multiselect(
        'Post-knockout activities',
        ['Activity 1', 'Activity 2', 'Activity 3'],
        [])

    success_activities = st.multiselect(
        'Success activities',
        ['Activity 1', 'Activity 2', 'Activity 3'],
        [])

    case_attributes = st.multiselect(
        'Case attributes considered',
        ['Attr A', 'Attr B', 'Attr B'],
        ['Attr A', 'Attr B', 'Attr B'])

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Load Log'):
            load_log()
    with col2:
        if st.button('Run Pipeline'):
            ko_redesign_adviser = run_analysis()

if not (ko_redesign_adviser is None):
    data = pd.read_csv(ko_redesign_adviser.knockout_analyzer.report_file_name)
    st.table(data)
