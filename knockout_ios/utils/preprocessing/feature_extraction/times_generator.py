# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:51:24 2020

@author: Manuel Camargo
"""
import itertools
from datetime import datetime

from operator import itemgetter
import numpy as np
import pandas as pd

from .extraction import role_discovery as rl
from .calc import intercase_features_calculator as it

from sklearn.preprocessing import MaxAbsScaler


class TimesGenerator():
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, log_df, parms):
        """constructor"""
        self.log = log_df
        self.parms = parms


    def _add_intercases(self):
        """Appends the indexes and relative time to the dataframe.
        parms:
            log: dataframe.
        Returns:
            Dataframe: The dataframe with the calculated features added.
        """
        log = pd.DataFrame(self.log)

        res_analyzer = rl.ResourcePoolAnalyser(
            log,
            sim_threshold=self.parms['rp_similarity'])
        resource_table = pd.DataFrame.from_records(res_analyzer.resource_table)
        resource_table.rename(columns={'resource': 'user'}, inplace=True)
        
        self.roles = {role: group.user.to_list() for role, group in resource_table.groupby('role')}
        log = log.merge(resource_table, on='user', how='left')
        inter_mannager = it.IntercaseMannager(log, 
                                              self.parms['all_r_pool'],
                                              self.parms['model_type'])
        log, mean_states = inter_mannager.fit_transform()
        self.mean_states = mean_states
        self.log = log

        roles_table = (self.log[['caseid', 'role', 'task']]
                .groupby(['task', 'role']).count()
                .sort_values(by=['caseid'])
                .groupby(level=0)
                .tail(1)
                .reset_index())
        self.roles_table = roles_table[['role', 'task']]
        
        return log, res_analyzer

    def _add_calculated_times(self, log):
        """Appends the indexes and relative time to the dataframe.
        parms:
            log: dataframe.
        Returns:
            Dataframe: The dataframe with the calculated features added.
        """
        log['daytime'] = 0
        log = log.to_dict('records')
        log = sorted(log, key=lambda x: x['caseid'])
        for _, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            events = sorted(events, key=itemgetter('start_timestamp'))
            for i in range(0, len(events)):
                time = events[i]['start_timestamp'].time()
                time = time.second + time.minute*60 + time.hour*3600
                events[i]['st_daytime'] = time
                events[i]['st_weekday'] = events[i]['start_timestamp'].weekday()
                events[i]['st_month'] = events[i]['start_timestamp'].month
                if self.parms['model_type'] == 'dual_inter':
                    time = events[i]['end_timestamp'].time()
                    time = time.second + time.minute*60 + time.hour*3600
                    events[i]['end_daytime'] = time
                    events[i]['end_weekday'] = events[i]['end_timestamp'].weekday()
                    events[i]['end_month'] = events[i]['end_timestamp'].month
        return pd.DataFrame.from_dict(log)

    def _transform_features(self):
        # scale continue features
        cols = ['processing_time', 'waiting_time']
        self.scaler = MaxAbsScaler()
        self.scaler.fit(self.log_train[cols])
        self.log_train[cols] = self.scaler.transform(self.log_train[cols])
        self.log_valdn[cols] = self.scaler.transform(self.log_valdn[cols])
        # scale intercase
        # para que se ajuste a los dos sets requeridos para el modelo dual
        if self.parms['model_type'] in ['inter', 'dual_inter', 'inter_nt']:
            inter_feat = ['st_wip', 'st_tsk_wip']
            self.inter_scaler = MaxAbsScaler()
            self.inter_scaler.fit(self.log_train[inter_feat])
            self.log_train[inter_feat] = (
                self.inter_scaler.transform(self.log_train[inter_feat]))
            self.log_valdn[inter_feat] = (
                self.inter_scaler.transform(self.log_valdn[inter_feat]))
            cols.extend(inter_feat)
            if self.parms['model_type'] in ['dual_inter']:
                inter_feat = ['end_wip']
                self.end_inter_scaler = MaxAbsScaler()
                self.end_inter_scaler.fit(self.log_train[inter_feat])
                self.log_train[inter_feat] = (
                    self.end_inter_scaler.transform(self.log_train[inter_feat]))
                self.log_valdn[inter_feat] = (
                    self.end_inter_scaler.transform(self.log_valdn[inter_feat]))
                cols.extend(inter_feat)
        # scale daytime
        self.log_train['st_daytime'] = np.divide(self.log_train['st_daytime'], 86400)
        self.log_valdn['st_daytime'] = np.divide(self.log_valdn['st_daytime'], 86400)
        cols.extend(['caseid', 'ac_index', 'st_daytime', 'st_weekday'])
        if self.parms['model_type'] in ['dual_inter']:
            self.log_train['end_daytime'] = np.divide(self.log_train['end_daytime'], 86400)
            self.log_valdn['end_daytime'] = np.divide(self.log_valdn['end_daytime'], 86400)
            cols.extend(['end_weekday'])
        if self.parms['model_type'] in ['inter', 'dual_inter', 'inter_nt']:
            suffixes = (['_st_oc', '_end_oc'] if (
                self.parms['model_type']  in ['dual_inter']) else ['_st_oc'])
            if self.parms['all_r_pool']:
                for suffix in suffixes:
                    cols.extend([c_n for c_n in self.log_train.columns 
                                 if suffix in c_n])
            else:
                cols.extend(['rp'+x for x in suffixes])
        # Add next activity
        if self.parms['model_type'] in ['inter_nt', 'dual_inter']:
            cols.extend(['n_ac_index'])
        # filter features
        self.log_train = self.log_train[cols]
        self.log_valdn = self.log_valdn[cols]
        # fill nan values
        self.log_train = self.log_train.fillna(0)
        self.log_valdn = self.log_valdn.fillna(0)
