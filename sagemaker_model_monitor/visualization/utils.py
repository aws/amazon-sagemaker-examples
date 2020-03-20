from enum import Enum
import warnings
import re
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib import pyplot as plt
plt.style.use('seaborn-muted')


##### TABLE

def group_by_feature(baseline_statistics, latest_statistics, violations):
    features = {}
    # add baseline statistics
    if baseline_statistics:
        for baseline_feature in baseline_statistics['features']:
            feature_name = baseline_feature['name']
            if feature_name not in features:
                features[feature_name] = {}
            features[feature_name]['baseline'] = baseline_feature
    # add latest statistics
    if latest_statistics:
        for latest_feature in latest_statistics['features']:
            feature_name = latest_feature['name']
            if feature_name not in features:
                features[feature_name] = {}
            features[feature_name]['latest'] = latest_feature
    # add violations
    if violations:
        for violation in violations:
            feature_name = violation['feature_name']
            if feature_name not in features:
                features[feature_name] = {}
            if 'violations' in features[feature_name]:
                features[feature_name]['violations'] += [violation]
            else:
                features[feature_name]['violations'] = [violation]
    return features


def violation_exists(feature, check_type):
    if 'violations' in feature:
        if check_type in set([v['constraint_check_type'] for v in feature['violations']]):
            return True
    return False


def create_data_type_df(feature_names, features):
    columns = ['data_type']
    rows = []
    rows_style = []
    for feature_name in feature_names:
        feature = features[feature_name]
        latest = feature['latest']['inferred_type']
        violation = violation_exists(feature, 'data_type_check')
        rows.append([latest])
        rows_style.append([violation])
    df = pd.DataFrame(rows, index=feature_names, columns=columns)
    df_style = pd.DataFrame(rows_style, index=feature_names, columns=columns)
    return df, df_style


def get_completeness(feature):
    if feature['inferred_type'] in set(['Fractional', 'Integral']):
        common = feature['numerical_statistics']['common']
    elif feature['inferred_type'] == 'String':
        common = feature['string_statistics']['common']
    else:
        raise ValueError('Unknown `inferred_type` {}.'.format(feature['inferred_type']))
    num_present = common['num_present']
    num_missing = common['num_missing']
    completeness = num_present / (num_present + num_missing)
    return completeness


def create_completeness_df(feature_names, features):
    columns = ['completeness']
    rows = []
    rows_style = []
    for feature_name in feature_names:
        feature = features[feature_name]
        latest = get_completeness(feature['latest'])
        violation = violation_exists(feature, 'completeness_check')
        rows.append([latest])
        rows_style.append([violation])
    df = pd.DataFrame(rows, index=feature_names, columns=columns)
    df_style = pd.DataFrame(rows_style, index=feature_names, columns=columns)
    return df, df_style


def get_baseline_drift(feature):
    if 'violations' in feature:
        for violation in feature['violations']:
            if violation['constraint_check_type'] == 'baseline_drift_check':
                desc = violation['description']
                matches = re.search('distance: (.+) exceeds', desc)
                if matches:
                    match = matches.group(1)
                    return float(match)
    return np.nan


def create_baseline_drift_df(feature_names, features):
    columns = ['baseline_drift']
    rows = []
    rows_style = []
    for feature_name in feature_names:
        feature = features[feature_name]
        latest = get_baseline_drift(feature)
        violation = violation_exists(feature, 'baseline_drift_check')
        rows.append([latest])
        rows_style.append([violation])
    df = pd.DataFrame(rows, index=feature_names, columns=columns)
    df_style = pd.DataFrame(rows_style, index=feature_names, columns=columns)
    return df, df_style


def get_categorical_values(feature):
    if 'violations' in feature:
        for violation in feature['violations']:
            if violation['constraint_check_type'] == 'categorical_values_check':
                desc = violation['description']
                matches = re.search('Value: (.+) does not meet the constraint requirement!', desc)
                if matches:
                    match = matches.group(1)
                    return float(match)
    return np.nan


def create_categorical_values_df(feature_names, features):
    columns = ['categorical_values']
    rows = []
    rows_style = []
    for feature_name in feature_names:
        feature = features[feature_name]
        latest = get_categorical_values(feature)
        violation = violation_exists(feature, 'categorical_values_check')
        rows.append([latest])
        rows_style.append([violation])
    df = pd.DataFrame(rows, index=feature_names, columns=columns)
    df_style = pd.DataFrame(rows_style, index=feature_names, columns=columns)
    return df, df_style


def create_violation_df(baseline_statistics, latest_statistics, violations):
    features = group_by_feature(baseline_statistics, latest_statistics, violations)
    feature_names = list(features.keys())
    feature_names.sort()
    data_type_df, data_type_df_style = create_data_type_df(feature_names, features)
    completeness_df, completeness_df_style = create_completeness_df(feature_names, features)
    baseline_drift_df, baseline_drift_df_style = create_baseline_drift_df(feature_names, features)
    categorical_values_df, categorical_values_df_style = create_categorical_values_df(feature_names, features)
    df = pd.concat([data_type_df, completeness_df, baseline_drift_df, categorical_values_df], axis=1)
    df_style = pd.concat([data_type_df_style, completeness_df_style, baseline_drift_df_style, categorical_values_df_style], axis=1)
    return df, df_style


def style_violation_df(df, df_style):
    
    def all_white(df):
        attr = 'background-color: white'
        return pd.DataFrame(attr, index=df.index, columns=df.columns)
    
    def highlight_failed_row(df):
        nonlocal df_style
        df_style_cp = df_style.copy()
        values = df_style_cp.values.any(axis=1, keepdims=True) * np.ones_like(df_style)
        df_style_cp = pd.DataFrame(values, index=df.index, columns=df.columns)
        df_style_cp = df_style_cp.replace(to_replace=True, value='background-color: #fff7dc')
        df_style_cp = df_style_cp.replace(to_replace=False, value='')
        return df_style_cp
    
    def highlight_failed(df):
        nonlocal df_style
        df_style_cp = df_style.copy()
        df_style_cp = df_style_cp.replace(to_replace=True, value='background-color: orange')
        df_style_cp = df_style_cp.replace(to_replace=False, value='')
        return df_style_cp
    
    def style_percentage(value):
        if np.isnan(value):
            return 'N/A'
        else:
            return '{:.2%}'.format(value)
        
    for column_name in ['completeness', 'baseline_drift', 'categorical_values']:
        df[column_name] = df[column_name].apply(style_percentage)
    
    return df.style\
        .apply(all_white, axis=None)\
        .apply(highlight_failed_row, axis=None)\
        .apply(highlight_failed, axis=None)


def show_violation_df(baseline_statistics, latest_statistics, violations):
    violation_df, violation_df_style = create_violation_df(baseline_statistics, latest_statistics, violations)
    return style_violation_df(violation_df, violation_df_style)


##### VISUALIZATION

def get_features(raw_data):
    return {feature['name']: feature for feature in raw_data['features']}

def show_distributions(features, baselines=None):
    string_features = [name for name, feature in features.items() if FeatureType(feature['inferred_type']) == FeatureType.STRING]
    numerical_features = [name for name, feature in features.items() if name not in string_features]
    numerical_table = pd.concat([_summary_stats(features[feat]) for feat in numerical_features], axis=0) if numerical_features else None
    string_table = pd.concat([_summary_stats(features[feat]) for feat in string_features], axis=0) if string_features else None
    if numerical_features:
        display(HTML("<h3>{msg}</h3>".format(msg="Numerical Features")))
        display(numerical_table)
        _display_charts(_get_charts(features, numerical_features, baselines))
    if string_features:
        display(HTML("<h3>{msg}</h3>".format(msg="String Features")))
        display(string_table)
        _display_charts(_get_charts(features, string_features, baselines), numerical=False)

def _display_charts(chart_tables, ncols=5, numerical=True):
    nrows = int(np.ceil(len(chart_tables)/ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 4*nrows))
    for i, chart_table in enumerate(chart_tables):
        row, col = i//5, i%5
        curr_ax = ax[row][col] if nrows > 1 else ax[col]
        opacity = 0.7
        if numerical:
            c = chart_table[0].sort_values(by=["lower_bound"])
            c_width = c.upper_bound.values[0] - c.lower_bound.values[0]
            pos_c = 0.5 * (c.upper_bound.values + c.lower_bound.values)
        
        else:
            c = chart_table[0].sort_values(by=["frequency"], ascending=False).iloc[:10] if len(chart_table[0]) > 10 else chart_table[0].sort_values(by=["frequency"], ascending=False)
            c_width = 0.35
            pos_c = np.arange(len(c.value.values))
        
        curr_ax.bar(pos_c, c.frequency, c_width, label='collected', alpha=opacity)
        
        if len(chart_table) > 1: #also includes baseline stats info
            if numerical:
                b = chart_table[1].sort_values(by=["lower_bound"])
                b_width = b.upper_bound.values[0] - b.lower_bound.values[0]
                pos_b = 0.5 * (b.upper_bound.values + b.lower_bound.values)
                curr_ax.bar(pos_b, b.frequency, b_width, label='baseline', alpha=opacity)
                
            else:
                b = c.merge(chart_table[1], how='left', on=['value'])
                b_width = 0.35
                pos_b = np.arange(len(b.value.values)) + b_width
                curr_ax.bar(pos_b, b.frequency_y, b_width, label='baseline', alpha=opacity)
                
            curr_ax.legend()
        
        if not numerical:
            curr_ax.set_xticks(pos_c + c_width/2)
            curr_ax.set_xticklabels([label[:10] if len(label) > 10 else label for label in c.value.values], )
            [(tick.set_rotation(90), tick.set_fontsize(8)) for tick in curr_ax.get_xticklabels()]
        curr_ax.set_xlabel(c.key.values[0])
    plt.ylabel('Frequency')
    if ncols*nrows != len(chart_tables):
        [a.set_visible(False) for a in ax.flat[-(ncols*nrows-len(chart_tables)):]]
    plt.show();
    
def _get_charts(features, feature_types, baselines=None):
    charts = [(_extract_dist(features[feat]), _extract_dist(baselines[feat])) for feat in feature_types] if baselines is not None else [(_extract_dist(features[feat]),) for feat in feature_types]
    return [chart for chart in charts if not chart[0].empty]
    
def _extract_dist(feature_dict):
    try:
        stats_key = 'string_statistics' if FeatureType(feature_dict['inferred_type']) == FeatureType.STRING else 'numerical_statistics'
        distribution_type = 'categorical' if FeatureType(feature_dict['inferred_type']) == FeatureType.STRING else 'kll'
        table = pd.DataFrame(feature_dict[stats_key]['distribution'][distribution_type]['buckets'])
        table['frequency'] = table['count']/table['count'].sum()
        table['key'] = [feature_dict['name']]*len(table)
    except KeyError:
        table = pd.DataFrame()
    return table

def _summary_stats(feature_dict):
    stats_key = 'string_statistics' if FeatureType(feature_dict['inferred_type']) == FeatureType.STRING else 'numerical_statistics'
    common = pd.DataFrame(feature_dict[stats_key]['common'], index=[feature_dict['name']])
    specific = pd.DataFrame({k:v for k,v in feature_dict[stats_key].items() if k != 'common' and k != 'distribution'},
                                            index=[feature_dict['name']])
    return pd.concat([common, specific], axis=1)


class FeatureType(Enum):
    INTEGRAL = "Integral"
    FRACTIONAL = "Fractional"
    STRING = "String"
    UNKNOWN = "Unknown"
        
        