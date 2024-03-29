# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00a_EDA_plotting_features_and_events.ipynb.

# %% auto 0
__all__ = ['IN', 'SMALL_RAW', 'steps_in_day', 'first_n_rows', 'RAW', 'plot', 'plot_sid']

# %% ../nbs/00a_EDA_plotting_features_and_events.ipynb 3
from fastcore.all import Path
import pandas as pd
import matplotlib.pyplot as plt

# %% ../nbs/00a_EDA_plotting_features_and_events.ipynb 4
IN = Path('../input')
SMALL_RAW = Path('../input_small/child-mind-institute-detect-sleep-states')
SMALL_RAW.mkdir(exist_ok=True, parents=True)
steps_in_day = 12 * 60 * 24 # based on steps being 5 seconds
first_n_rows = 10_000_000

# %% ../nbs/00a_EDA_plotting_features_and_events.ipynb 6
RAW = IN/'child-mind-institute-detect-sleep-states' 

# %% ../nbs/00a_EDA_plotting_features_and_events.ipynb 31
def plot(df_series, df_events, title='time_chunk', extra_cols=[]): 
    fig, ax = plt.subplots(figsize=(15, 5))
    df_series['anglez'].plot(ax=ax, alpha=.3, color='b')
    ax.set_ylabel('anglez', color='b')
    handles = [ax.get_legend_handles_labels()[0][-1]]
    labels = ['anglez']
    for col, color, alpha in extra_cols: 
        df_series[col].plot(ax=ax, alpha=alpha, color=color)
    ax1 = ax.twinx()
    df_series['enmo'].plot(ax=ax1, color='g', alpha=.3)
    ax1.set_ylabel('enmo', color='g')
    
    noon = df_series[df_series.is_noon]
    steps = [(df_series.index[0] - 1, 'noon')]
    for i, step in enumerate(noon.index): 
        steps.append((step, 'noon'))
        ax.axvline(step, color='black', alpha=1, linestyle='-.', label='noon')
        if i == 0: 
            handles.append(ax.get_legend_handles_labels()[0][-1])
            labels.append('noon')
    for i, step in enumerate(df_events.query('event == "onset"').index): 
        steps.append((step, 'onset'))
        ax.axvline(step, color='r', alpha=1, linestyle='-', label='onset')
        if i == 0: 
            handles.append(ax.get_legend_handles_labels()[0][-1])
            labels.append('onset')
    for i, step in enumerate(df_events.query('event == "wakeup"').index): 
        steps.append((step, 'wakeup'))
        ax.axvline(step, color='orange', alpha=1, linestyle='-', label='wakeup')
        if i == 0: 
            handles.append(ax.get_legend_handles_labels()[0][-1])
            labels.append('wakeup')
    steps = sorted(steps)
    for i, (step, kind) in enumerate(steps[1:], start=1): 
        if kind == 'noon': 
            if steps[i - 1][1] == 'onset': 
                ax.axvline(step - steps_in_day // 4, color='purple', alpha=.6, 
                           linestyle='-', label='no_wakeup', linewidth=15.0)
                handles.append(ax.get_legend_handles_labels()[0][-1])
                labels.append('no_wakeup')
            if steps[i - 1][1] == 'noon': 
                ax.axvline(step - steps_in_day // 2, color='brown', alpha=.6, 
                           linestyle='-', label='no_onset', linewidth=15.0)
                handles.append(ax.get_legend_handles_labels()[0][-1])
                labels.append('no_onset')
    ax.legend(loc='upper left', handles=handles, labels=labels)
    ax1.legend()
    plt.suptitle(title)
    plt.show()

def plot_sid(sid = '0ce74d6d2106', days_per_graph = 3, return_dfs=False, extra_cols=[]):
    df = trs[(trs.series_id == sid)]
    df = df.assign(is_noon=df.timestamp.str[11:19] == '12:00:00',
                  mod_step=df.step % steps_in_day).set_index('step')
    dfe_with_nans = tre[(tre.series_id == sid)]
    nights = dfe_with_nans.night.max()
    print('', '*' * 100, '\n', '*' * 38, sid, nights, 'nights', '*' * 38, '\n', '*' * 100)
    dfe = dfe_with_nans.query('step >= 0')
    dfe = dfe.assign(step = dfe.step.astype(int)).set_index('step')
    first_noon_index = int(df[df.is_noon].index[0])
    if first_noon_index > 0:
        df_series = df.iloc[:first_noon_index]
        df_events = dfe.join(df_series[[]], how='inner') 
        plot(df_series, df_events, title=f'sid {sid}: data until first noon')

    for i, chunk in enumerate(range(first_noon_index + 1, len(df), days_per_graph * steps_in_day)): 
        total_chunks = len(df) // (days_per_graph * steps_in_day) + 1
        df_series = df.iloc[chunk: chunk + days_per_graph * steps_in_day]
        df_events = dfe.join(df_series[[]], how='inner')  
        plot(df_series, df_events, extra_cols=extra_cols, 
             title = f'sid {sid}: chunk {i + 1} of {total_chunks}: interval {days_per_graph} days')
    if return_dfs: return df, dfe_with_nans
