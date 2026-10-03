# Utilities for severe liver composite outcome experiments.
import os
import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

DEFAULT_LIVER_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "liver"


def _resolve_liver_data_dir(data_root=None):
    if data_root is not None:
        return Path(data_root)
    return Path(os.environ.get("LIVER_DATA_DIR", DEFAULT_LIVER_DATA_DIR))


def split_patient_ids(df_ts, df_labels, id_col='ID', test_size=0.2, random_state=42, use_stratify=True, stratify_col='has_event',
                        subset_frac=None, balance=False, balance_range=(0.8, 1.2), min_per_class=1, verbose=True):
    """
    返回 train_ids, test_ids 两个集合（患者ID集合）。
    - 若 use_stratify=True：按 df_labels['has_event'](或其他指定列) 做“患者级分层”
    - 否则：纯分组随机
    只对同时出现在 df_rawTS 与 df_labels 的 ID 切分（避免漏标签）

    可选：
      - subset_frac：先按患者级分层抽一个子集（保持原始类比例），再在子集上切分
      - balance：把（子集内的）各类按最小类为基准做近似均衡抽样
                 每个类目标数 = clamp( floor(U(0.8,1.2)*min_count), [min_per_class, 该类总数] )
    """
    ids_ts  = pd.Index(df_ts[id_col].unique())
    ids_lab = pd.Index(df_labels[id_col].unique())
    valid_ids = ids_ts.intersection(ids_lab)  # 仅对有标签的患者切分

    lab_sr = (df_labels.set_index(id_col)
                        .reindex(valid_ids)[stratify_col]
                        .fillna(0).astype(int))

    ids_arr = valid_ids.to_numpy()

    work_ids = ids_arr.copy()
    y_work = lab_sr.values.copy()
    rng = np.random.default_rng(random_state)

    ## 1) 子集（保持原始类比例）
    if subset_frac is not None and 0 < subset_frac < 1.0:
        sss_sub = StratifiedShuffleSplit(n_splits=1, train_size=subset_frac, random_state=random_state)
        sub_idx, _ = next(sss_sub.split(work_ids.reshape(-1, 1), y_work))
        work_ids = work_ids[sub_idx]
        y_work = y_work[sub_idx]

    ## 2) 近似均衡（以最小类为基准）
    if balance and len(np.unique(y_work)) > 0:
        by_class = {c: work_ids[y_work == c] for c in np.unique(y_work)}
        min_count = min(len(v) for v in by_class.values()) if by_class else 0
        lo, hi = balance_range

        chosen = []
        for c, arr in by_class.items():
            base_count = int(np.floor(rng.uniform(lo, hi) * max(min_count, 0)))
            base_count = max(min_per_class, min(base_count, len(arr)))
            if base_count > 0:
                chosen.append(rng.choice(arr, size=base_count, replace=False))

        work_ids = np.concatenate(chosen) if chosen else np.array([], dtype=ids_arr.dtype)
        y_work = lab_sr.reindex(work_ids).to_numpy()

    ## 3） 最终划分
    if use_stratify:
        # 如果某一类太少，stratify 会报错：fallback 到非分层
        uniq, cnt = np.unique(y_work, return_counts=True)
        min_cnt = cnt.min() if len(cnt) else 0
        if min_cnt < 2:
            if verbose:
                print(f"[split] stratify disabled (min class count={min_cnt})")
            use_stratify = False
    if use_stratify:
        # y = lab_sr.to_numpy()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr_idx, te_idx = next(sss.split(work_ids.reshape(-1,1), y_work))
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr_idx, te_idx = next(gss.split(work_ids, groups=work_ids))

    train_ids = set(work_ids[tr_idx])
    test_ids = set(work_ids[te_idx])

    print(f"Patients original total={len(valid_ids)} | used total={len(train_ids)+len(test_ids)}  | Train={len(train_ids)} | Test={len(test_ids)}")
    if use_stratify:
        print("Pos rate (has_event) — Train:",
              lab_sr.loc[list(train_ids)].mean(),
              " Test:", lab_sr.loc[list(test_ids)].mean())

    return train_ids, test_ids

def slice_by_ids(df_ts: pd.DataFrame, train_ids: set, test_ids: set, id_col='ID'):
    df_train_ts = df_ts[df_ts[id_col].isin(train_ids)].copy()
    df_test_ts  = df_ts[df_ts[id_col].isin(test_ids)].copy()
    # 安全检查：患者不重叠
    assert set(df_train_ts[id_col].unique()).isdisjoint(set(df_test_ts[id_col].unique()))
    return df_train_ts, df_test_ts


import ast

STAT_INDEX = {'first': 0, 'last': 1, 'mean': 2, 'max': 3, 'min': 4, 'count': 5}

def _extract_from_cell(cell, idx):
    # NaN 直接返回
    if pd.isna(cell):
        return np.nan
    # 已经是 tuple/list
    if isinstance(cell, (tuple, list)):
        return cell[idx] if len(cell) > idx else np.nan
    # 是字符串 "(...)" 的安全解析
    if isinstance(cell, str):
        try:
            t = ast.literal_eval(cell)
            if isinstance(t, (tuple, list)) and len(t) > idx:
                return t[idx]
        except Exception:
            pass
        return np.nan
    # 已经是标量（不是打包值），直接返回
    return cell

def expand_zipped_columns(df, feature_cols, selection, drop_original=True):
    """
    将 (first, last, mean, max, min, count) 解包成若干新列。
    selection 可以是：
      - 列表/元组：对所有 feature 都取同一批统计量，如 ('last','mean')
      - 字典：对不同列取不同统计量，如 {'Albumin':['last'], 'Platelets':['mean','max']}
    返回: (新df, 新的特征列名列表)
    """
    df_out = df.copy()
    flag = False
    # 统一成字典形式
    if isinstance(selection, dict):
        sel_map = {col: tuple(selection[col]) for col in feature_cols}
    else:
        stats = tuple(selection)
        sel_map = {col: stats for col in feature_cols}
        if len(stats) == 1:
            flag = True

    new_cols = []
    for col in feature_cols:
        stats_for_col = sel_map.get(col, ('last',))
        # 判断是否需要解包（列里是否是 tuple/list/字符串）
        sample = df_out[col].dropna()
        looks_zipped = False
        if len(sample) > 0:
            v = sample.iloc[0]
            looks_zipped = isinstance(v, (tuple, list, str))

        if looks_zipped:
            for stat in stats_for_col:
                idx = STAT_INDEX[stat]
                new_name = f"{col}_{stat}"
                df_out[new_name] = df_out[col].map(lambda x: _extract_from_cell(x, idx))
                new_cols.append(new_name)
            if drop_original:
                df_out.drop(columns=[col], inplace=True)
            if len(stats_for_col) == 1:
                df_out.rename(columns={new_name: col}, inplace=True)
        else:
            # 已经是数值列，直接保留
            new_cols.append(col)
    if flag:
        new_cols = feature_cols

    return df_out, new_cols


def apply_fill_method(window_data, fill_method, feature_cols):
    """根据指定方法填充缺失值"""
    window_data = window_data.copy()
    
    if fill_method == 'nan':
        pass
    elif fill_method == 'inf':
        window_data[feature_cols] = window_data[feature_cols].fillna(np.inf)
    elif fill_method == 'zero':
        window_data[feature_cols] = window_data[feature_cols].fillna(0)
    elif fill_method == 'forward_fill':
        window_data[feature_cols] = window_data[feature_cols].fillna(method='ffill')
    elif fill_method == 'backward_fill':
        window_data[feature_cols] = window_data[feature_cols].fillna(method='bfill')
    elif fill_method == 'mean':
        for col in feature_cols:
            mean_val = window_data[col].mean()
            if not np.isnan(mean_val):
                window_data[col] = window_data[col].fillna(mean_val)
    elif fill_method == 'median':
        for col in feature_cols:
            median_val = window_data[col].median()
            if not np.isnan(median_val):
                window_data[col] = window_data[col].fillna(median_val)
    
    return window_data

def create_sliding_windows(df, window_size, min_win_valid=1, fill_method='nan', id_col='ID', time_col='TimeUnit', patient_info_cols=None, event_date_cols=None,
                           preexpanded=True, features_override=None, agg_select=('first',),
                           df_demo=None, df_labels=None, earliest_date_col='earliest_date', time_horizon_months=6, return_extra=True,
                           balance_windows=False, balance_range=(0.8, 1.2), min_per_class=1, balance_random_state=42):
    """创建基于时间值的滑动窗口数据（从当前时间点向前取window_size个时间点）"""
    
    if not preexpanded:
        # —— 原来的做法：在函数里先解包
        raw_feature_cols = [c for c in df.columns if c not in [id_col, time_col, 'MonthPeriod']]
        df_expanded, feature_cols = expand_zipped_columns(df, raw_feature_cols, agg_select, drop_original=True)
    else:
        # —— 外部已解包/标准化：这里不做解包
        df_expanded = df
        if features_override is not None:
            feature_cols = list(features_override)
        else:
            feature_cols = [c for c in df.columns if c not in [id_col, time_col, 'MonthPeriod']]
    
    print(f"特征列: {feature_cols}")
    print(f"窗口大小: {window_size}")
    print(f"填充方法: {fill_method}")
    
    windowed_data = []
    win_meta = []  # 每项: {id_window, ID, last_valid_mp(Period), first_mp(Period), window_end_time}
    window_id = 0
    
    # 按患者分组处理
    for patient_id in df_expanded[id_col].unique():
        patient_data = df_expanded[df_expanded[id_col] == patient_id].sort_values(time_col).reset_index(drop=True)
        
        if 'MonthPeriod' in patient_data.columns:
            mp_series = patient_data['MonthPeriod']
            # 兼容 datetime/str：统一转 Period[M]
            if not isinstance(mp_series.dtype, pd.PeriodDtype):
                try:
                    mp_period = pd.PeriodIndex(pd.to_datetime(mp_series, errors='coerce'), freq='M')
                except Exception:
                    mp_period = pd.PeriodIndex([pd.NaT]*len(patient_data), freq='M')
            else:
                mp_period = mp_series
        else:
            # 没有 MonthPeriod 就无法按日期打标；用全 NaT
            mp_period = pd.PeriodIndex([pd.NaT]*len(patient_data), freq='M')

        first_mp = mp_period[~mp_period.isna()].min() if (~mp_period.isna()).any() else pd.NaT

        # print(f"处理患者 {patient_id}, 时间点: {patient_data[time_col].tolist()}")

        patient_time_min = int(patient_data[time_col].min())
        patient_time_max = int(patient_data[time_col].max())

        # 为每个时间点创建滑动窗口
        for i, current_row in patient_data.iterrows():
            current_time = current_row[time_col]
            
            # 从当前时间点开始，取window_size个时间点
            # start_time = current_time
            # end_time = current_time + window_size - 1
            start_time = current_time - window_size + 1
            end_time = current_time

            
            # print(f"  时间点 {current_time}: 窗口范围 [{start_time}, {end_time}]")
            
            # 提取窗口时间范围内的数据
            window_mask = (patient_data[time_col] >= start_time) & (patient_data[time_col] <= end_time)
            window_data = patient_data[window_mask].copy()

            existing_times = set(window_data[time_col].tolist())

            window_mp = mp_period[window_mask.to_numpy()]
            last_valid_mp = window_mp[~window_mp.isna()].max() if (~window_mp.isna()).any() else pd.NaT
            
            # print(f"    找到数据时间点: {window_data[time_col].tolist()}")
            if len(window_data) < min_win_valid:
                # print(f"    有效数据点不足: {window_data[time_col].tolist()} < 有效数据点数量下限({min_win_valid})")
                continue
            
            # 如果窗口时间范围内的数据不足window_size，需要填充
            if len(window_data) < window_size:
                missing_rows = window_size - len(window_data)
                
                # 找出缺失的时间点
                
                missing_times = [t for t in range(start_time, end_time + 1) if t not in existing_times]
                missing_times = missing_times[:missing_rows]  # 只取需要的数量
                
                # print(f"    需要填充 {missing_rows} 个时间点: {missing_times}")
                
                # 创建缺失时间点的空行
                empty_rows = []
                for missing_time in missing_times:
                    empty_row = {col: np.nan for col in patient_data.columns}
                    empty_row[id_col] = patient_id
                    empty_row[time_col] = missing_time

                    is_padding = (missing_time < patient_time_min) or (missing_time > patient_time_max)
                    if is_padding:
                        for c in feature_cols:
                            empty_row[c] = -np.inf
                    empty_rows.append(empty_row)
                
                if empty_rows:
                    empty_df = pd.DataFrame(empty_rows)
                    # print(f'window_data dtypes:\n{window_data.dtypes}, \n empty_df dtypes:\n{empty_df.dtypes}')
                    empty_df = empty_df.astype(window_data.dtypes)
                    window_data = pd.concat([empty_df, window_data], ignore_index=True)
                    window_data = window_data.sort_values(time_col).reset_index(drop=True)
            
            # 应用填充策略
            window_data[feature_cols] = window_data[feature_cols].astype(float)
            window_data = apply_fill_method(window_data, fill_method, feature_cols)
            
            # 添加窗口信息
            window_data['id_window'] = window_id
            # window_data['window_position'] = range(len(window_data))
            # window_data['target_timeunit'] = current_time
            # window_data['window_start_time'] = start_time
            # window_data['window_end_time'] = end_time
            
            windowed_data.append(window_data)

            if (df_demo is not None) or (df_labels is not None):
                win_meta.append({
                    'id_window': window_id,
                    id_col: patient_id,
                    'window_end_time': end_time,
                    'last_valid_mp': last_valid_mp,  # Period[M] 或 NaT
                    'first_mp': first_mp             # Period[M] 或 NaT
                })

            window_id += 1

            if end_time >= patient_data['TimeUnit'].max():
                break

    
    # 合并所有窗口数据
    result_df = pd.concat(windowed_data, ignore_index=True)
    
    print(f"\n生成了 {window_id} 个窗口")
    print(f"结果数据形状: {result_df.shape}")

    ## 基于 win_meta 生成窗口对齐的 demo / label
    win_demo = None
    win_labels = None
    if len(win_meta) > 0:
        meta_df = pd.DataFrame(win_meta)

        ## 1) win_demo：合并 birthyear/age_entry/dmale，并计算 age_at_window
        if df_demo is not None:
            demo_cols = [id_col] + [c for c in patient_info_cols if c in df_demo.columns]
            tmp = meta_df.merge(df_demo[demo_cols], on=id_col, how='left')

            last_ts = tmp['last_valid_mp'].apply(
                lambda p: (p.to_timestamp(how='start') if not pd.isna(p) else pd.NaT)
            )

            # birthyear → 数值
            if 'birth_year' in tmp.columns:
                by = pd.to_numeric(tmp['birth_year'], errors='coerce')
            else:
                by = pd.Series(np.nan, index=tmp.index)

            # 优先用 birthyear 计算；若 birthyear 缺失且你想保留旧逻辑，可回退到 age_entry+月差/12
            age_from_birthyear = np.where(
                by.notna() & last_ts.notna(),
                last_ts.dt.year - by,
                np.nan
            )

            tmp['age_at_window'] = np.where(
                by.notna() & last_ts.notna(),
                age_from_birthyear,
                np.nan 
            )

            cols_out = ['id_window', id_col]
            if 'gender' in tmp.columns: cols_out.append('gender')
            if 'age_at_window' in tmp.columns: cols_out.append('age_at_window')
            win_demo = tmp[cols_out].copy()

        ## 2) win_labels：用 earliest_date（datetime）与 last_valid_mp + time_horizon_months 比较
        if df_labels is not None and event_date_cols is not None:
            # 只保留需要的列
            avaliable_event_cols = [c for c in event_date_cols if c in df_labels.columns]
            lab_cols = [id_col] + avaliable_event_cols
            lab = meta_df.merge(df_labels[lab_cols], on=id_col, how='left')

            # 把 Period[M] 转成该月的起始日期（或你喜欢的月末：how='end'）
            last_ts = lab['last_valid_mp'].apply(
                lambda p: (p.to_timestamp(how='start') if not pd.isna(p) else pd.NaT)
            )

            # 加上预测时长（单位：月）
            horizon_dt = last_ts + pd.offsets.DateOffset(months=int(time_horizon_months))

            # 规则：如果 earliest_date <= (last_valid_month + horizon) 则 y=1，否则 y=0；
            #       无 earliest_date 记为 0；无 last_valid_month 也记为 0。
            # ed = pd.to_datetime(lab[earliest_date_col], errors='coerce')
            # y = np.where(ed.notna() & horizon_dt.notna(), (ed <= horizon_dt).astype(int), 0)
            y_cols = {}
            for event_col in avaliable_event_cols:
                ed = pd.to_datetime(lab[event_col], errors='coerce')
                # 规则：如果 event_date <= (last_valid_month + horizon) 则 y=1，否则 y=0
                y = np.where(ed.notna() & horizon_dt.notna(), (ed <= horizon_dt).astype(int), 0)
                y_cols[f'y_{event_col}'] = y


            # win_labels = pd.DataFrame({
            #     'id_window': lab['id_window'].values,
            #     id_col: lab[id_col].values,
            #     'y': y
            # })
            win_labels = pd.DataFrame({
                    'id_window': lab['id_window'].values,
                    id_col: lab[id_col].values,
                    **y_cols  # 展开所有 y_* 列
                })

        ## 3) 窗口级近似均衡抽样（只在 win_labels 可用时启用） ===
        if balance_windows:
            # if (win_labels is None) or ('y' not in win_labels.columns):
            #     print("[balance] 未找到窗口级标签（win_labels['y']），跳过均衡。")
            y_columns = [c for c in win_labels.columns if c.startswith('y_')]
            if (win_labels is None) or len(y_columns) == 0:
                print("[balance] 未找到窗口级标签（win_labels['y_*']），跳过均衡。")
            else:
                rng = np.random.default_rng(balance_random_state)
                balanced_col = y_columns[-1]  # 以最后一个 y_* 列为准做均衡
                y_series = win_labels.set_index('id_window')[balanced_col].astype(int)
                
                # 统计每类窗口数
                counts = y_series.value_counts().sort_index()
                # print(f"[balance] before:")
                for y_col in y_columns:
                    counts = win_labels[y_col].value_counts().sort_index()
                    print(f"  {y_col}: {counts.to_dict()}")

                if len(counts) >= 1:
                    min_count = counts.min()
                    lo, hi = balance_range

                    chosen_win_ids = []
                    for cls_current, cls_count in counts.items():
                        wid_cls = y_series.index[y_series.values == cls_current].to_numpy()
                        base_count = int(np.floor(min_count * rng.uniform(lo, hi)))
                        base_count = max(min_per_class, min(base_count, cls_count))
                        if base_count > 0:
                            chosen = rng.choice(wid_cls, size=base_count, replace=False)
                            chosen_win_ids.append(chosen)

                    if len(chosen_win_ids) > 0:
                        keep_ids = np.concatenate(chosen_win_ids)
                    else:
                        keep_ids = np.array([], dtype=result_df['id_window'].dtype)

                    # 依据 keep_ids 过滤三张表
                    result_df = result_df[result_df['id_window'].isin(keep_ids)].copy()
                    if win_demo is not None:
                        win_demo = win_demo[win_demo['id_window'].isin(keep_ids)].copy()
                    if win_labels is not None:
                        win_labels = win_labels[win_labels['id_window'].isin(keep_ids)].copy()

                    
                    new_counts = win_labels[balanced_col].value_counts().sort_index()
                    # print(f"[balance] after (windows={win_labels.shape[0] if win_labels is not None else 'NA'}):")
                    for y_col in y_columns:
                        new_counts = win_labels[y_col].value_counts().sort_index()
                        # print(f"  {y_col}: {new_counts.to_dict()}")

    if return_extra:
        return result_df, win_demo, win_labels
    
    return result_df

import numpy as np
from sklearn.preprocessing import StandardScaler

def fit_static_stats(X_train_static, eps=1e-5):
    X = X_train_static.astype(float).copy()
    X[np.isinf(X)] = np.nan
    mean = np.nanmean(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.nanstd(X, axis=0) + eps
    std = np.where(np.isfinite(std), std, 1.0)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)

def process_static_by_train_stats(X, mean, std):
    if X is None:
        return None
    X = X.astype(float).copy()
    X[np.isinf(X)] = np.nan
    inds = np.where(np.isnan(X))
    if len(inds[0]) > 0:
        X[inds] = np.take(mean, inds[1])
    return ((X - mean) / std).astype(np.float32)

def _safe_fit_ts_scaler(df_train_ts, feature_cols, eps=1e-6):
    X = df_train_ts[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    scaler = StandardScaler()
    scaler.fit(X)
    bad = ~np.isfinite(scaler.mean_) | ~np.isfinite(scaler.scale_) | (scaler.scale_ < eps)
    if np.any(bad):
        scaler.mean_[bad] = 0.0
        scaler.scale_[bad] = 1.0
        if hasattr(scaler, "var_"):
            scaler.var_[bad] = 1.0
    return scaler

def _apply_ts_scaler(df_ts, feature_cols, scaler):
    df_ts = df_ts.copy()
    X = df_ts[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    Xn = scaler.transform(X)
    df_ts.loc[:, feature_cols] = Xn
    return df_ts

def _windows_to_arrays(df_win, demo_win, labels_win,
                      feature_cols, window_size,
                      target="any", id_col="ID", time_col="TimeUnit"):
    df_win = df_win.sort_values(["id_window", time_col]).reset_index(drop=True)
    labels_win = labels_win.sort_values(["id_window"]).reset_index(drop=True)
    if demo_win is not None:
        demo_win = demo_win.sort_values(["id_window"]).reset_index(drop=True)

    counts = df_win.groupby("id_window").size()
    assert counts.nunique() == 1 and counts.iloc[0] == window_size, \
        f"Windows not all size={window_size}: {counts.value_counts().head()}"

    N = counts.shape[0]
    Fts = len(feature_cols)
    X_ts = df_win[feature_cols].to_numpy(dtype=float).reshape(N, window_size, Fts)
    X_ts[np.isinf(X_ts)] = np.nan
    X_ts = X_ts.astype(np.float32)

    # y
    y_cols = [c for c in labels_win.columns if c.startswith("y_")]
    if target == "any":
        # y = labels_win[y_cols].max(axis=1).to_numpy(dtype=int)
        print(f"Using target columns: {y_cols[-1]}")
        y = labels_win[y_cols[-1]].to_numpy(dtype=int)
    else:
        print(f"Using target column: {target}")
        y = labels_win[target].to_numpy(dtype=int)

    # static raw (N,D)
    X_static = None
    static_cols = None
    if demo_win is not None:
        static_cols = [c for c in demo_win.columns if c not in ["id_window", id_col]]
        X_static = demo_win[static_cols].to_numpy(dtype=float)
        X_static[np.isinf(X_static)] = np.nan

    return X_ts, X_static, y.astype(int), static_cols

def make_pypots_sets(
    seed,
    ts, df_demo, df_labels,
    feature_cols,
    window_size,
    min_win_valid,
    fill_method,
    agg_select,
    time_horizon_months,
    test_size=0.2,
    val_size=0.2,
    val_seed=42,
    target="any",
    id_col="ID",
    time_col="TimeUnit",
    balance_windows_train=False,
    patient_info_cols=None,
    event_date_cols=None,
    expand=False,
    external=False,
    static_stats=None,
    ts_scaler=None,
):
    """
    产出：
      X_*_ts_norm (NaN保留), X_*_static_norm (mean-impute+zscore,无NaN), X_*_combined (NaN保留)
      y_train/y_val/y_test
      以及 train/val/test_set (含X_static) 直接给你的循环用
    """

    # A) raw train/test split
    if not external:
        raw_train_ids, test_ids = split_patient_ids(
            ts, df_labels, id_col=id_col,
            test_size=test_size, random_state=seed,
            use_stratify=True, stratify_col="has_event"
        )
        df_train_raw_ts, df_test_ts = slice_by_ids(ts, raw_train_ids, test_ids, id_col=id_col)
        df_train_raw_lab = df_labels[df_labels[id_col].isin(raw_train_ids)].copy()
        df_test_lab      = df_labels[df_labels[id_col].isin(test_ids)].copy()
        df_train_raw_demo = df_demo[df_demo[id_col].isin(raw_train_ids)].copy() if df_demo is not None else None
        df_test_demo      = df_demo[df_demo[id_col].isin(test_ids)].copy()      if df_demo is not None else None
    else:
        ## train/test are the same set, only use test, keep train set only for compatible with the pipeline
        df_train_raw_ts = ts.copy()
        df_train_raw_lab = df_labels.copy()
        df_train_raw_demo = df_demo.copy() if df_demo is not None else None
        df_test_ts = ts.copy()
        df_test_lab = df_labels.copy()
        df_test_demo = df_demo.copy() if df_demo is not None else None

    if expand:
        # 训练集：解包
        df_train_raw_ts, feature_cols = expand_zipped_columns(df_train_raw_ts, feature_cols,
                                                        selection=agg_select, drop_original=True)
        # 测试集：按同样的 selection 解包（得到的列名与训练集一致）
        df_test_ts, _ = expand_zipped_columns(df_test_ts, feature_cols,
                                            selection=agg_select, drop_original=True)

    # B) raw_train -> train/val (fixed seed)
    val_ratio_in_train = val_size / (1.0 - test_size)
    train_ids, val_ids = split_patient_ids(
        df_train_raw_ts, df_train_raw_lab, id_col=id_col,
        test_size=val_ratio_in_train, random_state=val_seed,
        use_stratify=True, stratify_col="has_event"
    )
    df_train_ts, df_val_ts = slice_by_ids(df_train_raw_ts, train_ids, val_ids, id_col=id_col)
    df_train_lab = df_labels[df_labels[id_col].isin(train_ids)].copy()
    df_val_lab   = df_labels[df_labels[id_col].isin(val_ids)].copy()
    df_train_demo = df_demo[df_demo[id_col].isin(train_ids)].copy() if df_demo is not None else None
    df_val_demo   = df_demo[df_demo[id_col].isin(val_ids)].copy()   if df_demo is not None else None

    # C) TS normalization (train fit only)
    if feature_cols is None:
        feature_cols = [c for c in ts.columns if c not in [id_col, time_col, "MonthPeriod"]]
    if ts_scaler is None:
        ts_scaler = _safe_fit_ts_scaler(df_train_ts, feature_cols)
    df_train_ts = _apply_ts_scaler(df_train_ts, feature_cols, ts_scaler)
    df_val_ts   = _apply_ts_scaler(df_val_ts,   feature_cols, ts_scaler)
    df_test_ts  = _apply_ts_scaler(df_test_ts,  feature_cols, ts_scaler)

    # D) windowing
    def _make_windows(df_ts_part, df_demo_part, df_labels_part, balance_windows=False):
        df_win, demo_win, labels_win = create_sliding_windows(
            df_ts_part,
            window_size=window_size,
            min_win_valid=min_win_valid,
            fill_method=fill_method,
            id_col=id_col,
            time_col=time_col,
            patient_info_cols=patient_info_cols,
            event_date_cols=event_date_cols,
            preexpanded=True,
            features_override=feature_cols,
            agg_select=agg_select,
            df_demo=df_demo_part,
            df_labels=df_labels_part,
            time_horizon_months=time_horizon_months,
            balance_windows=balance_windows,
        )
        return df_win, demo_win, labels_win

    df_tr_win, demo_tr_win, lab_tr_win = _make_windows(df_train_ts, df_train_demo, df_train_lab, balance_windows=balance_windows_train)
    df_va_win, demo_va_win, lab_va_win = _make_windows(df_val_ts,   df_val_demo,   df_val_lab, balance_windows=False)
    df_te_win, demo_te_win, lab_te_win = _make_windows(df_test_ts,  df_test_demo,  df_test_lab, balance_windows=False)

    # E) windows -> arrays
    X_train_ts_norm, X_train_static_raw, y_train, static_cols = _windows_to_arrays(
        df_tr_win, demo_tr_win, lab_tr_win, feature_cols, window_size,
        target=target, id_col=id_col, time_col=time_col
    )
    X_val_ts_norm, X_val_static_raw, y_val, _ = _windows_to_arrays(
        df_va_win, demo_va_win, lab_va_win, feature_cols, window_size,
        target=target, id_col=id_col, time_col=time_col
    )
    X_test_ts_norm, X_test_static_raw, y_test, _ = _windows_to_arrays(
        df_te_win, demo_te_win, lab_te_win, feature_cols, window_size,
        target=target, id_col=id_col, time_col=time_col
    )

    # F) static norm by train stats
    if X_train_static_raw is not None and X_train_static_raw.shape[1] > 0:
        if static_stats is not None:
            st_mean = static_stats["mean"]
            st_std  = static_stats["std"]
            static_cols = static_stats['static_cols']
        else:
            st_mean, st_std = fit_static_stats(X_train_static_raw, eps=1e-5)
        X_train_static_norm = process_static_by_train_stats(X_train_static_raw, st_mean, st_std)
        X_val_static_norm   = process_static_by_train_stats(X_val_static_raw,   st_mean, st_std)
        X_test_static_norm  = process_static_by_train_stats(X_test_static_raw,  st_mean, st_std)
        static_stats = {"mean": st_mean, "std": st_std, "static_cols": static_cols}
    else:
        X_train_static_norm = np.zeros((X_train_ts_norm.shape[0], 0), dtype=np.float32)
        X_val_static_norm   = np.zeros((X_val_ts_norm.shape[0],   0), dtype=np.float32)
        X_test_static_norm  = np.zeros((X_test_ts_norm.shape[0],  0), dtype=np.float32)
        static_stats = None

    # G) combined = ts_norm + repeat(static_norm)
    if X_train_static_norm.shape[1] > 0:
        def _concat(X_ts, Xs):
            Xs_rep = np.repeat(Xs[:, None, :], X_ts.shape[1], axis=1)  # (N,T,D)
            return np.concatenate([X_ts, Xs_rep.astype(np.float32)], axis=-1)
        X_train_combined = _concat(X_train_ts_norm, X_train_static_norm)
        X_val_combined   = _concat(X_val_ts_norm,   X_val_static_norm)
        X_test_combined  = _concat(X_test_ts_norm,  X_test_static_norm)
    else:
        X_train_combined, X_val_combined, X_test_combined = X_train_ts_norm, X_val_ts_norm, X_test_ts_norm

    # H) package: you can directly use these in your training loop
    train_set_ts   = {"X": X_train_ts_norm, "y": y_train, "X_static": X_train_static_norm}
    val_set_ts     = {"X": X_val_ts_norm,   "y": y_val,   "X_static": X_val_static_norm}
    test_set_ts    = {"X": X_test_ts_norm,              "X_static": X_test_static_norm}

    train_set_comb = {"X": X_train_combined, "y": y_train, "X_static": X_train_static_norm}
    val_set_comb   = {"X": X_val_combined,   "y": y_val,   "X_static": X_val_static_norm}
    test_set_comb  = {"X": X_test_combined,              "X_static": X_test_static_norm}

    return {
        "X_train_ts_norm": X_train_ts_norm,
        "X_val_ts_norm": X_val_ts_norm,
        "X_test_ts_norm": X_test_ts_norm,
        "X_train_static_norm": X_train_static_norm,
        "X_val_static_norm": X_val_static_norm,
        "X_test_static_norm": X_test_static_norm,
        "X_train_combined": X_train_combined,
        "X_val_combined": X_val_combined,
        "X_test_combined": X_test_combined,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "n_steps": X_train_ts_norm.shape[1],
        "n_features_ts": X_train_ts_norm.shape[2],
        "n_features_total": X_train_combined.shape[2],
        "train_set_ts": train_set_ts,
        "val_set_ts": val_set_ts,
        "test_set_ts": test_set_ts,
        "train_set_combined": train_set_comb,
        "val_set_combined": val_set_comb,
        "test_set_combined": test_set_comb,
        "static_stats": static_stats,
        "ts_scaler": ts_scaler,
    }




def convert_mimic_liver_units(df):
    df = df.copy()
    mask = (df["feature_name"].eq("Albumin")) & (df["valueuom"].astype(str).str.lower().eq("mg/dl"))
    df.loc[mask, "valuenum"] = df.loc[mask, "valuenum"] / 1000.0
    df.loc[mask, "valueuom"] = "g/dL"
    mask = (df["feature_name"].eq("Albumin")) & (df["valueuom"].astype(str).str.lower().eq("g/dl"))
    df.loc[mask, "valuenum"] = df.loc[mask, "valuenum"] * 10.0
    df.loc[mask, "valueuom"] = "g/L"
    mask = (df["feature_name"].str.lower().isin(["protein, total", "total protein"])) & (
        df["valueuom"].astype(str).str.lower().eq("mg/dl")
    )
    df.loc[mask, "valuenum"] = df.loc[mask, "valuenum"] / 1000.0
    df.loc[mask, "valueuom"] = "g/dL"
    mask = (df["feature_name"].str.lower().isin(["protein, total", "total protein"])) & (
        df["valueuom"].astype(str).str.lower().eq("g/dl")
    )
    df.loc[mask, "valuenum"] = df.loc[mask, "valuenum"] * 10.0
    df.loc[mask, "valueuom"] = "g/L"
    mask = df["valueuom"].astype(str).str.contains(r"#/uL|#/ul", case=False, na=False)
    df.loc[mask, "valuenum"] = df.loc[mask, "valuenum"] / 1000.0
    df.loc[mask, "valueuom"] = "K/uL"
    return df


def standardize_mimic_liver_features(labs):
    labs = labs.copy()
    labs["feature_std"] = labs["feature_name"].copy()

    is_eos = labs["feature_name"].str.lower().str.contains("eosin", na=False)
    mask_abs = is_eos & labs["valueuom"].astype(str).str.contains(r"#|/uL|/ul", case=False, na=False)
    mask_pct = is_eos & labs["valueuom"].astype(str).str.contains(r"%", case=False, na=False)
    labs.loc[mask_abs, "feature_std"] = "Eosinophils_abs"
    labs.loc[mask_pct, "feature_std"] = "Eosinophils_pct"
    labs.loc[mask_pct, "feature_std"] = "Eosinophils"
    labs = labs[labs["feature_std"] != "Eosinophils_abs"].copy()

    is_lym = labs["feature_name"].str.lower().str.contains("lymphocyte", na=False)
    mask_abs = is_lym & labs["valueuom"].astype(str).str.contains(r"#|/uL|/ul", case=False, na=False)
    mask_pct = is_lym & labs["valueuom"].astype(str).str.contains(r"%", case=False, na=False)
    labs.loc[mask_abs, "feature_std"] = "Lymphocytes_abs"
    labs.loc[mask_pct, "feature_std"] = "Lymphocytes_pct"
    labs.loc[mask_abs, "feature_std"] = "Lymphocytes"
    labs = labs[labs["feature_std"] != "Lymphocytes_pct"].copy()

    is_neutro = labs["feature_name"].str.lower().str.contains("neutrophil", na=False)
    mask_abs = is_neutro & labs["valueuom"].astype(str).str.contains(r"#|/uL|/ul", case=False, na=False)
    mask_pct = is_neutro & labs["valueuom"].astype(str).str.contains(r"%", case=False, na=False)
    labs.loc[mask_abs, "feature_std"] = "Neutrophils_abs"
    labs.loc[mask_pct, "feature_std"] = "Neutrophils_pct"
    labs.loc[mask_abs, "feature_std"] = "Neutrophils"
    labs = labs[labs["feature_std"] != "Neutrophils_pct"].copy()

    is_baso = labs["feature_name"].str.lower().str.contains("basophil", na=False)
    mask_abs = is_baso & labs["valueuom"].astype(str).str.contains(r"#|/uL|/ul", case=False, na=False)
    mask_pct = is_baso & labs["valueuom"].astype(str).str.contains(r"%", case=False, na=False)
    labs.loc[mask_abs, "feature_std"] = "Basophils_abs"
    labs.loc[mask_pct, "feature_std"] = "Basophils_pct"
    labs.loc[mask_abs, "feature_std"] = "Basophils"
    labs = labs[labs["feature_std"] != "Basophils_pct"].copy()
    return labs


def _read_mimic_raw(csv_path):
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
    date_cols = [
        "index_time", "t_hcc_first", "t_cirr_first", "t_fib_first", "t_lf_first",
        "t_event_min", "t_event_max", "last_followup_time", "charttime",
    ]
    return pd.read_csv(
        csv_path,
        parse_dates=[c for c in date_cols if c in header],
        low_memory=False,
    )


def load_mimic_liver_source(csv_path=None, data_root=None):
    if csv_path is None:
        liver_data_dir = _resolve_liver_data_dir(data_root)
        candidate_paths = [
            liver_data_dir / "mimiciv_liver_raw1.csv",
            liver_data_dir / "mimiciv_liver_raw.csv",
            liver_data_dir / "mimic" / "mimiciv_liver_raw1.csv",
            liver_data_dir / "mimic" / "mimiciv_liver_raw.csv",
        ]
        csv_path = next((p for p in candidate_paths if p.exists()), candidate_paths[0])
    csv_path = Path(csv_path)
    raw = _read_mimic_raw(csv_path)

    cohort_cols = [
        "subject_id", "index_hadm_id", "index_time", "gender", "age_at_index",
        "t_hcc_first", "t_cirr_first", "t_fib_first", "t_lf_first",
        "t_event_min", "t_event_max", "last_followup_time",
    ]
    cohort_cols = [c for c in cohort_cols if c in raw.columns]
    cohort = raw[cohort_cols].drop_duplicates("subject_id").reset_index(drop=True)

    if ("index_time" in cohort.columns) and ("age_at_index" in cohort.columns):
        cohort["birth_year"] = (cohort["index_time"].dt.year - cohort["age_at_index"]).astype("float")
    if "gender" in cohort.columns:
        cohort["gender_num"] = cohort["gender"].map({"F": 0, "M": 1}).astype("float")

    labs_cols = ["subject_id", "hadm_id", "charttime", "itemid", "feature_name", "valuenum", "valueuom"]
    labs_cols = [c for c in labs_cols if c in raw.columns]
    labs = raw[labs_cols].copy()
    labs = labs.dropna(subset=["subject_id", "charttime", "itemid", "valuenum"])
    labs["itemid"] = labs["itemid"].astype(int)
    labs["valuenum"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    labs = labs.dropna(subset=["valuenum"])
    labs = convert_mimic_liver_units(labs)
    labs = standardize_mimic_liver_features(labs)
    labs["MonthPeriod"] = labs["charttime"].dt.to_period("M")

    labs = labs.sort_values(by=["subject_id", "charttime"])
    agg = labs.groupby(["subject_id", "MonthPeriod", "feature_std"], as_index=False)["valuenum"].first()
    wide = agg.pivot_table(
        index=["subject_id", "MonthPeriod"],
        columns="feature_std",
        values="valuenum",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    def pick_t1(row):
        t1 = row.get("t_event_max", pd.NaT)
        if pd.isna(t1):
            t1 = row.get("last_followup_time", pd.NaT)
        return t1

    cohort["t1_extract"] = cohort.apply(pick_t1, axis=1)
    cohort = cohort[cohort["t1_extract"].notna() & cohort["index_time"].notna()].copy()
    cohort = cohort[cohort["t1_extract"] > cohort["index_time"]].copy()
    cohort["index_mp"] = cohort["index_time"].dt.to_period("M")
    cohort["t1_mp"] = cohort["t1_extract"].dt.to_period("M")
    cohort["T_max"] = (cohort["t1_mp"].astype("int64") - cohort["index_mp"].astype("int64") + 1).astype(int)
    cohort = cohort[cohort["T_max"] >= 2].copy()

    wide_scoped = wide.merge(cohort[["subject_id", "index_mp", "t1_mp"]], on="subject_id", how="inner")
    wide_scoped = wide_scoped[
        (wide_scoped["MonthPeriod"] >= wide_scoped["index_mp"]) &
        (wide_scoped["MonthPeriod"] <= wide_scoped["t1_mp"])
    ].copy()
    wide_scoped["TimeUnit"] = (
        wide_scoped["MonthPeriod"].astype("int64") - wide_scoped["index_mp"].astype("int64") + 1
    ).astype(int)
    ts = wide_scoped.drop(columns=["index_mp", "t1_mp"]).sort_values(
        ["subject_id", "TimeUnit", "MonthPeriod"]
    ).reset_index(drop=True)
    ts = ts[["subject_id", "MonthPeriod", "TimeUnit"] + [
        c for c in ts.columns if c not in ["subject_id", "MonthPeriod", "TimeUnit"]
    ]]

    event_cols = ["t_hcc_first", "t_cirr_first", "t_fib_first", "t_lf_first", "t_event_min"]
    event_cols = [c for c in event_cols if c in cohort.columns]
    feature_cols = [c for c in ts.columns if c not in ["subject_id", "MonthPeriod", "TimeUnit"]]
    df_labels = cohort[["subject_id"] + event_cols].copy()
    df_labels["has_event"] = df_labels[event_cols].notna().any(axis=1).astype(int)
    df_demo = cohort[["subject_id", "birth_year", "age_at_index", "gender_num"]].copy()
    df_demo = df_demo.rename(columns={"gender_num": "gender"})
    patient_info_cols = [c for c in df_demo.columns if c != "subject_id"]

    return {
        "ts": ts,
        "demo": df_demo,
        "labels": df_labels,
        "feature_cols": feature_cols,
        "event_date_cols": event_cols,
        "patient_info_cols": patient_info_cols,
        "id_col": "subject_id",
        "time_col": "TimeUnit",
    }


def load_ttsh_liver_source(data_root=None):
    data_root = _resolve_liver_data_dir(data_root)
    fp_ts_candidates = [
        data_root / "data_compact_2.csv",
        data_root / "Preprocess" / "data_compact_2.csv",
    ]
    fp_demo_candidates = [
        data_root / "Baseline.HCC.livercomp.FIB4.PLT.csv",
    ]
    fp_ts = next((p for p in fp_ts_candidates if p.exists()), fp_ts_candidates[0])
    fp_demo = next((p for p in fp_demo_candidates if p.exists()), fp_demo_candidates[0])

    feature_cols = [
        "Albumin", "Platelets", "Lymphocytes", "White cells count", "Neutrophils",
        "Basophils", "Eosinophils", "Total protein",
    ]
    df_raw_ts = pd.read_csv(fp_ts)
    df_raw_ts = df_raw_ts[["ID", "TimeUnit", "MonthPeriod"] + feature_cols]
    df_raw_ts["MonthPeriod"] = pd.to_datetime(df_raw_ts["MonthPeriod"]).dt.to_period("M")
    df_raw_ts = df_raw_ts.dropna(subset=feature_cols, how="all")

    df_raw_demo = pd.read_csv(fp_demo)
    df_raw_demo = df_raw_demo.iloc[:, 1:]
    df_raw_demo = df_raw_demo.rename(columns={
        "Study.ID": "ID",
        "birth.year": "birth_year",
        "age.entry": "age_entry",
        "Liver.failure": "Liver_failure",
        "Date_Liver.failure": "Date_Liver_failure",
    })
    demo_cols = [
        "birth_year", "age_entry", "gender", "earlieststeatosisentrydate",
        "HCC", "HCC_date", "Cirrhosis", "Date_Cirrhosis", "Fibrosis", "Date_Fibrosis",
        "Liver_failure", "Date_Liver_failure",
    ]
    df_raw_demo = df_raw_demo[["ID"] + demo_cols]
    for col in ["earlieststeatosisentrydate", "HCC_date", "Date_Cirrhosis", "Date_Fibrosis", "Date_Liver_failure"]:
        df_raw_demo[col] = pd.to_datetime(df_raw_demo[col], dayfirst=True, errors="coerce")
    df_raw_demo = df_raw_demo.sort_values(by=["ID"]).reset_index(drop=True)

    df_filtered_ts = pd.merge(
        df_raw_ts,
        df_raw_demo[["ID", "earlieststeatosisentrydate"]],
        on="ID",
        how="left",
    )
    df_filtered_ts = df_filtered_ts[
        df_filtered_ts["MonthPeriod"] >= df_filtered_ts["earlieststeatosisentrydate"].dt.to_period("M")
    ]
    df_filtered_ts = df_filtered_ts.drop(columns=["earlieststeatosisentrydate"])
    df_filtered_ts = df_filtered_ts[df_filtered_ts["ID"].isin(df_raw_demo["ID"])]

    event_date_cols = ["HCC_date", "Date_Cirrhosis", "Date_Fibrosis", "Date_Liver_failure"]
    patient_info_cols = ["birth_year", "age_entry", "gender", "earlieststeatosisentrydate"]
    df_demo = df_raw_demo[["ID"] + patient_info_cols].drop_duplicates(subset=["ID"]).reset_index(drop=True)
    if "gender" in df_demo.columns and not np.issubdtype(df_demo["gender"].dtype, np.number):
        df_demo["gender"] = df_demo["gender"].map({"F": 0, "M": 1, "Female": 0, "Male": 1}).astype("float")
    df_labels = df_raw_demo[["ID"] + event_date_cols].drop_duplicates(subset=["ID"]).reset_index(drop=True)
    df_labels["earliest_event_date"] = df_labels[event_date_cols].min(axis=1, skipna=True)
    df_labels["has_event"] = df_labels["earliest_event_date"].notna().astype(int)

    return {
        "ts": df_filtered_ts,
        "demo": df_demo,
        "labels": df_labels,
        "feature_cols": feature_cols,
        "event_date_cols": event_date_cols + ["earliest_event_date"],
        "patient_info_cols": patient_info_cols,
        "id_col": "ID",
        "time_col": "TimeUnit",
    }


def prepare_liver_split(
    source,
    seed,
    window_size=6,
    min_win_valid=None,
    fill_method="inf",
    agg_select=("first",),
    time_horizon_months=6,
    balance_windows_train=True,
    external=False,
    static_stats=None,
    ts_scaler=None,
):
    if min_win_valid is None:
        min_win_valid = int(window_size * 0.5)
    return make_pypots_sets(
        seed=seed,
        ts=source["ts"],
        df_demo=source["demo"],
        df_labels=source["labels"],
        feature_cols=source["feature_cols"],
        window_size=window_size,
        min_win_valid=min_win_valid,
        fill_method=fill_method,
        agg_select=agg_select,
        time_horizon_months=time_horizon_months,
        test_size=0.2,
        val_size=0.2,
        val_seed=42,
        target="any",
        id_col=source.get("id_col", "ID"),
        time_col=source.get("time_col", "TimeUnit"),
        balance_windows_train=balance_windows_train,
        patient_info_cols=source.get("patient_info_cols"),
        event_date_cols=source.get("event_date_cols"),
        expand=True,
        external=external,
        static_stats=static_stats,
        ts_scaler=ts_scaler,
    )


def save_liver_npz(prepared, path, dataset_name, feature_names=None, static_feature_names=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = feature_names or prepared.get("feature_cols") or []
    static_feature_names = static_feature_names or (
        prepared.get("static_stats", {}) or {}
    ).get("static_cols", []) or []
    np.savez(
        path,
        X_train_ts=prepared["X_train_ts_norm"],
        X_val_ts=prepared["X_val_ts_norm"],
        X_test_ts=prepared["X_test_ts_norm"],
        X_train_static=prepared["X_train_static_norm"],
        X_val_static=prepared["X_val_static_norm"],
        X_test_static=prepared["X_test_static_norm"],
        y_train=prepared["y_train"],
        y_val=prepared["y_val"],
        y_test=prepared["y_test"],
        feature_names=np.array(feature_names, dtype=object),
        static_feature_names=np.array(static_feature_names, dtype=object),
        dataset_name=np.array(dataset_name),
    )
    return path


def load_prepared_liver_npz(path):
    data = np.load(path, allow_pickle=True)
    return {
        "X_train_ts_norm": data["X_train_ts"],
        "X_val_ts_norm": data["X_val_ts"],
        "X_test_ts_norm": data["X_test_ts"],
        "X_train_static_norm": data["X_train_static"],
        "X_val_static_norm": data["X_val_static"],
        "X_test_static_norm": data["X_test_static"],
        "y_train": data["y_train"],
        "y_val": data["y_val"],
        "y_test": data["y_test"],
        "feature_names": data.get("feature_names", np.array([], dtype=object)).tolist(),
        "static_feature_names": data.get("static_feature_names", np.array([], dtype=object)).tolist(),
    }
