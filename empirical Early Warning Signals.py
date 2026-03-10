import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import welch
import statsmodels.formula.api as smf
import scipy.stats as stats
from tqdm import tqdm
import warnings
import gc
import csv
import traceback
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BIDS_ROOT_DIR = r"C:\Users\chiro\ds003838_pupil_data"
FS = 250  
DOWNSAMPLE_FACTOR = 5 # 250Hz -> 50Hz (AC1アーティファクト回避用)

# sub-013はアーティファクト排除の基準に基づき除外リストに追加
EXCLUDE_SUBJECTS = ['sub-017', 'sub-094']

sns.set_theme(style="whitegrid")

# ==============================================================================
# BULLETPROOF DATA LOADER (Bypassing Pandas C-Engine Bug)
# ==============================================================================
def load_pupil_safe(filepath):
    valid_cols = ['pupil_timestamp', 'diameter_3d', 'diameter', 'confidence', 'method']
    data = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            try:
                header = next(reader)
            except StopIteration:
                return pd.DataFrame()
                
            col_indices = {}
            for idx, col in enumerate(header):
                col_clean = col.strip()
                if col_clean in valid_cols:
                    col_indices[col_clean] = idx
                    data[col_clean] = []
                    
            if not col_indices:
                return pd.DataFrame()
                
            for row in reader:
                for col_name, idx in col_indices.items():
                    if idx < len(row):
                        data[col_name].append(row[idx])
                    else:
                        data[col_name].append(np.nan)
    except Exception as e:
        print(f"\n[Custom Parser Error] {os.path.basename(filepath)}: {e}")
        return pd.DataFrame()
        
    return pd.DataFrame(data)

# ==============================================================================
# PREPROCESSING
# ==============================================================================
def preprocess_pupil(filepath, fs=250):
    df = load_pupil_safe(filepath)
    if df.empty: return None, None

    if 'diameter_3d' in df.columns and df['diameter_3d'].notnull().any():
        target_col = 'diameter_3d'
        if 'method' in df.columns:
            mask = df['method'].astype(str).str.contains('3d', case=False, na=False)
            if mask.sum() > 0:
                df = df[mask].copy()
    elif 'diameter' in df.columns:
        target_col = 'diameter'
    else:
        return None, None

    df['pupil_timestamp'] = pd.to_numeric(df['pupil_timestamp'], errors='coerce')
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    if 'confidence' in df.columns:
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
        df.loc[df['confidence'] < 0.8, target_col] = np.nan

    df = df.dropna(subset=['pupil_timestamp', target_col])
    if df.empty: return None, None

    pupil_series = df.groupby('pupil_timestamp')[target_col].mean()
    t = pupil_series.index.values 
    vals = pupil_series.values

    valid_idx = ~np.isnan(vals) & (vals > 0)
    if valid_idx.sum() < fs: return None, None

    f_intp = interp1d(t[valid_idx], vals[valid_idx], bounds_error=False, fill_value='extrapolate')
    t_start = t[0] 
    t_end = t[-1]
    t_reg = np.arange(t_start, t_end, 1/fs)
    pupil_reg = f_intp(t_reg)

    q1, q99 = np.percentile(pupil_reg, [1, 99])
    pupil_reg = np.clip(pupil_reg, q1, q99)

    pupil_smooth = gaussian_filter1d(pupil_reg, sigma=2)

    if np.std(pupil_smooth) == 0: return None, None

    pupil_z = (pupil_smooth - np.mean(pupil_smooth)) / np.std(pupil_smooth)

    return pupil_z, t_start

# ==============================================================================
# EVENT EXTRACTION
# ==============================================================================
def extract_trials(events_filepath):
    try:
        events = pd.read_csv(events_filepath, sep='\t', quoting=3, on_bad_lines='skip')
    except Exception:
        return None

    events['label_str'] = events['label'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.replace('"', '')
    target_triggers = {'500105': 5, '500109': 9, '500113': 13}
    
    trials = []
    for _, row in events.iterrows():
        lbl = row['label_str']
        if lbl in target_triggers:
            trials.append({
                'onset': float(row['timestamp']),
                'load': target_triggers[lbl]
            })
            
    return pd.DataFrame(trials)

# ==============================================================================
# METRICS CALCULATION
# ==============================================================================
def calculate_metrics(segment_baseline, segment_task, fs=250):
    variance = np.var(segment_baseline)

    seg_down = segment_baseline[::DOWNSAMPLE_FACTOR]
    if len(seg_down) > 2 and np.std(seg_down) > 0:
        x = seg_down[:-1]
        y = seg_down[1:]
        ac1_raw = np.corrcoef(x, y)[0, 1]
        n = len(seg_down)
        ac1 = ac1_raw * ((n - 1) / n)
    else:
        ac1 = np.nan

    f_freq, Pxx = welch(segment_baseline, fs=fs, nperseg=min(len(segment_baseline), fs*2))
    valid_f = (f_freq >= 0.1) & (f_freq <= 10.0)
    if np.sum(valid_f) > 2:
        coeffs = np.polyfit(np.log10(f_freq[valid_f]), np.log10(Pxx[valid_f]), 1)
        beta = -coeffs[0] 
    else:
        beta = np.nan

    baseline_mean = np.mean(segment_baseline)
    phasic = np.max(segment_task) - baseline_mean

    return variance, ac1, phasic, beta

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
def run_pipeline():
    print("Starting Empirical Consistency Validation Pipeline...")
    physio_files = sorted(glob.glob(os.path.join(BIDS_ROOT_DIR, 'sub-*', 'pupil', '*_task-memory_pupil.tsv*')))
    results = []

    for physio_path in tqdm(physio_files, desc="Processing Subjects"):
        sub_id = os.path.basename(physio_path).split('_')[0]
        if sub_id in EXCLUDE_SUBJECTS: continue

        dir_name = os.path.dirname(physio_path)
        task_prefix = os.path.basename(physio_path).split('_pupil')[0]
        events_path = os.path.join(dir_name, f"{task_prefix}_events.tsv")
        
        if not os.path.exists(events_path): continue

        pupil_z, t_start = preprocess_pupil(physio_path, FS)
        events_df = extract_trials(events_path)
        
        if pupil_z is None or events_df is None or events_df.empty: 
            gc.collect() 
            continue

        trial_counter = 0 
        for _, trial in events_df.iterrows():
            onset_unix = trial['onset']
            
            idx_start_baseline = max(0, int((onset_unix - 2.0 - t_start) * FS))
            idx_onset = min(len(pupil_z), int((onset_unix - t_start) * FS))
            idx_end_task = min(len(pupil_z), int((onset_unix + 2.0 - t_start) * FS))
            
            if (idx_onset - idx_start_baseline) >= int(FS * 1.5) and (idx_end_task - idx_onset) >= int(FS * 1.5):
                segment_baseline = pupil_z[idx_start_baseline:idx_onset]
                segment_task = pupil_z[idx_onset:idx_end_task]
                
                var, ac1, phasic, beta = calculate_metrics(segment_baseline, segment_task, FS)
                
                results.append({
                    'subject': sub_id,
                    'trial_index': trial_counter,
                    'load': trial['load'],
                    'variance': var,
                    'ac1': ac1,
                    'phasic': phasic,
                    'beta': beta
                })
            trial_counter += 1
            
        del pupil_z, events_df
        gc.collect()

    return pd.DataFrame(results)

# ==============================================================================
# STATISTICS & PLOTTING
# ==============================================================================
def perform_statistics(df):
    print("\n" + "="*50)
    print("Linear Mixed Effects Model (LMM) Results: Consistency Check")
    print("="*50)
    metrics = ['variance', 'ac1', 'phasic', 'beta']
    
    for metric in metrics:
        df_valid = df.dropna(subset=[metric, 'load', 'subject'])
        if len(df_valid) == 0: continue
            
        try:
            model = smf.mixedlm(f"{metric} ~ load", data=df_valid, groups=df_valid["subject"], re_formula="~load")
            result = model.fit()
            print(f"\n[ Dependent Variable: {metric.upper()} ]")
            print(result.summary().tables[1])
        except Exception:
            try:
                model_fallback = smf.mixedlm(f"{metric} ~ load", data=df_valid, groups=df_valid["subject"])
                result_fallback = model_fallback.fit()
                print(f"\n[ Dependent Variable: {metric.upper()} (Fallback Model) ]")
                print(result_fallback.summary().tables[1])
            except:
                pass

def plot_results(df):
    metrics = {
        'variance': (r'Baseline Variance ($\langle \delta x^2 \rangle$)', 'Figure_S2_Variance.png'),
        'ac1': (r'Baseline Autocorrelation ($\tau_{relax}$)', 'Figure_S3_AC1.png'),
        'phasic': (r'Task Phasic Response ($\gamma$)', 'Figure_S4_Phasic.png'),
        'beta': (r'Baseline Spectral Slope ($\beta$)', 'Figure_S5_Beta.png')
    }

    for col, (ylabel, filename) in metrics.items():
        df_plot = df.dropna(subset=[col, 'load'])
        if df_plot.empty: continue
        
        plt.figure(figsize=(6, 5))
        sns.regplot(data=df_plot, x='load', y=col, scatter=False, color='red', line_kws={"linestyle": "--", "alpha":0.6})
        sns.pointplot(data=df_plot, x='load', y=col, capsize=.1, errorbar='se', color='black', markers='o')
        plt.xlabel('Memory Load (Number of Digits)', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f'Scaling Trend: {col.capitalize()} vs Load', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

# ==============================================================================
# Analysis Pipeline v1.3: Absolute Trial-Index Trend Analysis
# ==============================================================================
def execute_pipeline_v1_3(df, n_surrogates=1000):
    print("\n" + "="*50)
    print("Executing Analysis Pipeline v1.3: Absolute Trial-Index Trend Analysis")
    print("="*50)
    
    metrics = ['variance', 'ac1', 'phasic']
    empirical_taus = {m: [] for m in metrics}
    valid_subjects = []

    # 1. 経験的データのトレンド抽出（被験者レベル）
    for subj in df['subject'].unique():
        df_sub = df[df['subject'] == subj].sort_values('trial_index').copy()
        df_sub = df_sub.dropna(subset=metrics)
        
        # 統計的信頼性を確保するための最小試行数制約
        if len(df_sub) < 15:
            continue
            
        valid_subjects.append(subj)
        for m in metrics:
            tau, _ = stats.kendalltau(df_sub['trial_index'], df_sub[m])
            if not np.isnan(tau):
                empirical_taus[m].append(tau)

    # 2. グループレベルの検定
    results = []
    for m in metrics:
        taus = np.array(empirical_taus[m])
        if len(taus) == 0:
            continue
            
        median_tau = np.median(taus)
        stat, p_val_wilcoxon = stats.wilcoxon(taus)
        
        results.append({
            'Metric': m,
            'N_Subjects': len(taus),
            'Median_Tau': median_tau,
            'P_Wilcoxon': p_val_wilcoxon
        })

    df_stats = pd.DataFrame(results)
    if df_stats.empty: return df_stats, empirical_taus
    
    print("\n[Empirical Group-Level Statistics]")
    print(df_stats.to_string(index=False))

    # 3. サロゲート検定 (Null Model: 時系列順序のランダム化)
    print(f"\nRunning Surrogate Permutation Test (N={n_surrogates})...")
    surrogate_p_values = {}
    
    for m in metrics:
        if m not in df_stats['Metric'].values: continue
        emp_median_tau = df_stats.loc[df_stats['Metric'] == m, 'Median_Tau'].values[0]
        surrogate_median_taus = []
        
        for _ in tqdm(range(n_surrogates), desc=f"Surrogate {m}"):
            surr_taus = []
            for subj in valid_subjects:
                df_sub = df[df['subject'] == subj].dropna(subset=[m]).copy()
                if len(df_sub) < 15: continue
                
                shuffled_metric = np.random.permutation(df_sub[m].values)
                tau_surr, _ = stats.kendalltau(df_sub['trial_index'], shuffled_metric)
                
                if not np.isnan(tau_surr):
                    surr_taus.append(tau_surr)
                    
            surrogate_median_taus.append(np.median(surr_taus))
            
        surrogate_median_taus = np.array(surrogate_median_taus)
        
        if emp_median_tau > 0:
            p_surr = np.sum(surrogate_median_taus >= emp_median_tau) / n_surrogates
        else:
            p_surr = np.sum(surrogate_median_taus <= emp_median_tau) / n_surrogates
            
        surrogate_p_values[m] = p_surr

    df_stats['P_Surrogate'] = df_stats['Metric'].map(surrogate_p_values)
    print("\n[Final Results with Surrogate Testing]")
    print(df_stats.to_string(index=False))
    
    return df_stats, empirical_taus

# ==============================================================================
# 実行ブロック
# ==============================================================================
if __name__ == "__main__":
    try:
        df_results = run_pipeline()
        if not df_results.empty:
            perform_statistics(df_results)
            plot_results(df_results)
            print(f"\n[SUCCESS] 計 {len(df_results)} 件の試行に基づく経験的整合性の解析および画像出力が完了しました。")
            
            # --- SAIMの理論的実証（Fold分岐）の追加実行 ---
            df_stats, emp_taus = execute_pipeline_v1_3(df_results)
            
        else:
            print("\n[ERROR] 有効なデータが抽出されませんでした。")
    except Exception as e:
        print("\n[エラー発生]")
        print(traceback.format_exc())

    input("\n[処理完了] 結果を確認したらEnterキーを押して終了してください...")