import pandas as pd
import numpy as np
import pywt
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy.signal import filtfilt
from statistics import mean, variance

fs = 512
lcf = 0.05
hcf = 100

# path='/Users/polinaturiseva/Downloads/mammals_dataset/text_format'
path = ''


def get_pathes(path_train):
    files = []
    for r, d, f in os.walk(path_train):
        for file in f:
            files.append(os.path.join(r, file))
    return files


# folders - core folders with animal names
# subfolders - subfolders with recordings for each animal
def get_folders_and_subfolders():
    folders = os.listdir(path)
    subfolder = []
    for i in folders:
        subfolder.append(os.listdir(path + '/' + i))
    return folders, subfolder


def folder_and_subfolders_for_animal(animal):
    folders = os.listdir(path)
    folder = ""
    for i in folders:
        if animal in i:
            folder = i
    subfolder = (os.listdir(path + '/' + folder))

    return folder, subfolder


def upload_file(folder, subfolder_name, reverse=False):
    way = path + '/' + folder + '/' + subfolder_name + '/electrography_' + subfolder_name + '.txt'
    if reversed:
        file = np.loadtxt(way, skiprows=14) * (-1)
    else:
        file = np.loadtxt(way, skiprows=14)
    return file


def butterworth_filtering(lcf, hcf, fs, ecg):
    # removes all frequencies less 0.05 Hz
    [b_bas, a_bas] = butter(2, lcf / (fs / 2), 'high')
    # removes all frequencies greater 100 Hz
    [b_lp, a_lp] = butter(5, hcf / (fs / 2), 'high')
    bpfecg = ecg - filtfilt(b_lp, a_lp, ecg)
    bpfecg = filtfilt(b_bas, a_bas, bpfecg)
    return bpfecg


def get_rid_of_artifacts(arr, lim):
    final_arr = []
    to_drop, _ = find_peaks(arr, height=lim)
    start_point = []
    end_points = []
    for i in range(0, len(to_drop)):
        if start_point == to_drop[i]:
            continue
        final_arr.append(arr[start_point:to_drop[i] - 1])
        start_point = to_drop[i] + 1
    return final_arr


# https://dsp.stackexchange.com/questions/47437/discrete-wavelet-transform-visualizing-relation-between-decomposed-detail-coef
# функция для вивлетов брались из примера по ссылке выше
# попробовать вейвлет Daubechies 8

# find p should be false for mice
def find_q_s_t_p(r, arr, lev, animal, find_p=True, dist=None):
    q = []
    s = []
    theta_peak = []
    p_peak = []
    all_peaks = find_peaks(arr)
    is_first = True
    ecg_after_wavelet = wavelets(arr)
    s_pos = 0
    for i in r:
        pos = i
        while arr[pos] >= arr[pos - 1]:
            pos = pos - 1
        q.append(pos)
        pos = i
        while pos + 1 != len(arr) and arr[pos] >= arr[pos + 1]:
            pos = pos + 1
        if pos + 1 == len(arr):
            break
        s.append(pos)

        if not is_first and animal != 'mouse':
            #             ecg_after_wavelet = wavelets(arr[prev:cur])
            # indexes from [0; cur-prev]
            cur = q[s_pos]
            #             wave_mins, _ = find_peaks(ecg_after_wavelet*(-1))
            sub_ar_for_theta_peak = list(arr[prev:int(round((cur - prev) * 0.6) + prev)])
            sub_ar_for_p_peak = list(arr[int(round((cur - prev) * 0.6) + prev):cur])
            theta_peak.append(sub_ar_for_theta_peak.
                              index(max(sub_ar_for_theta_peak)) + prev)
            p_peak.append(int(sub_ar_for_p_peak.
                              index(max(sub_ar_for_p_peak)) + prev + round((cur - prev) * 0.6)))

        prev = s[s_pos]
        s_pos += 1
        is_first = False
    return np.array(q), np.array(s), np.array(theta_peak), np.array(p_peak)


def wavelets(ecg, lev_to_decompose=6, lev_to_recontr=3):
    decontr = pywt.wavedec(ecg, 'sym4', 'ppd', lev_to_decompose)
    #     reconstr = pywt.waverec(rabbit_ex_wav[:-lev_to_recontr] + [None] * lev_to_recontr, 'sym4')
    reconstr = pywt.waverec(decontr[:-lev_to_recontr] + [None] * lev_to_recontr, 'sym4')
    return reconstr


# умножение на маску 1, -1, чтобы ложно не определялись нижние пики
def find_r(ecg, animal, dist=None):
    ecg_ar = np.array(ecg)
    ecg = list(ecg)
    sign = np.where(ecg_ar > 0, 1, -1)
    ecg_sq = [item * item for item in ecg]
    ecg_sq = ecg_sq * sign

    #     peaks, _ = find_peaks(ecg_sq, distance=80)
    if dist is not None:
        peaks, _ = find_peaks(ecg_sq, distance=dist)
    elif animal == 'mouse':
        peaks, _ = find_peaks(ecg_sq, distance=80)
    elif animal == 'human':
        peaks, _ = find_peaks(ecg_sq, distance=250)
    else:
        peaks, _ = find_peaks(ecg_sq, distance=100, height=0.02)

    amp = [ecg[int(i)] for i in peaks]
    if len(amp) == 0:
        return None, None, None, None
    av = mean(amp)
    if len(amp) > 1:
        var = variance(amp)
    else:
        var = amp[0]
    if av < 0:
        ecg = [i * (-1) for i in ecg]
        return find_r(ecg, animal, dist)
    else:
        return peaks, av, var, np.array(ecg)

    # it will remove 2 seconds before and after artifact as well


def filt_electrodes(ecg, av, var):
    indexes_to_drop = np.where(abs(ecg) > abs(av) + abs(var))[0]
    start_drop = []
    end_drop = []
    pos = 0
    while pos < (len(indexes_to_drop) - 1):
        if indexes_to_drop[pos] - 2 * fs < 0:
            start_drop.append(0)
        else:
            start_drop.append(indexes_to_drop[pos] - 2 * fs)

        while (len(indexes_to_drop) - 1) > pos and indexes_to_drop[pos + 1] - indexes_to_drop[pos] < 100:
            pos += 1
        if indexes_to_drop[pos] + 2 * fs >= len(ecg):
            end_drop.append(len(ecg))
            break
        else:
            end_drop.append(indexes_to_drop[pos] + 2 * fs)
        pos += 1

    ecg_filtered = []
    for i in range(len(end_drop) - 1):
        if i == 0:
            if start_drop[0] < 100:
                continue
            else:
                ecg_filtered.append(ecg[0:start_drop[0]])
        else:
            ecg_filtered.append(ecg[end_drop[i - 1]:start_drop[i]])

    if len(end_drop) > 0:
        if end_drop[-1] == len(ecg) or len(ecg) - end_drop[-1] < 100:
            pass
        else:
            ecg_filtered.append(ecg[end_drop[-1]:len(ecg)])

    return ecg_filtered, len(end_drop), start_drop, end_drop


def step_by_step(folder, subfolder_name, animal, lim=1, level=6, qrs_approx=30, file_init=None, to_filter=True,
                 dist=None):
    file = file_init
    if folder is not None and subfolder_name is not None:
        file = upload_file(folder=folder, subfolder_name=subfolder_name)

    file = butterworth_filtering(lcf=lcf, hcf=hcf, fs=fs, ecg=file)
    #     arr = get_rid_of_artifacts(file, lim)
    arr = file
    preprocessed_arr = []
    r_peaks = []
    q_peaks = []
    s_peaks = []
    theta_peaks = []
    p_peaks = []
    array = []
    r_p, r_av, r_var, arr = find_r(arr, animal=animal, dist=dist)
    drops = None
    start_drops = None
    end_drops = None
    if to_filter:
        arr, drops, start_drops, end_drops = filt_electrodes(arr, r_av, r_var)
    else:
        arr = [arr]
    for i in range(len(arr)):
        if len(arr[i]) == 0:
            continue
        #         preprocessed_arr = preprocessed_arr.append(pywt.wavedec(arr, 'sym4', 'ppd', level))
        #         print(preprocessed_arr)
        r_, av_h, std, _ = find_r(arr[i], animal=animal, dist=dist)
        if r_ is None:
            continue
        r_peaks.append(r_)
        q_, s_, theta_, p_ = find_q_s_t_p(r_, arr[i], animal=animal, lev=level, dist=dist)
        q_peaks.append(q_)
        s_peaks.append(s_)
        theta_peaks.append(theta_)
        p_peaks.append(p_)
        array.append(arr[i])
    return q_peaks, s_peaks, theta_peaks, p_peaks, r_peaks, drops, start_drops, end_drops, array


def shifted_points(points, loc, start, shift):
    points_ = np.array(points)
    loc_points = np.array(np.where((points_ < start + shift) & (points_ > start)))
    x = (points_[loc_points] - start).reshape(-1)
    y = loc[points_[loc_points] - start].reshape(-1)
    return x, y


def find_amplitudes(peaks, ecg):
    amp = [ecg[i] for i in peaks]
    return amp


# period of wave
def find_period(peaks):
    dist = []
    for i, j in zip(peaks[:len(peaks) - 1], peaks[1:]):
        dist.append(j - i)
    return dist


# for finding distances between peaks, intervals or segments
# peaks_one should be logically before peaks_two
def find_distance(peaks_one, peaks_two):
    dist = []
    if peaks_one[0] >= peaks_two[0]:
        m = min(len(peaks_one), len(peaks_two) - 1)
        for i in range(m):
            dist.append(peaks_two[i + 1] - peaks_one[i])
    else:
        m = min(len(peaks_one), len(peaks_two) - 1)
        for i in range(m):
            dist.append(peaks_two[i + 1] - peaks_one[i])

    return dist


# will be saved in local directory 
def draw_graph(filtered_ecg, q_ar=[], s_ar=[], r_ar=[], p_ar=[], t_ar=[],
               p=True, theta=True, q=True, r=True, s=True,
               save=False, name=None, start=0, shift=2500):
    plt.figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(np.zeros(shift, ))

    plt.plot(filtered_ecg[start:start + shift], 'blue')
    loc = filtered_ecg[start:start + shift]

    if (len(q_ar) > 0) and q:
        x, y = shifted_points(q_ar, loc, start, shift)
        plt.plot(x, y, 'o', label='q')

    if (len(r_ar) > 0) and r:
        x, y = shifted_points(r_ar, loc, start, shift)
        plt.plot(x, y, 'o', label='r')

    if (len(s_ar) > 0) and s:
        x, y = shifted_points(s_ar, loc, start, shift)
        plt.plot(x, y, 'o', label='s')

    if (len(p_ar) > 0) and p:
        x, y = shifted_points(p_ar, loc, start, shift)
        plt.plot(x, y, 'o', label='p')

    if len(t_ar) > 0 and theta:
        x, y = shifted_points(t_ar, loc, start, shift)
        plt.plot(x, y, 'o', label='t')

    plt.legend()
    if name is not None:
        plt.title(name)

    if save:
        plt.savefig(name)


def flatten_list(l):
    lst = [item for sublist in l for item in sublist]
    return lst


def filter_empty(l):
    lst = []
    for i in l:
        if len(i) > 0:
            lst.append(i)
    return lst


# for one animal
def animal_describe(q_peaks, s_peaks, theta_peaks, p_peaks, r_peaks, arr,
                    p=True, theta=True, q=True, r=True, s=True):
    q_amp = []
    r_amp = []
    s_amp = []
    t_amp = []
    p_amp = []

    p_peaks_interval = []
    q_peaks_interval = []
    r_peaks_interval = []
    s_peaks_interval = []
    t_peaks_interval = []

    qrs_interval = []
    qr_interval = []
    rs_interval = []
    st_interval = []
    pq_interval = []

    for i in range(len(arr)):
        if (len(p_peaks[i]) > 0) and p:
            p_amp.append(find_amplitudes(p_peaks[i], arr[i]))
            p_peaks_interval.append(find_period(p_peaks[i]))
        if (len(q_peaks[i]) > 0) and q:
            q_amp.append(find_amplitudes(q_peaks[i], arr[i]))
            q_peaks_interval.append(find_period(q_peaks[i]))
        if (len(r_peaks[i]) > 0) and r:
            r_amp.append(find_amplitudes(r_peaks[i], arr[i]))
            r_peaks_interval.append(find_period(r_peaks[i]))
        if (len(theta_peaks[i]) > 0) and theta:
            t_amp.append(find_amplitudes(theta_peaks[i], arr[i]))
            t_peaks_interval.append(find_period(theta_peaks[i]))
        if (len(s_peaks[i]) > 0) and s:
            s_amp.append(find_amplitudes(s_peaks[i], arr[i]))
            s_peaks_interval.append(find_period(s_peaks[i]))

        if len(q_peaks[i]) > 0 and len(s_peaks[i]) > 0 and q and theta:
            qrs_interval.append(find_distance(q_peaks[i], s_peaks[i]))
        if len(q_peaks[i]) > 0 and len(r_peaks[i]) > 0 and r and q:
            qr_interval.append(find_distance(q_peaks[i], r_peaks[i]))
        if len(r_peaks[i]) > 0 and len(s_peaks[i]) > 0 and r and s:
            rs_interval.append(find_distance(r_peaks[i], s_peaks[i]))
        if len(s_peaks[i]) > 0 and len(theta_peaks[i]) > 0 and s and theta:
            st_interval.append(find_distance(s_peaks[i], theta_peaks[i]))
        if len(p_peaks[i]) > 0 and len(q_peaks[i]) > 0 and p and q:
            pq_interval.append(find_distance(p_peaks[i], q_peaks[i]))

    p_amp = flatten_list(p_amp)
    q_amp = flatten_list(q_amp)
    r_amp = flatten_list(r_amp)
    s_amp = flatten_list(s_amp)
    t_amp = flatten_list(t_amp)

    p_peaks_interval = flatten_list(p_peaks_interval)
    q_peaks_interval = flatten_list(q_peaks_interval)
    r_peaks_interval = flatten_list(r_peaks_interval)
    s_peaks_interval = flatten_list(s_peaks_interval)
    t_peaks_interval = flatten_list(t_peaks_interval)

    qrs_interval = flatten_list(qrs_interval)
    qr_interval = flatten_list(qr_interval)
    rs_interval = flatten_list(rs_interval)
    st_interval = flatten_list(st_interval)
    pq_interval = flatten_list(pq_interval)

    p_stats = []
    q_stats = []
    r_stats = []
    s_stats = []
    t_stats = []
    p_interval_stats = []
    q_interval_stats = []
    r_interval_stats = []
    s_interval_stats = []
    t_interval_stats = []
    qrs_stats = []
    qr_stats = []
    rs_stats = []
    st_stats = []
    pq_stats = []

    if len(p_amp) > 0:
        p_stats = pd.DataFrame(p_amp).describe()
    if len(p_peaks_interval) > 0:
        p_interval_stats = pd.DataFrame(p_peaks_interval).describe()
    if len(q_amp) > 0:
        q_stats = pd.DataFrame(q_amp).describe()
    if len(q_peaks_interval) > 0:
        q_interval_stats = pd.DataFrame(q_peaks_interval).describe()
    if len(r_amp) > 0:
        r_stats = pd.DataFrame(r_amp).describe()
    if len(r_peaks_interval) > 0:
        r_interval_stats = pd.DataFrame(r_peaks_interval).describe()
    if len(s_amp) > 0:
        s_stats = pd.DataFrame(s_amp).describe()
    if len(s_peaks_interval) > 0:
        s_interval_stats = pd.DataFrame(s_peaks_interval).describe()
    if len(t_amp) > 0:
        t_stats = pd.DataFrame(t_amp).describe()
    if len(t_peaks_interval) > 0:
        t_interval_stats = pd.DataFrame(t_peaks_interval).describe()

    if len(qrs_interval) > 0:
        qrs_stats = pd.DataFrame(qrs_interval).describe()
    if len(qr_interval) > 0:
        qr_stats = pd.DataFrame(qr_interval).describe()
    if len(rs_interval) > 0:
        rs_stats = pd.DataFrame(rs_interval).describe()
    if len(pq_interval) > 0:
        pq_stats = pd.DataFrame(pq_interval).describe()
    if len(st_interval) > 0:
        st_stats = pd.DataFrame(st_interval).describe()

    raw = [p_amp, q_amp, r_amp, s_amp, t_amp,
           p_peaks_interval, q_peaks_interval, r_peaks_interval, s_peaks_interval, t_peaks_interval,
           qrs_interval, qr_interval, rs_interval, st_interval, pq_interval]

    stats = [p_stats, q_stats, r_stats, s_stats, t_stats,
             p_interval_stats, q_interval_stats, r_interval_stats, s_interval_stats, t_interval_stats,
             qrs_stats, qr_stats, rs_stats, st_stats, pq_stats]

    return (raw, stats)


# for all animals of one type
def type_describe(animal):
    folder, subfolders = folder_and_subfolders_for_animal(animal)
    animals_p = []
    animals_q = []
    animals_r = []
    animals_s = []
    animals_t = []

    animals_p_interval = []
    animals_q_interval = []
    animals_r_interval = []
    animals_s_interval = []
    animals_t_interval = []

    animals_qrs = []
    animals_qr = []
    animals_rs = []
    animals_st = []
    animals_pq = []

    p = True
    theta = True
    q = True
    r = True
    s = True

    if animal.lower() == 'dog':
        theta = False

    if animal.lower() == 'mice':
        theta = False
        p = False

    for sub in subfolders:
        if sub == 'Mouse_02' or sub == 'Mouse_05' or sub == 'Mouse_07':
            print(f"{sub} this data is damaged")
            continue
        q_peaks, s_peaks, theta_peaks, p_peaks, r_peaks, drops, start_drops, end_drops, arr = step_by_step(
            folder=folder, subfolder_name=sub, animal=animal)
        res = animal_describe(q_peaks, s_peaks, theta_peaks, p_peaks, r_peaks, arr, p, theta, q, r, s)[0]

        animals_p.append(res[0])
        animals_q.append(res[1])
        animals_r.append(res[2])
        animals_s.append(res[3])
        animals_t.append(res[4])

        animals_p_interval.append(res[5])
        animals_q_interval.append(res[6])
        animals_r_interval.append(res[7])
        animals_s_interval.append(res[8])
        animals_t_interval.append(res[9])

        animals_qrs.append(res[10])
        animals_qr.append(res[11])
        animals_rs.append(res[12])
        animals_st.append(res[13])
        animals_pq.append(res[14])

    animals_p = filter_empty(animals_p)
    animals_q = filter_empty(animals_q)
    animals_r = filter_empty(animals_r)
    animals_s = filter_empty(animals_s)
    animals_t = filter_empty(animals_t)

    animals_p_interval = filter_empty(animals_p_interval)
    animals_q_interval = filter_empty(animals_q_interval)
    animals_r_interval = filter_empty(animals_r_interval)
    animals_s_interval = filter_empty(animals_s_interval)
    animals_t_interval = filter_empty(animals_t_interval)

    animals_qrs = filter_empty(animals_qrs)
    animals_qr = filter_empty(animals_qr)
    animals_rs = filter_empty(animals_rs)
    animals_pq = filter_empty(animals_pq)
    animals_st = filter_empty(animals_st)

    if len(animals_p) > 0:
        animals_p = pd.DataFrame(flatten_list(animals_p)).describe()
    if len(animals_q) > 0:
        animals_q = pd.DataFrame(flatten_list(animals_q)).describe()
    if len(animals_r) > 0:
        animals_r = pd.DataFrame(flatten_list(animals_r)).describe()
    if len(animals_s) > 0:
        animals_s = pd.DataFrame(flatten_list(animals_s)).describe()
    if len(animals_t) > 0:
        animals_t = pd.DataFrame(flatten_list(animals_t)).describe()

    if len(animals_p_interval) > 0:
        animals_p_interval = pd.DataFrame(flatten_list(animals_p_interval)).describe()
    if len(animals_q_interval) > 0:
        animals_q_interval = pd.DataFrame(flatten_list(animals_q_interval)).describe()
    if len(animals_r_interval) > 0:
        animals_r_interval = pd.DataFrame(flatten_list(animals_r_interval)).describe()
    if len(animals_s_interval) > 0:
        animals_s_interval = pd.DataFrame(flatten_list(animals_s_interval)).describe()
    if len(animals_t_interval) > 0:
        animals_t_interval = pd.DataFrame(flatten_list(animals_t_interval)).describe()

    if len(animals_qrs) > 0:
        animals_qrs = pd.DataFrame(flatten_list(animals_qrs)).describe()
    if len(animals_qr) > 0:
        animals_qr = pd.DataFrame(flatten_list(animals_qr)).describe()
    if len(animals_rs) > 0:
        animals_rs = pd.DataFrame(flatten_list(animals_rs)).describe()
    if len(animals_st) > 0:
        animals_st = pd.DataFrame(flatten_list(animals_st)).describe()
    if len(animals_pq) > 0:
        animals_pq = pd.DataFrame(flatten_list(animals_pq)).describe()

    return animals_p, animals_q, animals_r, animals_s, animals_t, animals_p_interval, animals_q_interval, \
           animals_r_interval, animals_s_interval, animals_t_interval, animals_qrs, animals_qr, animals_rs, \
           animals_st, animals_pq


def printing(arr, name):
    if len(arr) > 0:
        print(f"\n Statistics for {name}: \n {arr}")
    else:
        print(f"Statistics for parameter {name} was not evaluated")


# will return 1-dimentional array
def upload_dat(filename):
    return np.fromfile(filename, dtype=float)


def basic_stat(ecg, wave):
    amps = [ecg[i] for i in wave]
    return min(amps), max(amps), mean(amps)


def is_normally_detected(ecg, qs, rs, ss, ts, ps):
    min_r, max_r, avg_r = basic_stat(ecg, rs)
    if max_r / min_r > 4:
        return False
    else:
        min_q, max_q, avg_q = basic_stat(ecg, qs)
        min_s, max_s, avg_s = basic_stat(ecg, ss)
        min_t, max_t, avg_t = basic_stat(ecg, ts)
        min_p, max_p, avg_p = basic_stat(ecg, ps)

        if avg_r * 0.7 <= avg_p or avg_r * 0.7 <= avg_t:
            return False
        else:
            return True


def redetect(rs, animal, file=None, folder=None, subfolder=None):
    rs_new = []
    for i in range(1, len(rs)):
        rs_new.append(rs[i] - rs[i - 1])
    new_dist = mean(rs_new) * 1.2
    if animal == "human":
        return step_by_step(folder, subfolder, animal, file_init=file * 1e+269, to_filter=False, dist=new_dist)
    else:
        return step_by_step(folder, subfolder, animal, file_init=file, to_filter=False, dist=new_dist)
