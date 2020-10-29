from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from collections import defaultdict

def get_gender_ids_each_cluster(clusters, labels):
    male_id_c = defaultdict(list)
    female_id_c = defaultdict(list)
    for idx in range(len(labels)):
        c = clusters[idx]
        gender = labels[idx]['gender']
        if gender == [1,0]:
            male_id_c[c].append(idx)
        elif gender == [0, 1]:
            female_id_c[c].append(idx)
        else:
            print("wrong gender")
    return male_id_c, female_id_c

def plot_clusters(clusters, decom, NCluster):
    plt.scatter(decom[:,0], decom[:,1], c=clusters, cmap=plt.cm.get_cmap('tab20', NCluster))
    plt.colorbar(ticks=range(NCluster), label='cluster')
    plt.show()

def get_res_cluster(clusters, labels): #gender and object in each cluster
    res = defaultdict(dict)
    for idx in range(len(clusters)):
        obj = labels[idx]['objects']
        gender = labels[idx]['gender']
        if 'gender' not in res[clusters[idx]]:
            res[clusters[idx]]['gender'] = []
        if 'obj' not in res[clusters[idx]]:
            res[clusters[idx]]['obj'] = []
        res[clusters[idx]]['gender'].append(gender)
        res[clusters[idx]]['obj'].append(obj)
    return res

def get_ratio_cluster(res):
    for c in range(len(res)):
        gender = np.array(res[c]['gender']).sum(axis = 0)
        obj = np.array(res[c]['obj']).sum(axis = 0)
        total_c = len(res[c]['gender'])
        print(f'{c}-cluster, {total_c} instances: {gender[0]} males, {gender[1]} females, m/(m+f) = {gender[0]/(gender[0] + gender[1])}')
    return gender, obj

def show_localbias(kmeans_res, obj_imgs, labels, obj_gt, preds_binary, preds, fobj, objid, mk=''):
    kmeans_clusters = kmeans_res.labels_
    res_cluster_kmeans = get_res_cluster(kmeans_clusters, labels)
    kmeans_ratio = get_ratio_cluster(res_cluster_kmeans)


    # 1. for overall male and female across whole test set; only on one object
    male_ids = [x for x in range(len(labels)) if labels[x]['gender'] == [1,0]]
    female_ids = [x for x in range(len(labels)) if labels[x]['gender'] == [0,1]]
    assert len(male_ids) + len(female_ids) == len(obj_imgs)
    
    f1_m = f1_score(obj_gt[male_ids][:,objid], preds_binary[male_ids][:, objid])
    f1_f = f1_score(obj_gt[female_ids][:,objid], preds_binary[female_ids][:,objid])
    acc_m = metrics.accuracy_score(y_true = obj_gt[male_ids][:,objid], y_pred = preds_binary[male_ids][:, objid])
    acc_f = metrics.accuracy_score(y_true = obj_gt[female_ids][:,objid], y_pred = preds_binary[female_ids][:,objid])
   
    print(f'{fobj}: {len(male_ids)} male instances, {len(female_ids)} female instances')
    print(f'avg performance for `{fobj}`: \
    f1_m:{f1_m:.4f}, f1_f:{f1_f:.4f}, acc_m:{acc_m:.4f}, acc_f:{acc_f:.4f},f1_d:{abs(f1_m - f1_f):.4f}, acc_d:{acc_m-acc_f:.3f}')
    

    #based on vanilla Kmeans
    male_ids_c, female_ids_c = get_gender_ids_each_cluster(kmeans_clusters, labels)
    diffs = get_diff_each_cluster(male_ids_c, female_ids_c, kmeans_clusters, obj_gt, preds_binary, preds, fobj, objid, mk)

def get_diff_each_cluster(male_ids_c, female_ids_c, clusters, obj_gt, preds_binary, preds, fcat, objs_ids = -1, mk = ''):
    diffs = []
    m_ratios = []
    for c in range(len(np.unique(clusters))):
        male_ids = male_ids_c[c]
        female_ids = female_ids_c[c]
        if len(male_ids) < 20 or len(female_ids) < 20:
            flag = 1
            continue
        if objs_ids == -1:
            print("Consider all objects")
            f1_m = f1_score(obj_gt[male_ids], preds_binary[male_ids], average='micro')
            f1_f = f1_score(obj_gt[female_ids], preds_binary[female_ids], average='micro')
        else:
            if len(male_ids) == 0:
                f1_m = 0
            else:
                f1_m = f1_score(obj_gt[male_ids][:, objs_ids], preds_binary[male_ids][:, objs_ids], average='micro')
            if len(female_ids) == 0:
                f1_f = 0
            else:
                f1_f = f1_score(obj_gt[female_ids][:, objs_ids], preds_binary[female_ids][:, objs_ids], average='micro')
            acc_m = metrics.accuracy_score(y_true=obj_gt[male_ids][:, objs_ids], y_pred=preds_binary[male_ids][:, objs_ids])
            acc_f = metrics.accuracy_score(y_true=obj_gt[female_ids][:, objs_ids], y_pred=preds_binary[female_ids][:, objs_ids])
            if np.isnan(acc_m):
                acc_m = 0
            if np.isnan(acc_f):
                acc_f = 0
            conf_m = np.average(preds[male_ids][:, objs_ids])
            conf_f = np.average(preds[female_ids][:, objs_ids])
            c_m = np.sum(preds[male_ids][:, objs_ids] >= 0.5)
            c_f = np.sum(preds[female_ids][:, objs_ids] >= 0.5)
        diff = (f1_m - f1_f)
        diffs.append([diff, (acc_m - acc_f), acc_m, acc_f, len(male_ids), len(female_ids)])
        m_ratios.append(len(male_ids)/ (len(male_ids) + len(female_ids) + 1e-5))
        print(f'c:{c:2}, tot:{len(male_ids) + len(female_ids)}, m:{len(male_ids):2} f:{len(female_ids):2}, m%:{m_ratios[-1]:.3f}\
        f1_m:{f1_m:.4f}, f1_f:{f1_f:.4f}, acc_m:{acc_m:.4f}, acc_f:{acc_f:.4f}| f1_diff:{diff:.4f}, acc_diff:{diffs[-1][1]:.4f} |  high_m:{c_m}, high_f:{c_f}')
    if len(diffs) == 0:
        print("No cluster exists")
        return np.array(diffs)
    reg = LinearRegression()
    print("avg F1 and acc abs_diff:", np.average(abs(np.array(diffs))[:,:2], axis = 0))
    reg.fit(np.array(m_ratios).reshape(-1, 1), abs(np.array(diffs)[:, 1]).reshape(-1, 1))
    Y_preds = reg.predict(np.array(m_ratios).reshape(-1, 1))
    plt.figure(figsize=(15, 4))
    ax = plt.subplot(131)
#     ax.set_aspect(1)
    plt.scatter(m_ratios, abs(np.array(diffs)[:, 1]))
    plt.plot(m_ratios, Y_preds, color='red')
    plt.title(f'd_acc vs m_ratio for `{fcat}`:{reg.coef_}')
    plt.subplot(132)
    plt.scatter(m_ratios, np.array(diffs)[:, 2])
    reg.fit(np.array(m_ratios).reshape(-1, 1), abs(np.array(diffs)[:, 2]).reshape(-1, 1))
    Y_preds = reg.predict(np.array(m_ratios).reshape(-1, 1))
    plt.plot(m_ratios, Y_preds, color='red')
    plt.title(f"m_acc vs m_ratio:{reg.coef_}")
    plt.subplot(133)
    plt.scatter(m_ratios, np.array(diffs)[:, 3])
    reg.fit(np.array(m_ratios).reshape(-1, 1), abs(np.array(diffs)[:, 3]).reshape(-1, 1))
    Y_preds = reg.predict(np.array(m_ratios).reshape(-1, 1))
    plt.plot(m_ratios, Y_preds, color='red')
    plt.title(f"f_acc vs m_ratio: :{reg.coef_}")
    # plt.savefig(f'res_{fcat}{mk}.pdf')
    plt.show()
    return np.array(diffs)

def get_gender_cluster(clusters, labels): #gender in each cluster
    res = defaultdict(dict)
    tmp = []
    for idx in range(len(clusters)):
        gender = labels[idx]['gender']
        if 'gender' not in res[clusters[idx]]:
            res[clusters[idx]]['gender'] = []
        res[clusters[idx]]['gender'].append(gender)
    for c in range(len(res.keys())):
        tmp.append(np.array(res[c]['gender']).sum(axis = 0))
    return tmp

from copy import deepcopy
def merge_clusters(gender_c, kmeans_c, kmeans_clusters, features):
    ng = deepcopy(gender_c)
    kcc = deepcopy(kmeans_c)
    n2o = list(range(len(gender_c)))
    kc = deepcopy(kmeans_clusters)
    while True:
        min_c = np.min(np.array(ng))
        if min_c >= 20:
            print("Done merge for all clusters have at least 20 M/F images")
            break
        if len(ng) <= 5:
            print("Finish merging as only a few clusters left")
            break
        for c in range(len(ng)):
            if ng[c][0] < 20 or ng[c][1] < 20: #<10 m/f examples => merge to the closet cluster
                distances = np.dot(np.delete(kcc, c, 0), kcc[c])
                merge2 = np.argmin(distances)
                if merge2 >= c:
                    merge2 += 1
#                 print(c, merge2)
                ng[merge2] += ng[c]
                ng[c] = ng[merge2]
                kc[np.where(kc == n2o[c])] = n2o[merge2]
#                 print(kc)
                kcc[merge2] = np.mean(features[np.where(kc == n2o[merge2])], axis = 0)
                ng.pop(c)
                kcc = np.delete(kcc, c, 0)
#                 print(ng, len(ng), len(kcc), np.unique(kc))
                n2o.pop(c)
                break
    #remap the clusters to avoid skipped values
    oc2nc = {}
    for idx in range(len(np.unique(kc))):
        oc2nc[np.unique(kc)[idx]] = idx
    kc = [oc2nc[x] for x in kc]
    return ng, len(np.unique(kc)), np.array(kc)

def load_poss_objs():
    possible_objects = []
    with open('testCountLarger100', 'r') as f:
        for line in f.readlines()[1:]:
            tokens = line.strip().split(',')
            if int(tokens[-1]) > 50 and int(tokens[-2]) > 50: #female>50
                possible_objects.append(tokens[0])
    print(f"{len(possible_objects)} objs")
    return possible_objects