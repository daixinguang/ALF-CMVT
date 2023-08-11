from utils.filter import *
from utils.visual_rr import visual_rerank
from sklearn.cluster import AgglomerativeClustering
import sys
sys.path.append('../../../')
from config import cfg
import numpy as np
import pandas as pd
import scipy
import torch
from TSMatchNet.TSMatch_training.model import TSMatchNet

def bspline_interpolate_t(bev, min_thr=5, point_nums=30):
    if max(bev.shape) < min_thr: 
        return bev
    min_w, max_w = min(bev[1]), max(bev[1])
    min_h, max_h = min(bev[2]), max(bev[2])
    width = max_w - min_w
    height = max_h - min_h
    if width > height:
        x_tmp = bev[1]
        y_tmp = bev[2]
    else:
        x_tmp = bev[2]
        y_tmp = bev[1]
    # sort xtmp ytmp
    x_tmp_sorted = []
    y_tmp_sorted = []

    sorted_idx = np.argsort(x_tmp)
    ts, te = min(bev[0]), max(bev[0])
    if bev[0, sorted_idx[0]] - ts > bev[0, sorted_idx[-1]] - ts: # ts和排序后的最小值对应的time match，则不用颠倒顺序,反之要颠倒顺序
        ts, te = te, ts
    for idx in sorted_idx:
        x_tmp_sorted.append(x_tmp[idx])
        y_tmp_sorted.append(y_tmp[idx])
    # print(len(x_tmp_sorted))
    spl = scipy.interpolate.splrep(x=x_tmp_sorted, y=y_tmp_sorted, s=1000*len(x_tmp_sorted)) # 为保证平滑，s越大越好，但是建议和输入点的个数相关。default s= len(x_tmp_sorted) - math.sqrt(2*len(x_tmp_sorted))
    bev_x_bs = np.linspace(x_tmp_sorted[0], x_tmp_sorted[-1], point_nums)
    time_bs = np.linspace(ts, te, point_nums)
    bev_y_bs = scipy.interpolate.splev(bev_x_bs, spl)
    if width > height:
        return np.vstack((time_bs, bev_x_bs, bev_y_bs))
    else:
        return np.vstack((time_bs, bev_y_bs, bev_x_bs))
    

def _calc_bev_pos(src_point_M, warpMatrix):
    one_column = np.ones((len(src_point_M), 1))
    src_point_M = np.column_stack((src_point_M, one_column))
    tar_pt = np.matmul(src_point_M, warpMatrix.T)
    tar_pt[:, 0] = tar_pt[:, 0] / tar_pt[:, 2]
    tar_pt[:, 1] = tar_pt[:, 1] / tar_pt[:, 2]
    return tar_pt[:, 0:2]

def get_sim_matrix_TSMatch(_cfg,cid_tid_dict,cid_tids, use_st_filter=True):
    count = len(cid_tids)
    print('count: ', count)

    # 相机的投影矩阵
    c001_hom_p = r'../../../TSMatchNet/trans_cfg/c001_trans_bev.npy'
    c002_hom_p = r'../../../TSMatchNet/trans_cfg/c002_trans_bev.npy'
    c003_hom_p = r'../../../TSMatchNet/trans_cfg/c003_trans_bev.npy'
    c004_hom_p = r'../../../TSMatchNet/trans_cfg/c004_trans_bev.npy'
    c001_hom = np.load(c001_hom_p) 
    c002_hom = np.load(c002_hom_p)
    c003_hom = np.load(c003_hom_p)
    c004_hom = np.load(c004_hom_p)
    
    q_cam = [cid_tid_dict[cid_tids[i]]['cam'] for i in range(count)]
    q_frame = [cid_tid_dict[cid_tids[i]]['frame_list'] for i in range(count)]
    q_tracklet = [cid_tid_dict[cid_tids[i]]['tracklet'] for i in range(count)]
    tracklets = []
    for i in range(count):
        cam = q_cam[i] # 当前ID所属cam
        tracklet = q_tracklet[i] # 当前ID对应的tracklet
        
        for frame in tracklet.keys():
            bbox = tracklet[frame]['bbox']
            tid = tracklet[frame]['id']
            tracklets_infos = [] # [cam,frame, x1,y1,x2,y2]
            tracklets_infos.append(cam) 
            tracklets_infos.append(frame)
            tracklets_infos.append(tid)
            tracklets_infos.extend(bbox)
            tracklets.append(tracklets_infos)
    tracklets = np.asarray(tracklets)
    tracklets = pd.DataFrame(tracklets)
    tracklets.columns = ['cam', 'frame', 'ID', 'x1', 'y1', 'x2', 'y2']
    # 计算bx,by
    tracklets_box = tracklets[['x1', 'y1', 'x2', 'y2']].values
    tracklets_box_x = tracklets_box[:, 0] + (tracklets_box[:, 2] - tracklets_box[:, 0]) / 2
    tracklets_box_y = tracklets_box[:, 1] + (tracklets_box[:, 3] - tracklets_box[:, 1]) / 4 * 3
    tracklets['px'] = tracklets_box_x
    tracklets['py'] = tracklets_box_y
    tracklets_cam1 = tracklets[tracklets['cam'] == 1]
    tracklets_cam2 = tracklets[tracklets['cam'] == 2]
    tracklets_cam3 = tracklets[tracklets['cam'] == 3]
    tracklets_cam4 = tracklets[tracklets['cam'] == 4]
    tracklets_out = pd.DataFrame(columns=['cam', 'frame', 'ID', 'x1', 'y1', 'x2', 'y2','px','py', 'bx', 'by'])
    
    # 帧同步
    fps=10
    timestamp = [0, 1.640, 2.049, 2.177, 2.235]
    timestamp = list(map(lambda x: x*10, timestamp))

    # 拼接
    if not tracklets_cam1.empty:
        tar_pt_M1 = _calc_bev_pos(tracklets_cam1[['px', 'py']].values, c001_hom)
        tracklets_cam1[['bx', 'by']] = tar_pt_M1
        tracklets_cam1['frame'] = (tracklets_cam1['frame'] + timestamp[0]) / 10
        tracklets_out = pd.concat([tracklets_out, tracklets_cam1], axis=0, ignore_index=True, join = "outer")
    if not tracklets_cam2.empty:
        tar_pt_M2 = _calc_bev_pos(tracklets_cam2[['px', 'py']].values, c002_hom)
        tracklets_cam2[['bx', 'by']] = tar_pt_M2
        tracklets_cam2['frame'] = (tracklets_cam2['frame'] + timestamp[1]) / 10
        tracklets_out = pd.concat([tracklets_out, tracklets_cam2], axis=0, ignore_index=True, join = "outer")
    if not tracklets_cam3.empty:
        tar_pt_M3 = _calc_bev_pos(tracklets_cam3[['px', 'py']].values, c003_hom)
        tracklets_cam3[['bx', 'by']] = tar_pt_M3
        tracklets_cam3['frame'] = (tracklets_cam3['frame'] + timestamp[2]) / 10
        tracklets_out = pd.concat([tracklets_out, tracklets_cam3], axis=0, ignore_index=True, join = "outer")
    if not tracklets_cam4.empty:
        tar_pt_M4 = _calc_bev_pos(tracklets_cam4[['px', 'py']].values, c004_hom)
        tracklets_cam4[['bx', 'by']] = tar_pt_M4
        tracklets_cam4['frame'] = (tracklets_cam4['frame'] + timestamp[3]) / 10
        tracklets_out = pd.concat([tracklets_out, tracklets_cam4], axis=0, ignore_index=True, join = "outer")
    
    # tracklets = pd.concat([tar_pt_M1, tar_pt_M2, tar_pt_M3, tar_pt_M4], axis=0, ignore_index=True, join = "outer")
    
    
    # # 使用BSpline对tracklet进行规范化
    '''
    {
        tid: array([cam, frame, ])
    }
    '''
    output = {}
    tracklets_ids = np.unique(tracklets_out['ID'].values)
    for idx, id in enumerate(tracklets_ids):
        tracklets_out_tid_pd = tracklets_out[tracklets_out['ID'] == id]
        cam_ids = np.unique(tracklets_out_tid_pd['cam'].values)
        for cam in cam_ids:
            tracklets_out_cid_tid_pd = tracklets_out_tid_pd[tracklets_out_tid_pd['cam']==cam]
            bev = tracklets_out_cid_tid_pd[['frame','bx', 'by']].values.T
            bev_bs = bspline_interpolate_t(bev, min_thr=5, point_nums=30) # frame也进行了插值 frame, bx, by
            time_tmp = np.ones((1, bev_bs.shape[1]))*int(cam)
            bev_bs = np.vstack((time_tmp, bev_bs)).T
            
            # tmp = np.hstack((S01_cid_tid_pd, bev_bs))
            if bev_bs.shape[0] < 30:
                bev_bs = -1 * np.ones((30,4))
            cid_tid = cam*10000 + id
            if cid_tid in output:
                output[cid_tid] = np.vstack((output[id], bev_bs))
            else:
                output[cid_tid] = bev_bs
    '''
    cid_tid_dict{
        'cam': ,
        'tid':,
        # 'frame_list': q_frame,
        'tracklet': key:frame,value:tracklet
    }
    '''
    # 得到一个[cam,tid,frame,bevx,bevy]

    

    
    sim_matrix = np.zeros((len(output),len(output)))
    model = TSMatchNet()
    model.cuda()
    model.eval()
    for i, key_i in enumerate(output.keys()):
        for j, key_j in enumerate(output.keys()):
            a = output[key_i].reshape(1,1,30,4).astype(np.float32)
            b = output[key_j].reshape(1,1,30,4).astype(np.float32)
            a = torch.from_numpy(a).cuda()
            b = torch.from_numpy(b).cuda()
            scores = model(a,b)
            scores = [0 if x[0] > x[1] else x[1] for x in scores]
            sim_matrix[i,j] = scores[0]
    return sim_matrix

def get_sim_matrix(_cfg,cid_tid_dict,cid_tids, use_st_filter=True):
    count = len(cid_tids)
    print('count: ', count)

    q_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    g_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    q_arr = normalize(q_arr, axis=1)
    g_arr = normalize(g_arr, axis=1)
    # sim_matrix = np.matmul(q_arr, g_arr.T)

    # st mask
    st_mask = np.ones((count, count), dtype=np.float32)
    st_mask = intracam_ignore(st_mask, cid_tids)
    #### dxg use_st_filter s ####
    # dxg 这是一个时空过滤，只能用于test场景的时空过滤！！
    if use_st_filter:
        st_mask = st_filter(st_mask, cid_tids, cid_tid_dict)
    #### dxg use_st_filter e ####
    # visual rerank
    visual_sim_matrix = visual_rerank(q_arr, g_arr, cid_tids, _cfg)
    visual_sim_matrix = visual_sim_matrix.astype('float32')
    print(visual_sim_matrix) # feature sim matrix

    # merge result
    np.set_printoptions(precision=3)
    sim_matrix = visual_sim_matrix * st_mask

    # sim_matrix[sim_matrix < 0] = 0
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix

def normalize(nparray, axis=0):
    nparray = preprocessing.normalize(nparray, norm='l2', axis=axis)
    return nparray

def get_match(cluster_labels):
    cluster_dict = dict()
    cluster = list()
    for i, l in enumerate(cluster_labels):
        if l in list(cluster_dict.keys()):
            cluster_dict[l].append(i)
        else:
            cluster_dict[l] = [i]
    for idx in cluster_dict:
        cluster.append(cluster_dict[idx])
    return cluster

def get_cid_tid(cluster_labels,cid_tids):
    cluster = list()
    for labels in cluster_labels:
        cid_tid_list = list()
        for label in labels:
            cid_tid_list.append(cid_tids[label])
        cluster.append(cid_tid_list)
    return cluster

def combin_cluster(sub_labels,cid_tids):
    cluster = list()
    for sub_c_to_c in sub_labels:
        if len(cluster)<1:
            cluster = sub_labels[sub_c_to_c]
            continue

        for c_ts in sub_labels[sub_c_to_c]:
            is_add = False
            for i_c, c_set in enumerate(cluster):
                if len(set(c_ts) & set(c_set))>0:
                    new_list = list(set(c_ts) | set(c_set))
                    cluster[i_c] = new_list
                    is_add = True
                    break
            if not is_add:
                cluster.append(c_ts)
    labels = list()
    num_tr = 0
    for c_ts in cluster:
        label_list = list()
        for c_t in c_ts:
            label_list.append(cid_tids.index(c_t))
            num_tr+=1
        label_list.sort()
        labels.append(label_list)
    print("new tricklets:{}".format(num_tr))
    return labels,cluster

def combin_feature(cid_tid_dict,sub_cluster):
    for sub_ct in sub_cluster:
        if len(sub_ct)<2: continue
        mean_feat = np.array([cid_tid_dict[i]['mean_feat'] for i in sub_ct])
        for i in sub_ct:
            cid_tid_dict[i]['mean_feat'] = mean_feat.mean(axis=0)
    return cid_tid_dict

def get_labels(_cfg, cid_tid_dict, cid_tids, score_thr):
    # 1st cluster
    sub_cid_tids = subcam_list(cid_tid_dict,cid_tids)
    sub_labels = dict()
    dis_thrs = [0.7,0.5,0.5,0.5,0.5,
                0.7,0.5,0.5,0.5,0.5]
    for i,sub_c_to_c in enumerate(sub_cid_tids):
        #### dxg modified get_sim_matrix ####
        sim_matrix = get_sim_matrix(_cfg,cid_tid_dict,sub_cid_tids[sub_c_to_c]) # dxg 这里不用改，因为只有41和46cam才会触发
        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-dis_thrs[i], affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
        labels = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(labels,sub_cid_tids[sub_c_to_c])
        sub_labels[sub_c_to_c] = cluster_cid_tids
    print("old tricklets:{}".format(len(cid_tids)))
    labels,sub_cluster = combin_cluster(sub_labels,cid_tids)

    # 2ed cluster # dxg 这里是计算相似性矩阵的地方
    cid_tid_dict_new = combin_feature(cid_tid_dict, sub_cluster) # 将上一步合并轨迹的特征进行整合，特征再求均值，1st没执行，所以这里也没合并
    sub_cid_tids = subcam_list2(cid_tid_dict_new,cid_tids) # combin_feature和subcam_list2是一对，前者合并feature，对于41-46后者把轨迹分成两部分，后者是把相邻相机的放在一个列表里，具体如下：.
    '''
    cid_tids:      [41,42,43,44,45,46]
    sub_cid_tids: [[  ,42,43,44,45,46]
                   [41,42,43,44,45,  ]]
    cid_tids: [1,2,3,4,5] 1-109,2-114,3-139,4-102,5-113 total=577
    sub_cid_tids: [1,2,3,4,5,6] 1--109,2--109+114=223,3--114+139=253, 4--139+102=241, 5--102+113=215,6--113
    '''
    sub_labels = dict()
    for i,sub_c_to_c in enumerate(sub_cid_tids):
        #### dxg modified get_sim_matrix ####
        sim_matrix = get_sim_matrix(_cfg,cid_tid_dict_new,sub_cid_tids[sub_c_to_c], use_st_filter=False) # dxg feature sim matrix
        sim_matrix_location = get_sim_matrix_TSMatch(_cfg,cid_tid_dict_new,sub_cid_tids[sub_c_to_c], use_st_filter=False)
        
        # appearance only
        # sim_matrix = sim_matrix

        # location only
        # if sim_matrix.shape == sim_matrix_location.shape:
        #     sim_matrix = sim_matrix_location
        
        # fusion
        if sim_matrix.shape == sim_matrix_location.shape:
            sim_matrix = sim_matrix*0.7 + sim_matrix_location*0.3
        
        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-dis_thrs[i], affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix) # 相似度越大，距离越小。
        labels = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(labels,sub_cid_tids[sub_c_to_c])
        sub_labels[sub_c_to_c] = cluster_cid_tids
    print("old tricklets:{}".format(len(cid_tids))) # 2ed虽然算了相似性矩阵，但是矩阵值全是0，在2ed没合并
    labels,sub_cluster = combin_cluster(sub_labels,cid_tids)
    # dxg 执行了两次第二次聚类
    cid_tid_dict_new = combin_feature(cid_tid_dict, sub_cluster)
    sub_cid_tids = subcam_list2(cid_tid_dict_new,cid_tids)
    sub_labels = dict()
    for i,sub_c_to_c in enumerate(sub_cid_tids):
        #### dxg modified get_sim_matrix ####
        sim_matrix = get_sim_matrix(_cfg,cid_tid_dict_new,sub_cid_tids[sub_c_to_c], use_st_filter=False) # dxg
        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-0.1, affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
        labels = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(labels,sub_cid_tids[sub_c_to_c])
        sub_labels[sub_c_to_c] = cluster_cid_tids
    print("old tricklets:{}".format(len(cid_tids)))
    labels,sub_cluster = combin_cluster(sub_labels,cid_tids)

    # # 3rd cluster
    # cid_tid_dict_new = combin_feature(cid_tid_dict,sub_cluster)
    # sim_matrix = get_sim_matrix(_cfg,cid_tid_dict_new, cid_tids)
    # cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1, affinity='precomputed',
    #                                          linkage='complete').fit_predict(1 - sim_matrix)
    # labels = get_match(cluster_labels)
    return labels

if __name__ == '__main__':
    cfg.merge_from_file(f'../../../config/{sys.argv[1]}')
    cfg.freeze()
    #### dxg s ####
    # scene_name = ['S06']
    # scene_cluster = [[41, 42, 43, 44, 45, 46]]
    scene_cluster=[]
    c = []
    for cam in os.listdir(cfg.DET_SOURCE_DIR):
        #### dxg ####
        # if cam == 'c005': # dxg no_c005
        #     pass
        c.append(int(cam[1:]))
    scene_cluster.append(c) # dxg
    mode, scenes = cfg.DET_SOURCE_DIR.split(os.sep)[-3:-1]
    # fea_dir = './exp/viz/test/S06/trajectory/'
    fea_dir = f'./exp/viz/{mode}/{scenes}/trajectory/' # dxg
    #### dxg e ####
    cid_tid_dict = dict()

    for pkl_path in os.listdir(fea_dir):
        cid = int(pkl_path.split('.')[0][-3:])
        with open(opj(fea_dir, pkl_path),'rb') as f:
            lines = pickle.load(f)
        for line in lines:
            tracklet = lines[line]
            tid = tracklet['tid']
            if (cid, tid) not in cid_tid_dict:
                cid_tid_dict[(cid, tid)] = tracklet


    cid_tids = sorted([key for key in cid_tid_dict.keys() if key[0] in scene_cluster[0]])
    clu = get_labels(cfg,cid_tid_dict,cid_tids,score_thr=cfg.SCORE_THR) # 这里就已经得到聚类后的结果了
    print('all_clu:', len(clu))
    new_clu = list()
    for c_list in clu:
        if len(c_list) <= 1: continue
        cam_list = [cid_tids[c][0] for c in c_list]
        if len(cam_list)!=len(set(cam_list)): continue
        new_clu.append([cid_tids[c] for c in c_list])
    print('new_clu: ', len(new_clu))

    all_clu = new_clu

    cid_tid_label = dict()
    for i, c_list in enumerate(all_clu):
        for c in c_list:
            cid_tid_label[c] = i + 1
    pickle.dump({'cluster': cid_tid_label}, open('test_cluster.pkl', 'wb'))
