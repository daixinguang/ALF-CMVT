# ALF-CMVT

## dataset
和AIC22一样，生成的检测结果保存在dataset/train/detect/c001/c001_dets.pkl
reid结果保存在dataset/train/detect_reid*/c001/c001_dets_feat.pkl
merge后的结果保存在dataset/train/detect/c001/c001_dets_feat.pkl
单摄像头跟踪结果保存在dataset/train/detect/c001/c001_mot_feat.pkl c001_not.txt

跨摄像头轨迹融合部分
trajectory_fusion 结果保存在
  单摄像头断裂轨迹拼接dataset/train/detect/c001/c001_mot_feat_break.pkl 这里保存了单文件的轨迹id和对应的bbox
  和c001_mot_feat./exp/viz/test/S01/trajectory/ 可视化是读取该文件实现的，subcluster也是以该文件为输入
sub_cluster 结果保存在 test_cluster.pkl 这个文件对每个摄像头的id进行了聚类
  在这里添加TSMatch模块，得到结果
       替换TSMatch模块，得到结果

gen_res 读取c001_mot_feat_break.pkl 获取bbox，读取 test_cluster.pkl获取合并后的轨迹在不同摄像头对应的ID track_S01.txt
interpolation 可执行可不执行，AIC22 在S06通过插值涨了一点分数，但是S01却降了


## TSMatch
```
2023-06-03 15:30:55.166423
  f1/p/r: 0.68/0.68/0.68
  ConfMat:  [[203, 82], [67, 123]]
2023.33-06-03 15:30:57.590720
```
单独测TSMatch F1:0.68