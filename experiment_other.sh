# TextCNN
for dataset in "moocen" "mooczh"; do
  for model in "TextCNN"; do
    for seed in {0..3}; do
      cmd="python main.py -dataset $dataset -model $model -result_path base.txt -seed $seed"
      echo $cmd
      $cmd
    done
    mv result/base.txt result/${model}_${dataset}.txt
  done
done
# LSTM_S
for dataset in "mooczh"; do
  for model in "LSTM_S"; do
    for seed in {0..3}; do
      cmd="python main.py -dataset $dataset -model $model -result_path base.txt -seed $seed"
      echo $cmd
      $cmd
    done
    mv result/base.txt result/${model}_${dataset}.txt
  done
done
# GCN moocen alpha
for alpha in 0.1 0.3 0.5 0.7 0.9; do
  cmd="python build_graph.py -dataset moocen -alpha $alpha -no_course_dependency -no_user_act"
  echo $cmd
  $cmd
  for seed in {0..3}; do
    cmd="python main.py -dataset moocen -model GCN -result_path alpha_en.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/alpha_en.txt result/GCN_moocen_alpha${alpha}_v.txt
done
# GCN moocen feature
cmd="python build_graph.py -dataset moocen -no_course_dependency -no_user_act"
echo $cmd
$cmd
for feature_dim in 4 8 16 24 32; do
  for seed in {0..3}; do
    cmd="python main.py -dataset moocen -model GCN -feature_dim $feature_dim -result_path feature_en.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/feature_en.txt result/GCN_moocen_dim${feature_dim}_v.txt
done
# GCN mooczh alpha
for alpha in 0.1 0.3 0.5 0.7 0.9; do
  cmd="python build_graph.py -dataset mooczh -alpha $alpha -no_user_act"
  echo $cmd
  $cmd
  for seed in {0..3}; do
    cmd="python main.py -dataset mooczh -model GCN -result_path alpha_zh.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/alpha_zh.txt result/GCN_mooczh_alpha${alpha}_cv.txt
  cmd="python build_graph.py -dataset mooczh -alpha $alpha -no_video_order -no_course_dependency"
  echo $cmd
  $cmd
  for seed in {0..3}; do
    cmd="python main.py -dataset mooczh -model GCN -result_path alpha_zh.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/alpha_zh.txt result/GCN_mooczh_alpha${alpha}_s.txt
  cmd="python build_graph.py -dataset mooczh -alpha $alpha"
  echo $cmd
  $cmd
  for seed in {0..3}; do
    cmd="python main.py -dataset mooczh -model GCN -result_path alpha_zh.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/alpha_zh.txt result/GCN_mooczh_alpha${alpha}_cvs.txt
done
# GCN mooczh feature
cmd="python build_graph.py -dataset mooczh -no_user_act"
echo $cmd
$cmd
for feature_dim in 4 8 16 24 32; do
  for seed in {0..3}; do
    cmd="python main.py -dataset mooczh -model GCN -feature_dim $feature_dim -result_path feature_zh.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/feature_zh.txt result/GCN_mooczh_dim${feature_dim}_cv.txt
done
cmd="python build_graph.py -dataset mooczh -no_video_order -no_course_dependency"
echo $cmd
$cmd
for feature_dim in 4 8 16 24 32; do
  for seed in {0..3}; do
    cmd="python main.py -dataset mooczh -model GCN -feature_dim $feature_dim -result_path feature_zh.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/feature_zh.txt result/GCN_mooczh_dim${feature_dim}_s.txt
done
cmd="python build_graph.py -dataset mooczh"
echo $cmd
$cmd
for feature_dim in 4 8 16 24 32; do
  for seed in {0..3}; do
    cmd="python main.py -dataset mooczh -model GCN -feature_dim $feature_dim -result_path feature_zh.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/feature_zh.txt result/GCN_mooczh_dim${feature_dim}_cvs.txt
done
# gcn mooczh user_act_type
for user_act_type in "none" "sequential_only" "cross_course_only" "backward_only" "no_sequential" "no_cross_course" "no_backward" "no_skip"; do
  cmd="python build_graph.py -dataset mooczh -no_video_order -no_course_dependency -user_act_type $user_act_type"
  echo $cmd
  $cmd
  for seed in {0..3}; do
    cmd="python main.py -dataset mooczh -model GCN -result_path user.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/user.txt result/GCN_mooczh_user_act_type_${user_act_type}_s.txt
done
# gcn mooczh user_number
for user_num in 0 25 50 100 250 500; do
  for seed in {0..3}; do
    cmd="python build_graph.py -dataset mooczh -no_video_order -no_course_dependency -user_num $user_num -seed $seed"
    echo $cmd
    $cmd
    cmd="python main.py -dataset mooczh -model GCN -result_path user.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/user.txt result/GCN_mooczh_user_num${user_num}_s.txt
done
for user_prop in 0.1 0.2 0.4 0.7 1.0; do
  for seed in {0..3}; do
    cmd="python build_graph.py -dataset mooczh -no_video_order -no_course_dependency -user_prop $user_prop -seed $seed"
    echo $cmd
    $cmd
    cmd="python main.py -dataset mooczh -model GCN -result_path user.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/user.txt result/GCN_mooczh_user_prop${user_prop}_s.txt
done
# gcn no weight
cmd="python build_graph.py -dataset moocen -no_course_dependency -no_user_act -no_weight"
echo $cmd
$cmd
for seed in {0..3}; do
  cmd="python main.py -dataset moocen -model GCN -result_path no_weight.txt -seed $seed"
  echo $cmd
  $cmd
done
mv result/no_weight.txt result/GCN_moocen_v_nw.txt
cmd="python build_graph.py -dataset mooczh -no_user_act -no_weight"
echo $cmd
$cmd
for seed in {0..3}; do
  cmd="python main.py -dataset moocen -model GCN -result_path no_weight.txt -seed $seed"
  echo $cmd
  $cmd
done
mv result/no_weight.txt result/GCN_mooczh_cv_nw.txt
cmd="python build_graph.py -dataset mooczh -no_video_order -no_course_dependency -no_weight"
echo $cmd
$cmd
for seed in {0..3}; do
  cmd="python main.py -dataset moocen -model GCN -result_path no_weight.txt -seed $seed"
  echo $cmd
  $cmd
done
mv result/no_weight.txt result/GCN_mooczh_s_nw.txt
cmd="python build_graph.py -dataset mooczh -no_weight"
echo $cmd
$cmd
for seed in {0..3}; do
  cmd="python main.py -dataset moocen -model GCN -result_path no_weight.txt -seed $seed"
  echo $cmd
  $cmd
done
mv result/no_weight.txt result/GCN_mooczh_cvs_nw.txt