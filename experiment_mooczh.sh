for alpha in 0.1 0.3 0.5 0.7 0.9; do
  cmd="python build_graph.py -dataset mooczh -alpha $alpha -no_course_dependency -no_user_act"
  echo $cmd
  $cmd
  for seed in {0..3}; do
    cmd="python main.py -dataset mooczh -model GCN -result_path alpha_zh.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/alpha_zh.txt result/GCN_mooczh_alpha${alpha}_nocs.txt
  cmd="python build_graph.py -dataset mooczh -alpha $alpha"
  echo $cmd
  $cmd
  for seed in {0..3}; do
    cmd="python main.py -dataset mooczh -model GCN -result_path alpha_zh.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/alpha_zh.txt result/GCN_mooczh_alpha${alpha}_cs.txt
done

cmd="python build_graph.py -dataset mooczh -no_course_dependency -no_user_act"
echo $cmd
$cmd
for feature_dim in 4 8 16 24 32; do
  for seed in {0..3}; do
    cmd="python main.py -dataset mooczh -model GCN -feature_dim $feature_dim -result_path feature_zh.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/feature_zh.txt result/GCN_mooczh_dim${feature_dim}_nocs.txt
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
  mv result/feature_zh.txt result/GCN_mooczh_dim${feature_dim}_cs.txt
done

for exclude in "no_video_order" "no_course_dependency" "no_user_act"; do
  cmd="python build_graph.py -dataset mooczh -$exclude"
  echo $cmd
  $cmd
  for seed in {0..3}; do
    cmd="python main.py -dataset mooczh -model GCN -result_path exclude.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/exclude.txt result/GCN_mooczh_${exclude}.txt
done
