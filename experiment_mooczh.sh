cmd="python build_graph.py -dataset mooczh -no_video_order"
echo $cmd
$cmd
for t in {1..4}; do
  cmd="python main.py -dataset mooczh -model GCN -output exclude.txt"
  echo $cmd
  $cmd
done
mv result/exclude.txt result/GCN_mooczh_no_video_order.txt

cmd="python build_graph.py -dataset mooczh -no_course_dependency"
echo $cmd
$cmd
for t in {1..4}; do
  cmd="python main.py -dataset mooczh -model GCN -output exclude.txt"
  echo $cmd
  $cmd
done
mv result/exclude.txt result/GCN_mooczh_no_course_dependency.txt

cmd="python build_graph.py -dataset mooczh -no_user_data"
echo $cmd
$cmd
for t in {1..4}; do
  cmd="python main.py -dataset mooczh -model GCN -output exclude.txt"
  echo $cmd
  $cmd
done
mv result/exclude.txt result/GCN_mooczh_no_user_data.txt

for alpha in 0.1 0.3 0.5 0.7 0.9; do
  cmd="python build_graph.py -dataset mooczh -alpha $alpha -no_course_dependency -no_user_data"
  echo $cmd
  $cmd
  for t in {1..4}; do
    cmd="python main.py -dataset mooczh -model GCN -output alpha_zh.txt"
    echo $cmd
    $cmd
  done
  mv result/alpha_zh.txt result/GCN_mooczh_alpha${alpha}_nocs.txt
  cmd="python build_graph.py -dataset mooczh -alpha $alpha"
  echo $cmd
  $cmd
  for t in {1..4}; do
    cmd="python main.py -dataset mooczh -model GCN -output alpha_zh.txt"
    echo $cmd
    $cmd
  done
  mv result/alpha_zh.txt result/GCN_mooczh_alpha${alpha}_cs.txt
done

cmd="python build_graph.py -dataset mooczh -no_course_dependency -no_user_data"
echo $cmd
$cmd
for feature_dim in 6 12 18 24 30; do
  for t in {1..4}; do
    cmd="python main.py -dataset mooczh -model GCN -feature_dim $feature_dim -output feature_zh.txt"
    echo $cmd
    $cmd
  done
  mv result/feature_zh.txt result/GCN_mooczh_dim${feature_dim}_nocs.txt
done

cmd="python build_graph.py -dataset mooczh"
echo $cmd
$cmd
for feature_dim in 6 12 18 24 30; do
  for t in {1..4}; do
    cmd="python main.py -dataset mooczh -model GCN -feature_dim $feature_dim -output feature_zh.txt"
    echo $cmd
    $cmd
  done
  mv result/feature_zh.txt result/GCN_mooczh_dim${feature_dim}_cs.txt
done
