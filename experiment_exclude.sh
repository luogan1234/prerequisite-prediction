cmd="python build_graph.py -dataset mooczh -no_video_order"
echo $cmd
$cmd
for seed in {0..4}; do
  cmd="python main.py -dataset mooczh -model GCN -output exclude.txt -seed $seed"
  echo $cmd
  $cmd
done
mv result/exclude.txt result/GCN_mooczh_no_video_order.txt

cmd="python build_graph.py -dataset mooczh -no_course_dependency"
echo $cmd
$cmd
for seed in {0..4}; do
  cmd="python main.py -dataset mooczh -model GCN -output exclude.txt -seed $seed"
  echo $cmd
  $cmd
done
mv result/exclude.txt result/GCN_mooczh_no_course_dependency.txt

cmd="python build_graph.py -dataset mooczh -no_user_data"
echo $cmd
$cmd
for seed in {0..4}; do
  cmd="python main.py -dataset mooczh -model GCN -output exclude.txt -seed $seed"
  echo $cmd
  $cmd
done
mv result/exclude.txt result/GCN_mooczh_no_user_data.txt
