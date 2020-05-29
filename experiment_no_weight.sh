cmd="python build_graph.py -dataset moocen -no_weight"
echo $cmd
$cmd
for seed in {0..3}; do
  cmd="python main.py -dataset moocen -model GCN -result_path no_weight.txt -seed $seed"
  echo $cmd
  $cmd
done
mv result/no_weight.txt result/GCN_moocen_no_weight.txt

cmd="python build_graph.py -dataset mooczh -no_course_dependency -no_user_act -no_weight"
echo $cmd
$cmd
for seed in {0..3}; do
  cmd="python main.py -dataset moocen -model GCN -result_path no_weight.txt -seed $seed"
  echo $cmd
  $cmd
done
mv result/no_weight.txt result/GCN_mooczh_nocs_no_weight.txt

cmd="python build_graph.py -dataset mooczh -no_video_order -no_course_dependency -no_weight"
echo $cmd
$cmd
for seed in {0..3}; do
  cmd="python main.py -dataset moocen -model GCN -result_path no_weight.txt -seed $seed"
  echo $cmd
  $cmd
done
mv result/no_weight.txt result/GCN_mooczh_novc_no_weight.txt

cmd="python build_graph.py -dataset mooczh -no_weight"
echo $cmd
$cmd
for seed in {0..3}; do
  cmd="python main.py -dataset moocen -model GCN -result_path no_weight.txt -seed $seed"
  echo $cmd 
  $cmd
done
mv result/no_weight.txt result/GCN_mooczh_no_weight.txt
