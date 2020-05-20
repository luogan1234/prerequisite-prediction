for user_num in 0 25 50 100 250 500; do
  for seed in {0..3}; do
    cmd="python build_graph.py -dataset mooczh -no_video_order -no_course_dependency -user_num $user_num -seed $seed"
    echo $cmd
    $cmd
    cmd="python main.py -dataset mooczh -model GCN -result_path user.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/user.txt result/GCN_mooczh_user_num_${user_num}.txt
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
  mv result/user.txt result/GCN_mooczh_user_prop_${user_prop}.txt
done

for user_act_type in "none" "all" "sequential_only" "cross_course_only" "backward_only" "no_sequential" "no_cross_course" "no_backward" "no_skip"; do
  cmd="python build_graph.py -dataset mooczh -no_video_order -no_course_dependency -user_act_type $user_act_type"
  echo $cmd
  $cmd
  for seed in {0..3}; do
    cmd="python main.py -dataset mooczh -model GCN -result_path user.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/user.txt result/GCN_mooczh_user_act_type_${user_act_type}.txt
done