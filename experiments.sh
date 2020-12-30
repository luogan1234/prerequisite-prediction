#gcn video only
for dataset in "moocen" "mooczh"; do
  cmd="python build_graph.py -dataset $dataset -no_course_dependency -no_user_act"
  echo $cmd & $cmd
  for model in "lstm1" "lstm2" "gcn"; do
    for seed in {0..49}; do
      cmd="python main.py -dataset $dataset -model $model -info main_v -seed $seed"
      echo $cmd & $cmd
    done
  done
done
# gcn mooczh cvs
for dataset in "mooczh"; do
  cmd="python build_graph.py -dataset $dataset -no_user_act"
  echo $cmd & $cmd
  for model in "gcn"; do
    for seed in {0..49}; do
      cmd="python main.py -dataset $dataset -model $model -info main_cv -seed $seed"
      echo $cmd & $cmd
    done
  done
  cmd="python build_graph.py -dataset $dataset -no_course_dependency"
  echo $cmd & $cmd
  for model in "gcn"; do
    for seed in {0..49}; do
      cmd="python main.py -dataset $dataset -model $model -info main_vs -seed $seed"
      echo $cmd & $cmd
    done
  done
  cmd="python build_graph.py -dataset $dataset -no_video_order"
  echo $cmd & $cmd
  for model in "gcn"; do
    for seed in {0..49}; do
      cmd="python main.py -dataset $dataset -model $model -info main_cs -seed $seed"
      echo $cmd & $cmd
    done
  done
  cmd="python build_graph.py -dataset $dataset -no_video_order -no_course_dependency"
  echo $cmd & $cmd
  for model in "gcn"; do
    for seed in {0..49}; do
      cmd="python main.py -dataset $dataset -model $model -info main_s -seed $seed"
      echo $cmd & $cmd
    done
  done
  cmd="python build_graph.py -dataset $dataset"
  echo $cmd & $cmd
  for model in "gcn"; do
    for seed in {0..49}; do
      cmd="python main.py -dataset $dataset -model $model -info main_cvs -seed $seed"
      echo $cmd & $cmd
    done
  done
done

# lstm+feature
cmd="python build_graph.py -dataset mooczh"
echo $cmd & $cmd
for model in "lstm1" "lstm2" "gcn"; do
  for seed in {0..49}; do
    cmd="python main.py -dataset mooczh -model $model -concat_feature -info concat_feature -seed $seed"
    echo $cmd & $cmd
  done
done
# gcn moocen alpha
for alpha in 0.1 0.3 0.5 0.7 0.9; do
  cmd="python build_graph.py -dataset moocen -alpha $alpha -no_course_dependency -no_user_act"
  echo $cmd & $cmd
  for seed in {0..49}; do
    cmd="python main.py -dataset moocen -model gcn -info v_alpha$alpha -seed $seed"
    echo $cmd & $cmd
  done
done
# gcn mooczh alpha
for alpha in 0.1 0.3 0.5 0.7 0.9; do
  cmd="python build_graph.py -dataset mooczh -alpha $alpha -no_video_order -no_course_dependency"
  echo $cmd & $cmd
  for seed in {0..49}; do
    cmd="python main.py -dataset mooczh -model gcn -info s_alpha$alpha -seed $seed"
    echo $cmd & $cmd
  done
  cmd="python build_graph.py -dataset mooczh -alpha $alpha"
  echo $cmd & $cmd
  for seed in {0..49}; do
    cmd="python main.py -dataset mooczh -model gcn -info cvs_alpha$alpha -seed $seed"
    echo $cmd & $cmd
  done
done
# gcn mooczh user_act_type
for user_act_type in "no_sequential" "no_cross_course" "no_backward" "no_skip"; do
  cmd="python build_graph.py -dataset mooczh -user_act_type $user_act_type"
  echo $cmd & $cmd
  for seed in {0..49}; do
    cmd="python main.py -dataset mooczh -model gcn -info user_act_$user_act_type -seed $seed"
    echo $cmd & $cmd
  done
done
# gcn mooczh user number
for user_num in 0 25 50 100 250 500; do
  for seed in {0..49}; do
    cmd="python build_graph.py -dataset mooczh -no_video_order -no_course_dependency -user_num $user_num -seed $seed"
    echo $cmd & $cmd
    cmd="python main.py -dataset mooczh -model gcn -info user_number_$user_num -seed $seed"
    echo $cmd & $cmd
  done
done
for user_prop in 0.1 0.2 0.4 0.7 1.0; do
  for seed in {0..49}; do
    cmd="python build_graph.py -dataset mooczh -no_video_order -no_course_dependency -user_prop $user_prop -seed $seed"
    echo $cmd & $cmd
    cmd="python main.py -dataset mooczh -model gcn -info user_prop_$user_prop -seed $seed"
    echo $cmd & $cmd
  done
done
