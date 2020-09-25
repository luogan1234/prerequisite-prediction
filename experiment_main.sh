for dataset in "moocen" "mooczh"; do
  cmd="python build_graph.py -dataset $dataset -no_course_dependency -no_user_act"
  echo $cmd & $cmd
  for model in "GCN"; do
    for seed in {0..49}; do
      cmd="python main.py -dataset $dataset -model $model -result_path main.txt -seed $seed"
      echo $cmd & $cmd
    done
    mv result/result/main.txt result/result/${model}_${dataset}_v.txt
  done
  
  for model in "LSTM"; do
    for seed in {0..49}; do
      cmd="python main.py -dataset $dataset -model $model -result_path main.txt -seed $seed"
      echo $cmd & $cmd
    done
	mv result/result/main.txt result/result/${model}_${dataset}.txt
  done
done

for dataset in "mooczh"; do
  cmd="python build_graph.py -dataset $dataset -no_user_act"
  echo $cmd & $cmd
  for model in "GCN"; do
    for seed in {0..49}; do
      cmd="python main.py -dataset $dataset -model $model -result_path main.txt -seed $seed"
      echo $cmd & $cmd
    done
    mv result/result/result/result/main.txt result/result/${model}_${dataset}_cv.txt
  done
  
  cmd="python build_graph.py -dataset $dataset -no_course_dependency"
  echo $cmd & $cmd
  for model in "GCN"; do
    for seed in {0..49}; do
      cmd="python main.py -dataset $dataset -model $model -result_path main.txt -seed $seed"
      echo $cmd & $cmd
    done
    mv result/result/result/result/main.txt result/result/${model}_${dataset}_vs.txt
  done
  
  cmd="python build_graph.py -dataset $dataset -no_video_order"
  echo $cmd & $cmd
  for model in "GCN"; do
    for seed in {0..49}; do
      cmd="python main.py -dataset $dataset -model $model -result_path main.txt -seed $seed"
      echo $cmd & $cmd
    done
    mv result/result/result/result/main.txt result/result/${model}_${dataset}_cs.txt
  done
  
  cmd="python build_graph.py -dataset $dataset -no_video_order -no_course_dependency"
  echo $cmd & $cmd
  for model in "GCN"; do
    for seed in {0..49}; do
      cmd="python main.py -dataset $dataset -model $model -result_path main.txt -seed $seed"
      echo $cmd & $cmd
    done
    mv result/result/main.txt result/result/${model}_${dataset}_s.txt
  done
  
  cmd="python build_graph.py -dataset $dataset"
  echo $cmd & $cmd
  for model in "GCN"; do
    for seed in {0..49}; do
      cmd="python main.py -dataset $dataset -model $model -result_path main.txt -seed $seed"
      echo $cmd & $cmd
    done
    mv result/result/main.txt result/result/${model}_${dataset}_cvs.txt
  done
done
