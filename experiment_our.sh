for dataset in "moocen" "mooczh"; do
  for model in "GCN"; do
    cmd="python build_graph.py -dataset $dataset -no_course_dependency -no_user_data"
    echo $cmd
    $cmd
    for t in {1..1}; do
      cmd="python main.py -dataset $dataset -model $model"
      echo $cmd
      $cmd
    done
    mv result/GCN_${dataset}_500.txt result/GCN_${dataset}_500_nocs.txt
    cmd="python build_graph.py -dataset $dataset"
    echo $cmd
    $cmd
    for t in {1..1}; do
      cmd="python main.py -dataset $dataset -model $model"
      echo $cmd
      $cmd
    done
    mv result/GCN_${dataset}_500.txt result/GCN_${dataset}_500_cs.txt
  done
done
