for alpha in 0.1 0.3 0.5 0.7 0.9; do
  for dataset in "moocen" "mooczh"; do
    cmd="python build_graph.py -dataset $dataset -alpha $alpha -no_course_dependency -no_user_data"
    echo $cmd
    $cmd
    for t in {1..1}; do
      cmd="python main.py -dataset $dataset -model GCN"
      echo $cmd
      $cmd
    done
    mv result/GCN_${dataset}_500.txt result/GCN_${dataset}_500_alpha${alpha}_nocs.txt
    cmd="python build_graph.py -dataset $dataset -alpha $alpha"
    echo $cmd
    $cmd
    for t in {1..1}; do
      cmd="python main.py -dataset $dataset -model GCN"
      echo $cmd
      $cmd
    done
    mv result/GCN_${dataset}_500.txt result/GCN_${dataset}_500_alpha${alpha}_cs.txt
  done
done
