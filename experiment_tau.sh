for tau in 0.1 0.2 0.3 0.4 1.0; do
  for dataset in "moocen" "mooczh"; do
    cmd="python build_graph.py -dataset $dataset -tau $tau -no_course_dependency -no_user_data"
    echo $cmd
    $cmd
    for t in {1..1}; do
      cmd="python main.py -dataset $dataset -model GCN"
      echo $cmd
      $cmd
    done
    mv result/GCN_${dataset}_500.txt result/GCN_${dataset}_500_tau${tau}_nocs.txt
    cmd="python build_graph.py -dataset $dataset -tau $tau"
    echo $cmd
    $cmd
    for t in {1..1}; do
      cmd="python main.py -dataset $dataset -model GCN"
      echo $cmd
      $cmd
    done
    mv result/GCN_${dataset}_500.txt result/GCN_${dataset}_500_tau${tau}_cs.txt
  done
done
