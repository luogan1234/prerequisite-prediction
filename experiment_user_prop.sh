for user_prop in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  for k in {1..4}; do
    cmd="python build_graph.py -dataset mooczh -user_prop $user_prop"
    echo $cmd
    $cmd
    for seed in {0..4}; do
      cmd="python main.py -dataset mooczh -model GCN -output user_prop.txt -seed $seed"
      echo $cmd
      $cmd
    done
  done
  mv result/user_prop.txt result/GCN_mooczh_prop${user_prop}.txt
done
