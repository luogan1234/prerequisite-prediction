for alpha in 0.1 0.3 0.5 0.7 0.9; do
  cmd="python build_graph.py -dataset moocen -alpha $alpha"
  echo $cmd
  $cmd
  for seed in {0..3}; do
    cmd="python main.py -dataset moocen -model GCN -result_path alpha_en.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/alpha_en.txt result/GCN_moocen_alpha${alpha}.txt
done

cmd="python build_graph.py -dataset moocen"
echo $cmd
$cmd
for feature_dim in 4 8 16 24 32; do
  for seed in {0..3}; do
    cmd="python main.py -dataset moocen -model GCN -feature_dim $feature_dim -result_path feature_en.txt -seed $seed"
    echo $cmd
    $cmd
  done
  mv result/feature_en.txt result/GCN_moocen_dim${feature_dim}.txt
done
