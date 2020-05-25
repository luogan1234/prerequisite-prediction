for dataset in "moocen" "mooczh"; do
  for model in "LSTM" "TextCNN"; do
    for seed in {0..3}; do
      cmd="python main.py -dataset $dataset -model $model -result_path base.txt -seed $seed"
      echo $cmd
      $cmd
    done
    mv result/base.txt result/${model}_${dataset}.txt
  done
done

for dataset in "mooczh"; do
  for model in "LSTM_S"; do
    for seed in {0..3}; do
      cmd="python main.py -dataset $dataset -model $model -result_path base.txt -seed $seed"
      echo $cmd
      $cmd
    done
    mv result/base.txt result/${model}_${dataset}.txt
  done
done
