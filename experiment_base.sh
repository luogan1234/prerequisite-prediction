for epochs in 500; do
  for dataset in "moocen" "mooczh"; do
    for model in "TextCNN" "LSTM"; do
      for t in {1..1}; do
        cmd="python main.py -dataset $dataset -model $model -epochs $epochs"
        echo $cmd
        $cmd
      done
    done
  done
done
