set -e -x

# wiki, wsj, conll_eng, conll_deu
prefix=$1 

# small, large
size=$2

# >= 0 for GPU, -1 for CPU
gpu=$3

if [ "$#" -ne 3 ] ; then
  echo "Usage: $0 <corpus> [small|large] <gpuid>" >&2
  exit 1
fi

data=data/$prefix
testdata=$data/test.lower.txt
model=lstm
rnn=300
layer=2
if [ $size = "large" ]; then
  rnn=700
  layer=3
fi

cv_dir=cv/"$prefix"_"$model"_"$rnn"hidden_"$layer"layer
model=`ls $cv_dir/*.pt | python best_model.py`
beam=10
samplescript=truecase.py

echo "Truecasing using $model $size"
time python $samplescript $model -beamsize $beam -verbose 0 -gpuid $gpu < $testdata > $cv_dir/output.txt
