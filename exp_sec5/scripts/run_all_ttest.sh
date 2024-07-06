FILES="../logs/experiments_difference_expectation/*"
STRING="summary"
for f in $FILES
do
  if ! [[ "$f" == *"$STRING"* ]];then
    python3.8 ttest.py $f ../logs/
  fi
  # take action on each file. $f store current file name
done