

for dataset in news spam german
do
    for delta in 0.05 0.1 0.2 0.3 
    do
        for samples in 1000 10000
        do
            python3.8 sampling.py $dataset ../datasets/$dataset/ ../models/ ../logs/ --delta $delta --samples $samples --runs 10
        done
    done
done

