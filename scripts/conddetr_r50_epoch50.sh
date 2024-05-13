script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}

python main.py \
    --batch_size 1 \
    --coco_path ../data/voc2coco \
    --output_dir output/$script_name