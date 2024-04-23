script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}

python main.py \
    --batch_size 2 \
    --coco_path ../data/coco \
    --output_dir output/$script_name