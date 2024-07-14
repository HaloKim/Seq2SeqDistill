torchrun --nproc_per_node=3 src/seq2seqdistill/main.py --model-type t5 --teacher paust/pko-t5-large --num-encoder-layers 12 --num-decoder-layers 12 --hidden-dim 768 --vocab-size 50358 --output-dir ./distilled_model --batch-size 40 --max_length 512 --epochs 1000 --learning_rate 3e-5 --warmup_steps 1000 --gradient-accumulation 2 --dataset "daekeun-ml/naver-news-summarization-ko" --dataset-input-column document --dataset-target-column summary

#--dataset-local-path "/workspace/address-bot/dataset/" --dataset-input-column ASR --dataset-target-column RAW --dataset-data-type "csv"

