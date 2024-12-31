MODEL_PATH=$1
MODEL_DEFINITION=$2
TARGET_MODEL_DEFINITION=/p/scratch/westai0008/zhang57/lm-evaluation-harness/lm_eval/models/new_model.py
num_fs=${3:-"0"}
TASKS=${4:-"arc_challenge,arc_easy,openbookqa,boolq,hellaswag,piqa,winogrande,mmlu,social_iqa,sciq"}
# arc_challenge,arc_easy,openbookqa,boolq,hellaswag,piqa,winogrande
# mmlu,social_iqa,sciq
# bad: bbh,hendrycks_math
OUTPUT_PATH=${5:-"results/"$MODEL_DEFINITION}
# Copy model file content to target path
# /p/scratch/westai0008/zhang57/nanoGPT-neo/model/GPT_cp2_v3_4_hf.py
cat /p/scratch/westai0008/zhang57/nanoGPT-neo/model/$MODEL_DEFINITION.py > $TARGET_MODEL_DEFINITION

lm_eval --model hf --model_args pretrained=$MODEL_PATH --trust_remote_code --tasks $TASKS --batch_size 16 --output_path $OUTPUT_PATH --num_fewshot $num_fs