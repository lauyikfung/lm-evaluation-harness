# lm-evaluation-harness
First edit the test.sh:
 - **TARGET_MODEL_DEFINITION=./lm_eval/models/new_model.py** 
   - change it as /YOUR_DIR_TO_LM_EVALUATION_HARNESS_REPOSITORY/lm_eval/models/new_model.py for your directory to lm-evaluation-harness repo
 - **cat /p/scratch/westai0008/zhang57/nanoGPT-neo/model/$MODEL_DEFINITION.py > $TARGET_MODEL_DEFINITION**
   - change the prefix to your directory of nanoGPT-neo
Then run test.sh: bash test.sh MODEL_PATH MODEL_DEFINITION num_fs OUTPUT_PATH TASKS
 - MODEL_PATH: the directory of the saved checkpoint
 - MODEL_DEFINITION: the type of model: Llama_hf, llama-gqa_hf (both gqa and mqa use the same architecture), llama-mla_hf, GPT_cp2_v5_3_kvonly_hf
 - num_fs: number of few-shot, default 0
 - OUTPUT_PATH: output result json path, default: "results/"$MODEL_DEFINITION, you can change for different location
 - TASKS: tasks, default: arc_challenge,arc_easy,openbookqa,boolq,hellaswag,piqa,winogrande,mmlu,social_iqa,sciq