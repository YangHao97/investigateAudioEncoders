# Investigating Pre-trained Audio Encoders in the Low-Resource Condition
*The code is for Interspeech 2023 paper: [Investigating Pre-trained Audio Encoders in the Low-Resource Condition](https://arxiv.org/pdf/2305.17733.pdf)*
## Environment and Dataset
Our work is developed based on [SUPERB benchmark](https://github.com/s3prl/s3prl). Please follow their instructions to set up environment, download dataset and preprocess data.
## Fine-tune models on downstream Tasks
Please select the model you want to fine-tune and the data proportion in ***upstream/wav2vec2_hug/expert.py*** and ***downstream/[task]/dataset.py & expert.py***  

### W2V2 & WavLM
```
python3 run_downstream.py -n ExpName -m train -u wav2vec2_hug_large_ll60k -d [task]
```
### Whisper
```
python3 run_downstream.py -n ExpName -m train -u wav2vec2_hug_large_ll60k -d [task] -s last_hidden_state
```
## Citation
```
@article{yang2023investigating,
  title={Investigating Pre-trained Audio Encoders in the Low-Resource Condition},
  author={Yang, Hao and Zhao, Jinming and Haffari, Gholamreza and Shareghi, Ehsan},
  journal={arXiv preprint arXiv:2305.17733},
  year={2023}
}
```
