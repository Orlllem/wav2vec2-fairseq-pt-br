---
language: pt
datasets:
- common_voice
metrics:
- wer
- cer
tags:
- audio
- automatic-speech-recognition
- speech
- xlsr-fine-tuning-week
license: apache-2.0
model-index:
- name: XLSR Wav2Vec2 Portuguese by Orlem Santos
  results:
  - task: 
      name: Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Common Voice pt
      type: common_voice
      args: pt
    metrics:
       - name: Test WER
         type: wer
         value: 10.74
       - name: Test CER
         type: cer
         value: 3.43
---

# wav2vec2-fairseq-pt-br


## Training a new model with the CLI tools

Inside the `wav2vec2-fairseq-pt-br` folder

```
git clone https://github.com/pytorch/fairseq
cd fairseq
git checkout f3b6f5817fbee59057ae2506f01502ea3c301b4b
cd ..
```

Given a directory containing wav files to be used for finetuning the multi-lingual xlsr-53 in portuguese
### Prepare training data manifest:

```
$ python fairseq/examples/wav2vec/wav2vec_manifest.py data/waves --dest manifest --ext wav
```

### Fine-tune a pre-trained model with CTC:

Fine-tuning a model requires parallel audio and labels file, as well as a vocabulary file in fairseq format.

The [script](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/libri_labels.py) that generates labels for the common-voice-pt dataset from the tsv file produced by wav2vec_manifest.py

```shell script
$ python fairseq/examples/wav2vec/libri_labels.py manifest/train.tsv --output-dir manifest --output-name train
```

```shell script
$ python fairseq/examples/wav2vec/libri_labels.py manifest/valid.tsv --output-dir manifest --output-name valid
```


Fine-tuning on common-voice-pt with letter targets:
```shell script
$ fairseq-hydra-train \
    task.data=manifest \
    model.w2v_path=models/xlsr_53_56k.pt \
    --config-dir . \
    --config-name xlrs53_pt
```

### Evaluating and Inference of the CTC model:

Evaluating the CTC model with a language model requires flashlight python bindings (previously called wav2letter to be installed. 

Next, run the evaluation command

```shell script
$ python fairseq/examples/speech_recognition/infer.py data/waves_test/pt/br --task audio_pretraining \
--nbest 1 --path models/checkpoint_best.pt --gen-subset test --results-path reports/ --w2l-decoder viterbi \
--lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 --beam 500 \
--post-process letter 
```

This script will generate the hypos and refs `.txt` files on the `reports` folder. These predictions can be evaluated to create the `metrics.json` in the `reports` folder.   

```shell script
$ python scripts/evaluate_hypos.py
```

This implementation achieved the state-of-the-art for the dataset [Speech Recognition on Common Voice Portuguese](https://paperswithcode.com/sota/speech-recognition-on-common-voice-portuguese).

```shell script
{"WER": "10.741223841660522", "CER": "3.4280799475753603"}
```