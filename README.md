# infilling-adversarial

Generating adversarial textual examples with infilling language models

## Installation

1. Clone the repository recursively to add the ILM submodule:
``git clone --recursive``
2. Install dependencies: ``pip install -r requirements.txt``
3. Install nltk punkt (required by ILM): ``python -c "import nltk; nltk.download('punkt')"``
4. Install ilm as a python module: ``pip install -e ./ilm``
5. Set the ILM_DIR environment variable (we have set it to the working directory): ``export ILM_DIR="$PWD"``.
6. Download the pretrained ACL 2020 ILM evaluation model: ``python ./ilm/acl20_repro.py model ${model} ilm | bash``, where ``${model}`` can be ``sto`` for the stories-trained model, ``abs`` for the arXiv-trained model or ``lyr`` for the song lyric trained model. The model should be downloaded to ``models/${model}_ilm`` under the ILM_DIR directory.
7. (Optional) to use the ILM recipe with TextAttack itself, the MODEL_DIR environment variable should be set to the model path mentioned in the last step.

Running ``source setup.sh`` in the main repo will perform steps 2-7 automatically for you.

## Usage

``make_examples.sh`` makes training examples for a HuggingFace dataset given its ID.

``train.sh`` finetunes an ILM model on examples created by ``make_examples.sh`` and stores them in ``training/train``.

``transform_sampler.py`` will run the ILM transformation on a string, given some of its indices. Current options: 
- ``--train`` will use a pretrained model stored under ``training/train``.

``attack_sampler.py`` attacks a victim model using a random shuffle of a given dataset. Current options: 
- ``--train`` will use a pretrained model stored under ``training/train``.

``attack.sh`` will run the attack sampler with some reasonable settings. In order, its four arguments are:
1. The HuggingFace dataset ID
2. The name of the experiment; results will be written to ``results/${NAME}``.
3. The number of examples to evaluate on.
4. The index of the filtered dataset to begin from.

For example, to replicate the CLARE experiments on BERT-base, you can run the following:
```
source attack.sh yelp_polarity clare_yelp 1000 0
```

``eval_languagetool.py`` and ``eval_use.py`` compute average change in LanguageTool error frequency and USE similarity metrics on a ``_slim.txt`` file given the experiment name. Current options:
- ``--num_examples`` will add a progress bar to the given number of examples when specified.
- The scripts will write metrics to correspondingly named files in the experiment results folder.