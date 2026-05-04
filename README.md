# SIFRank
The code of our paper [SIFRank: A New Baseline for Unsupervised Keyphrase Extraction Based on Pre-trained Language Model](https://ieeexplore.ieee.org/document/8954611)

## Versions Notes
* 2020/02/21——``Initial version`` 
Provided the most basic functions.
* 2020/02/28——``Second version``
Added new algorithms ``DS``(document segmentation) and ``EA``(embeddings alignment) to ``speed up`` SIFRank and SIFRank+.
* 2020/03/02——``Third version``
A little change of SIFRank+ in ``./model/method.py`` about making a simple normalization of position_score.
## Environment
```
Python 3.10
torch 1.12.1
torchvision 0.13.1
numpy 1.23.5
scipy 1.10.1
scikit-learn 1.2.2
overrides 3.1.0
allennlp 2.10.1
elmoformanylangs 0.0.4.post2
ginza 5.1.3
ja-ginza 5.1.3
spacy >=3.3,<3.4
StanfordCoreNLP 3.9.1.1 (for the original English pipeline)
```

Install Python dependencies with:

```
python3.10 -m pip install --no-cache-dir -r requirements.txt
```

For CUDA 11.3 PyTorch on Ampere GPUs such as RTX 3090:

```
python3.10 -m pip install --no-cache-dir --force-reinstall -r requirements-gpu-cu113.txt
```
## Download
* ELMo ``elmo_2x4096_512_2048cnn_2xhighway_options.json`` and ``elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5`` from [here](https://allennlp.org/elmo) , and save it to the ``auxiliary_data/`` directory
* ELMoForManyLangs Japanese model directory containing ``config.json``, ``encoder.pkl``, ``token_embedder.pkl``, ``word.dic``, and ``char.dic`` can also be saved under ``auxiliary_data/ja.model/``. The ELMoForManyLangs wrapper runs on CPU by default; pass ``cuda_device=0`` only when your PyTorch CUDA build supports your GPU.
* StanfordCoreNLP ``stanford-corenlp-full-2018-02-27`` from [here](https://stanfordnlp.github.io/CoreNLP/), and save it to anywhere

``GinzaNLPAdapter`` uses GiNZA's built-in ``STOP_WORDS`` by default. Pass ``stopwords=[]`` to disable them or pass your own iterable to override them.
Pass ``split_mode="A"``, ``"B"``, or ``"C"`` to control Sudachi tokenization granularity.

Build Japanese Wikipedia frequency files for split modes A and B with:

```
python3.10 util/build_ja_wikipedia_vocab.py /path/to/wikiextractor/output --modes A B --output-dir auxiliary_data --output-prefix ja_wikipedia_vocab --use-ginza-stopwords
```

For large corpora, tokenize first with Sudachi's Java CLI and then count tokenized output:

```
java -jar sudachi.jar -m A -o wiki_A.sudachi wiki.txt
java -jar sudachi.jar -m B -o wiki_B.sudachi wiki.txt
python3.10 util/count_sudachi_tokens.py wiki_A.sudachi --output auxiliary_data/ja_wikipedia_vocab_A.txt
python3.10 util/count_sudachi_tokens.py wiki_B.sudachi --output auxiliary_data/ja_wikipedia_vocab_B.txt
```

## Usage
```
import nltk
from embeddings import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus
from stanfordcorenlp import StanfordCoreNLP
import time

#download from https://allennlp.org/elmo
options_file = "../auxiliary_data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "../auxiliary_data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

ELMO = word_emb_elmo.WordEmbeddings(options_file, weight_file, cuda_device=0)
SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
en_model = StanfordCoreNLP(r'E:\Python_Files\stanford-corenlp-full-2018-02-27',quiet=True)#download from https://stanfordnlp.github.io/CoreNLP/
elmo_layers_weight = [0.0, 1.0, 0.0]

text = "Discrete output feedback sliding mode control of second order systems - a moving switching line approach The sliding mode control systems (SMCS) for which the switching variable is designed independent of the initial conditions are known to be sensitive to parameter variations and extraneous disturbances during the reaching phase. For second order systems this drawback is eliminated by using the moving switching line technique where the switching line is initially designed to pass the initial conditions and is subsequently moved towards a predetermined switching line. In this paper, we make use of the above idea of moving switching line together with the reaching law approach to design a discrete output feedback sliding mode control. The main contributions of this work are such that we do not require to use system states as it makes use of only the output samples for designing the controller. and by using the moving switching line a low sensitivity system is obtained through shortening the reaching phase. Simulation results show that the fast output sampling feedback guarantees sliding motion similar to that obtained using state feedback"
keyphrases = SIFRank(text, SIF, en_model, N=15,elmo_layers_weight=elmo_layers_weight)
keyphrases_ = SIFRank_plus(text, SIF, en_model, N=15, elmo_layers_weight=elmo_layers_weight)
print(keyphrases)
print(keyphrases_)
```

Use ``extract_keyphrases`` to switch between SIFRank and SIFRank+:

```
from model.method import extract_keyphrases

keyphrases = extract_keyphrases(text, SIF, ja_model, rank_method="sifrank", N=15)
keyphrases_plus = extract_keyphrases(text, SIF, ja_model, rank_method="sifrank_plus", N=15, position_bias=3.4)
```
## Evaluate the model
Use this ``eval/sifrank_eval.py`` to evaluate SIFRank on ``Inspec``, ``SemEval2017`` and ``DUC2001 datasets``
We also have evaluation codes for other baseline models. We will organize and upload them later, so stay tuned.
**F1 score** when the number of keyphrases extracted N is set to 5.

| Models       | Inspec       | SemEval2017   | DUC2001      |
| :-----       | :----:       | :----:        |:----:        |
| TFIDF        | 11.28        | 12.70         |  9.21        |
| YAKE         | 15.73        | 11.84         | 10.61        |
| TextRank     | 24.39        | 16.43         | 13.94        |
| SingleRank   | 24.69        | 18.23         | 21.56        |
| TopicRank    | 22.76        | 17.10         | 20.37        |
| PositionRank | 25.19        | 18.23         | 24.95        |
| Multipartite | 23.05        | 17.39         | 21.86        |
| RVA          | 21.91        | 19.59         | 20.32        |
| EmbedRank d2v| 27.20        | 20.21         | 21.74        |
| SIFRank      | **29.11**        | **22.59**         | 24.27        |
| SIFRank+     | 28.49        | 21.53         | **30.88**        |

## Cite
If you use this code, please cite this paper
```
@article{DBLP:journals/access/SunQZWZ20,
  author    = {Yi Sun and
               Hangping Qiu and
               Yu Zheng and
               Zhongwei Wang and
               Chaoran Zhang},
  title     = {SIFRank: {A} New Baseline for Unsupervised Keyphrase Extraction Based
               on Pre-Trained Language Model},
  journal   = {{IEEE} Access},
  volume    = {8},
  pages     = {10896--10906},
  year      = {2020},
  url       = {https://doi.org/10.1109/ACCESS.2020.2965087},
  doi       = {10.1109/ACCESS.2020.2965087},
  timestamp = {Fri, 07 Feb 2020 12:04:22 +0100},
  biburl    = {https://dblp.org/rec/journals/access/SunQZWZ20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
