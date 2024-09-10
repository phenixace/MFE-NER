# MFE-NER
This is the official repo of paper: MFE-NER: Multi-feature Fusion Embedding for Chinese Named Entity Recognition (Accepted by CCL 2024)

## Usage
```
python main.py --dataset msra --fusion linear --glyph 1 --earlystop 1 > msra_glyph_bert.txt
python main.py --dataset msra --fusion linear --pron 1 --earlystop 1 > msra_pron_bert.txt
python main.py --dataset msra --fusion linear --glyph 1 --earlystop 1 --embedding static > msra_glyph.txt
python main.py --dataset msra --fusion linear --pron 1 --earlystop 1 --embedding static > msra_pron.txt
```
