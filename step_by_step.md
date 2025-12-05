1. Build a KG:
```bash
python three_layer_import.py \
    --clear \
    --bottom ../data/layer3_umls \
    --middle ../data/layer2_pmc \
    --top ../data/layer1_mimic_ex \
    --grained_chunk \
    --ingraphmerge \
    --trinity
```

2. Model Inference:
```bash
python run.py --inference
``` 

python three_layer_import.py \
    --clear \
    --bottom ../data/layer3_umls \
    --middle ../data/layer2_pmc \
    --top ../data/layer1_mimic_ex \
    --grained_chunk \
    --ingraphmerge \
    --trinity