# R2Gen

This is the implementation of [Generating Radiology Reports via Memory-driven Transformer]() at EMNLP-2020.

## Citations

If you use or extend our work, please cite our paper at EMNLP-2020.
```
@inproceedings{chen-emnlp-2020-r2g,
    title = "Generating Radiology Reports via Memory-driven Transformer",
    author = "Chen, Zhihong and
      Song, Yan  and
      Chang, Tsung-Hui and
      Wan, Xiang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2020",
}
```

## Requirements

- `torch==1.5.0`
- `torchvision==0.6.0`
- `opencv-python==4.4.0.42`


## Run on IU X-Ray

Run `bash run_iu_xray.sh` to train a model on the IU X-Ray data.

## Run on MIMIC CXR

Run `bash run_mimic_cxr.sh` to train a model on the MIMIC CXR data.
