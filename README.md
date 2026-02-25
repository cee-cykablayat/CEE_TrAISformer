# TrAISformer

Pytorch implementation of TrAISformer---A generative transformer for AIS trajectory prediction (https://arxiv.org/abs/2109.03958).

The transformer part is adapted from: https://github.com/karpathy/minGPT

---
<p align="center">
  <img width="600" height="450" src="./figures/t18_3.png">
</p>


#### Requirements: 
See requirements.yml

### Datasets:

The data used in this paper are provided by the [Danish Maritime Authority (DMA)](https://dma.dk/safety-at-sea/navigational-information/ais-data). 
Please refer to [the paper](https://arxiv.org/abs/2109.03958) for the details of the pre-processing step. The code is available here: https://github.com/CIA-Oceanix/GeoTrackNet/blob/master/data/csv2pkl.py

A processed dataset can be found in `./data/ct_dma/`
(the format is `[lat, log, sog, cog, unix_timestamp, mmsi]`).

### Run

Run `trAISformer.py` to train and evaluate the model.
(Please note that the values given by the code are in km, while the values presented in the paper were converted to nautical mile.)

### Replication Workspace (`CEE-Replication/`)

This repository now includes a full replication workspace under:

- `CEE-Replication/Notebook/03_embedding_pipeline.ipynb`
- `CEE-Replication/Notebook/04_TRAISformer_01.ipynb`
- `CEE-Replication/Notebook/traisformer_mlops.py`

The updated `04_TRAISformer_01.ipynb` includes:

- modular preprocessing/export flow
- TrAISformer-compatible dataset creation (`*.pkl`)
- MLflow experiment tracking (loss/accuracy/curves/artifacts)
- early stopping and crash-safe resume checkpoints

### Custom Region GeoJSON

The custom Region-1 polygon used in the replication pipeline is stored at:

- `CEE-Replication/Notebook/region_1.geojson`

GeoJSON content:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [-98.73875969190912, 31.100306355446108],
            [-98.73875969190912, 17.444407942371157],
            [-80.20286516849585, 17.444407942371157],
            [-80.20286516849585, 31.100306355446108],
            [-98.73875969190912, 31.100306355446108]
          ]
        ]
      }
    }
  ]
}
```


### License

See `LICENSE`

### Contact
For any questions, please open an issue and assign it to @dnguyengithub.
