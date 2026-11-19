import glob
import json
import os


for filename in sorted(glob.glob("analysis/sparse_*.json")):
    with open(filename, encoding="utf-8") as stream:
        data = json.load(stream)
    megabytes = (data["keyAtlasBytes"] + data["flowAtlasBytes"]) / 1e6
    print(os.path.basename(filename),
          "keys", data["keyframeCount"],
          "MB", round(megabytes, 2),
          "MAE", data["sparseMeanMae"],
          "PSNR", data["sparseMeanPsnr"],
          "sharpness", data["sparseMeanSharpness"])
