import logging
import numpy as np
import os
import scipy.io
from pathlib import Path
from box import Box

_logger = logging.getLogger(__name__)

def build_file_lists(params: Box):
    if not Path(params.file_lists.output_dir).exists():
        os.makedirs(params.file_lists.output_dir, exist_ok=True)

    query_glob = params.file_lists.get("input_query_glob", "*.jpg")
    query_imgnames_all = np.array(
        [[np.array(str(i)) for i in Path(params.file_lists.input_query_dir).glob(query_glob)]],
        dtype=np.object
    )
    _logger.info(f"Found {query_imgnames_all.shape[1]} query images.")

    db_glob = params.file_lists.get("input_db_glob", "**/cutout*.jpg")
    db_imgnames_all = np.array(
        [[np.array(str(i)) for i in Path(params.file_lists.input_db_dir).glob(db_glob)]],
        dtype=np.object
    )
    _logger.info(f"Found {db_imgnames_all.shape[1]} db images.")

    scipy.io.savemat(
        Path(params.file_lists.output_dir) / params.file_lists.get("output_query_mat_name", "query_imgnames_all.mat"),
        {"query_imgnames_all": query_imgnames_all}
    )

    scipy.io.savemat(
        Path(params.file_lists.output_dir) / params.file_lists.get("output_db_mat_name", "db_imgnames_all.mat"),
        {"db_imgnames_all": db_imgnames_all}
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("param_yaml_file", type=Path)
    args = parser.parse_args()

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    _logger.setLevel(logging.INFO)
    _logger.addHandler(handler)

    params = Box.from_yaml(filename=args.param_yaml_file)
    build_file_lists(params)
 