#!/bin/bash

MESH_FN=out/meshes/ycb/mug/textured.obj
DATA_DIR=/home/markvdm/RobotSetup/catkin_ws/src/ncf_real/out/poke_experiments/mug/v6
OUT_DIR=out/icp/mug/v6/

python scripts/poke_experiment/icp_gt.py $MESH_FN $DATA_DIR/mug_motioncam.ply $DATA_DIR/mug_photoneo.ply $OUT_DIR/test.pkl.gzip
