# Weights


## Location of weights

* Development larflow Y->U network. (~70% accuracy for <10 pixel matches.) [checkpoint_gpu=0]

      /cluster/tufts/wongjiradlab/twongj01/larflow/weights/dev/dev_larflow_y2u_832x512_32inplanes.tar

* Development larflow Y->V network. (~80% accuracy for <10 pixel matches) [checkpoint_gpu=3,4,5]

      /cluster/tufts/wongjiradlab/twongj01/larflow/weights/dev/dev_larflow_y2v_832x512_32inplanes.tar

* Development larflow Y->U,Y->V simultaneous prediction

      /cluster/tufts/wongjiradlab/twongj01/larflow/weights/

* Development infill network: Y,U,V(?)

      /cluster/tufts/wongjiradlab/twongj01/larflow/weights/dev/dev_infill_final_checkpoint_yplane.tar
      /cluster/tufts/wongjiradlab/twongj01/larflow/weights/dev/dev_infill_final_checkpoint_uplane.tar
      /cluster/tufts/wongjiradlab/twongj01/larflow/weights/dev/dev_infill_final_checkpoint_vplane.tar

* Development endpoint network:

      /cluster/tufts/wongjiradlab/twongj01/larflow/weights/dev/dev_endpoint_checkpoint.52500th_y.tar
      /cluster/tufts/wongjiradlab/twongj01/larflow/weights/dev/dev_endpoint_model_best_u.tar
      /cluster/tufts/wongjiradlab/twongj01/larflow/weights/dev/dev_endpoint_model_best_v.tar


## Scripts

* get all the dev weights [5.9 GB]: `get_all_dev_weights.sh`
