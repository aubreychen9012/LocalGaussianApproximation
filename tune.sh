python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--fprate 0.001 \
--weight 10.

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--fprate 0.001 \
--weight 5.

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--fprate 0.005 \
--weight 5.

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--fprate 0.05 \
--weight 5.

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--fprate 0.05 \
--weight 4.5

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--fprate 0.05 \
--weight 4.

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--fprate 0.05 \
--weight 3.5

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--fprate 0.05 \
--weight 3.0

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--fprate 0.05 \
--weight 3.0


python restore_tv.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--fprate 0.05 \
--weight 10.0

python restore_tv.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--fprate 0.05 \
--weight 5.5

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193000 \
--config conf.yaml \
--fprate 0.05 \
--weight 9.
% 0.3384666211900168, 1.2776165860408355, 2.09324139740868, 3.0437043722055117
$ .975014328286832

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193000 \
--config conf.yaml \
--fprate 0.05 \
--weight 7.
% 0.3528152558455371, 0.8052265335899105, 1.1296974664831265, 1.8209473389312456


python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193000 \
--config conf.yaml \
--fprate 0.05 \
--weight 8.
%0.2766825441221076, 0.5754595924262097, 0.775105259629806, 1.3386468234964335

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193000 \
--config conf.yaml \
--fprate 0.05 \
--weight 7.

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--fprate 0.05 \
--weight 8.5

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--fprate 0.05 \
--weight 6.5

python restore_tv.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--constraint TV \
--fprate 0.05 \
--weight 0.02

python restore_tv.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--constraint L1 \
--fprate 0.05 \
--weight 0.02

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--method estimate \
--constraint L1 \
--fprate 0.05 \
--weight 200

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--config conf.yaml \
--method estimate \
--constraint TV \
--fprate 0.05 \
--weight 100

python restore_camcan.py --model_name covVAE_reverse_retry \
--load_step 193000 \
--config conf.yaml \
--method neural \
--constraint TV_combined \
--fprate 0.05 \
--weight 15.

python restore_tv.py --model_name vae_camcan_anneal_z4096 \
--load_step 18000 \
--config conf.yaml \
--constraint L1 \
--fprate 0.05 \
--weight 1.15

python restore_tv.py --model_name vae_camcan_anneal_z4096 \
--load_step 18000 \
--config conf.yaml \
--constraint TV \
--fprate 0.05 \
--weight 1.15

python restore_tv.py --model_name vae_camcan_01data \
--load_step 99500 \
--config conf.yaml \
--constraint TV \
--fprate 0.05 \
--weight 3.

python restore_tv.py --model_name vaeres_t1 \
--load_step 99500 \
--config conf.yaml \
--constraint TV \
--fprate 0.05 \
--weight 0.


python run_cluster.py --model_name covVAE_reverse_retry \
--load_step 193000 \
--config conf.yaml \
--method neural \
--constraint TV_combined \
--fprate 0.05 \
--weight 15.