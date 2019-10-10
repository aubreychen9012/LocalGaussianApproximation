## VAE-MAP TV

python restore_tv_test.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--test_files brats_test_list.p \
--fprate 0.001 \
--weight 0.022 \
--restore_constraint TV \
--preset_threshold 1.8448490835243772 1.2231466735094387 0.980452835301016 0.4516081251681586 \
--config conf.yaml

## VAE-MAP L1
python restore_tv_test.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--test_files brats_test_list.p \
--fprate 0.001 \
--weight 0.022 \
--restore_constraint L1 \
--preset_threshold 2.445389656837459 1.7549987883446698 1.1787369588217206 0.4260177767425019 \
--config conf.yaml


## VAE-estimate, L1

python restore.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--test_files brats_test_list.p \
--fprate 0.001 \
--weight 500 \
--restore_method estimate \
--restore_constraint L1 \
--preset_threshold 1.9988858272668115 1.5590743938302656 1.300833122771115 0.587965447538305 \
--config conf.yaml

## VAE-neural, L1
python restore.py --model_name covVAE_reverse_retry \
--load_step 193000 \
--test_files brats_test_list.p \
--fprate 0.001 \
--weight 8. \
--restore_method neural \
--restore_constraint L1 \
--preset_threshold 0.2766825441221076 0.5754595924262097 0.775105259629806 1.3386468234964335 \
--config conf.yaml

## VAE-neural, L1-combined
python restore.py --model_name covVAE_reverse_retry \
--load_step 193000 \
--test_files brats_test_list.p \
--fprate 0.001 \
--weight 8. \
--restore_method neural \
--restore_constraint L1_combined \
--preset_threshold 1.3386468234964335 0.775105259629806 0.5754595924262097 0.2766825441221076 \
--config conf.yaml

## VAE-neural, TV

python restore.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--test_files ../brats_test.txt \
--fprate 0.001 \
--weight 17.0 \
--restore_method neural \
--restore_constraint TV \
--preset_threshold 1.6147044312655163 0.7766226131131138 0.5516066311673229 0.28456133727271216 \
--config conf.yaml


## VAE-neural, TV-combined

python restore.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--test_files brats_test_list.p \
--fprate 0.001 \
--weight 15 \
--restore_method neural \
--restore_constraint TV_combined \
--preset_threshold 2.0573840631930786 0.7730446958345301 0.523397830713065 0.27063946836172414 \
--config conf.yaml

# old thresholds: 2.5254229568983897 1.8012439932553819 1.003178889604678 0.6078989496812324


python restore.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--test_files brats_test_list.p \
--fprate 0.001 \
--weight 200 \
--restore_method estimate \
--restore_constraint TV \
--preset_threshold 2.3856368372966728 1.9838725338953274 1.7626150766270172 1.0192098657719435 \
--config conf.yaml



python restore_tv_test.py --model_name vae_camcan_anneal_z4096 \
--load_step 18000 \
--test_files brats_test_list.p \
--fprate 0.001 \
--weight 1.1 \
--restore_constraint L1 \
--preset_threshold 0.5810966075299717 1.4189008786753248 1.7473892764027208 2.3792834641455753 \
--config conf.yaml

python restore_tv_test.py --model_name vae_camcan_anneal_z4096 \
--load_step 18000 \
--test_files brats_test_list.p \
--fprate 0.001 \
--weight 2.02 \
--restore_constraint TV \
--preset_threshold 0.31909454389269887 0.4893179539127721 0.6441785286708364 1.017097258151633 \
--config conf.yaml

0.31909454389269887, 0.4893179539127721, 0.6441785286708364, 1.017097258151633

0.5810966075299717, 1.4189008786753248, 1.7473892764027208, 2.3792834641455753

python restore_tv_test.py --model_name vae_camcan_01data \
--load_step 99500 \
--test_files brats_test_list.p \
--fprate 0.001 \
--weight 4.25 \
--restore_constraint TV \
--preset_threshold 0.16637816056074692 0.19267491696355718 0.19808478022284384 0.1500650784731307 \
--config conf.yaml

python restore_simple_approximation.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--test_files ../brats_test.txt \
--fprate 0.001 \
--weight 17.0 \
--restore_method neural \
--restore_constraint TV \
--preset_threshold 1.6147044312655163 0.7766226131131138 0.5516066311673229 0.28456133727271216 \
--config conf.yaml

python restore.py --model_name covVAE_reverse_retry \
--load_step 193500 \
--test_files ../brats_test.txt \
--fprate 0.001 \
--weight 17.0 \
--restore_method cov_estimate \
--restore_constraint TV \
--preset_threshold 1.6147044312655163 0.7766226131131138 0.5516066311673229 0.28456133727271216 \
--config conf.yaml

python restore.py --model_name aae_camcan \
--load_step 17500 \
--test_files ../brats_test_list.p \
--fprate 0.001 \
--weight 15.0 \
--restore_method neural \
--restore_constraint TV \
--preset_threshold 1.6147044312655163 0.7766226131131138 0.5516066311673229 0.28456133727271216 \
--config conf.yaml

python restore.py --model_name vae_camcan \
--load_step 40000 \
--test_files ../brats_test_list.p \
--fprate 0.001 \
--weight 15.0 \
--restore_method neural \
--restore_constraint TV \
--preset_threshold 1.6147044312655163 0.7766226131131138 0.5516066311673229 0.28456133727271216 \
--config conf.yaml

python restore.py --model_name vae_camcan \
--load_step 40000 \
--test_files ../brats_test_list.p \
--fprate 0.001 \
--weight 15.0 \
--restore_method neural \
--restore_constraint TV \
--preset_threshold 1.6147044312655163 0.7766226131131138 0.5516066311673229 0.28456133727271216 \
--config conf.yaml

python restore_MAP.py --model_name vaeres_t1 \
--load_step 99500 \
--test_files atlas_test_list.p \
--fprate 0.001 \
--weight 1. \
--preset_threshold 1.59 0.879506969263945 0.969506969263945 \
--config conf.yaml

 0.3852874724030092

0.4960756133558828, 0.764372477047829, 0.879506969263945, 0.3852874724030092, 0.05] 0.1

python restore_MAP.py --model_name vaeres_t1 \
--load_step 99500 \
--test_files atlas_test_list.p \
--fprate 0.001 \
--weight 30. \
--preset_threshold 0.4960756133558828 0.764372477047829 0.879506969263945 \
--config conf.yaml

python restore_MAP.py --model_name vae_camcan \
--load_step 99500 \
--test_files MSL_test_list.p \
--fprate 0.001 \
--weight 0.0 \
--preset_threshold 1.6147044312655163 0.7766226131131138 0.5516066311673229 \
--config conf.yaml

python restore_gmvae.py --model_name gmvae_res \
--test_files ../brats_test_list.p \
--weight 4. \
--visualize 0 \
--save_evaluation 0

python restore_gmvae.py --model_name gmvae_res \
--test_files ../brats_test_list.p \
--weight 5.5 \
--visualize 0 \
--preset_threshold 0.5109632732949269 0.7394615298077362 0.4004220084854395 0.7394615298077362 \
--save_evaluation 0

python restore_gmvae.py --model_name gmvae_res \
--test_files ../brats_test.txt \
--weight 5.5 \
--visualize 1 \
--preset_threshold 0.5109632732949269 0.7394615298077362 0.4004220084854395 0.7394615298077362 \
--save_evaluation 1

python restore_gmvae.py --model_name gmvae_res \
--test_files ../brats_test_list.p \
--weight 5.5 \
--visualize 0 \
--preset_threshold 0.5109632732949269 0.7394615298077362 0.4004220084854395 0.7394615298077362 \
--save_evaluation 0

python restore_gmvae.py --model_name gmvae_res \
--test_files ../brats_test_list.p \
--weight 9. \
--visualize 0 \
--preset_threshold 0.41103258160132217 0.5646932628184602 0.3357632678970222 0.5646932628184602 \
--save_evaluation 0


python restore_gmvae.py --model_name gmvae_c6 \
--test_files ../brats_test_list.p \
--weight 4.8 \
--visualize 0 \
--preset_threshold 0.33084701285081053 0.5162543595638394 0.2608170422625512 0.5162543595638394 \
--save_evaluation 0


python restore_gmvae.py --model_name gmvae_c6 \
--test_files ../brats_test_list.p \
--weight 7. \
--visualize 0 \
--preset_threshold 0.3134956527185678 0.4320680033161194 0.25763198956604305 0.4320680033161194 \
--save_evaluation 0


python restore_gmvae.py --model_name gmvae_c6 \
--test_files ../brats_test_list.p \
--weight 9. \
--visualize 0 \
--preset_threshold 0.3893079716833413 0.5421399220266778 0.3165705088651852 0.5421399220266778 \
--save_evaluation 0


python restore_gmvae.py --model_name gmvae_c3 \
--test_files ../brats_test_list.p \
--weight 5.0 \
--visualize 0 \
--preset_threshold 0.4319607021834304 0.570302396018094 0.35715327605523894 0.570302396018094 \
--save_evaluation 0

python restore_gmvae.py --model_name gmvae_res_z16 \
--test_files ../brats_test_list.p \
--weight 5.5 \
--visualize 0 \
--preset_threshold 0.5109632732949269 0.7394615298077362 0.4004220084854395 0.7394615298077362 \
--save_evaluation 0

python restore_gmvae.py --model_name gmvae_res_z64 \
--test_files ../brats_test_list.p \
--weight 5.5 \
--visualize 0 \
--preset_threshold 0.5109632732949269 0.7394615298077362 0.4004220084854395 0.7394615298077362 \
--save_evaluation 0

python restore_gmvae_t1.py --model_name gmvae_c3_t1 \
--test_files atlas_test_list.p \
--weight 1.0 \
--visualize 0 \
--preset_threshold 0.5825873136007067 0.9095250731311475 0.4453675999027207 0.9095250731311475 \
--save_evaluation 0

python restore_gmvae_t1.py --model_name gmvae_c6_t1 \
--test_files atlas_test_list.p \
--weight 9.0 \
--visualize 0 \
--preset_threshold 0.3478389847229493 0.4842747424130033 0.2826096761669864 0.4842747424130033 \
--save_evaluation 0

python restore_gmvae_t1.py --model_name gmvae_c9_t1 \
--test_files atlas_test_list.p \
--weight 5.0 \
--visualize 0 \
--preset_threshold 0.2824197066890273 0.4164070400215904 0.22230979936702422 0.4164070400215904 \
--save_evaluation 0

python restore_gmvae_t1.py --model_name gmvae_c9_t1 \
--test_files atlas_test_list.p \
--weight 3.0 \
--visualize 0 \
--preset_threshold 0.340462254711764 0.5462880166930957 0.2575304795570943 0.5462880166930957 \
--save_evaluation 0

python restore_gmvae_t1.py --model_name gmvae_c9_t1 \
--test_files atlas_test_list.p \
--weight 7.0 \
--visualize 0 \
--preset_threshold 0.28694377186311215 0.40004882098330985 0.23326467464145678 0.40004882098330985 \
--save_evaluation 0

python restore_gmvae_t1.py --model_name gmvae_c6_t1 \
--test_files atlas_test_list.p \
--weight 5.0 \
--visualize 0 \
--preset_threshold 0.3175158298955608 0.5016637643647299 0.23929978167481095 0.5016637643647299 \
--save_evaluation 0

python restore_gmvae_t1.py --model_name gmvae_c3_t1 \
--test_files atlas_test_list.p \
--weight 9.0 \
--visualize 0 \
--preset_threshold 0.3296486201365819 0.45685944976443427 0.26819094653667586 0.45685944976443427 \
--save_evaluation 0

python restore_gmvae_t1.py --model_name gmvae_c3_t1 \
--test_files atlas_test_list.p \
--weight 7.0 \
--visualize 0 \
--preset_threshold 0.3296486201365819 0.45685944976443427 0.26819094653667586 0.45685944976443427 \
--save_evaluation 0

python restore_gmvae_t1.py --model_name gmvae_c3_t1 \
--test_files atlas_test_list.p \
--weight 5.0 \
--visualize 0 \
--preset_threshold 0.2687130360582425 0.3888635666163266 0.2128245491993242 0.3888635666163266 \
--save_evaluation 0

python restore_gmvae_t1.py --model_name gmvae_c3_t1 \
--test_files atlas_test_list.p \
--weight 4.0 \
--visualize 0 \
--preset_threshold 0.4793434510057307 0.7621541310349431 0.38333789426086873 0.7621541310349431 \
--save_evaluation 0