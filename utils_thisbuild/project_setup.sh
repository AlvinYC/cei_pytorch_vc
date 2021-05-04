cd ~/
#sh -c 'mv utils_thisbuild content'
cd ~/cei_pytorch_vc/utils_thisbuild

# download necessary model file and map file 
# TODO: check file existed first
sudo /home/docker/.local/bin/gdown --id "11dHZ_Fl2DLkfxlg0_4ayimRgjowLOkgs"
sudo /home/docker/.local/bin/gdown --id "1uFBbk0iqyCDAAnrs8jqNgz3T3I9ewuB5"
sudo /home/docker/.local/bin/gdown --id "16clV1dTpIBjdZossSDXGnjroe8BjKTzL"; :< 'ljsppech cei pretrain'
# p225/p226/p227 adaptation model 
:<'
sudo /home/docker/.local/bin/gdown --id "1VFNPydEDdz9jWhZeAYjodtygM_87mXm1"
sudo /home/docker/.local/bin/gdown --id "1Dv1LD-usuGilqn4yBarrG1YGUdUgzNs8"
sudo /home/docker/.local/bin/gdown --id "1exapyU-P7o9xiRiJ1nbH8tLPnGMCmTjE"
'
#sudo /home/docker/.local/bin/gdown --id "1OVwFpKiI4NHSAA1n5EjneAUaDcQfH9_c" //p376
# p225/226/p227 10min adapation model
sudo /home/docker/.local/bin/gdown --id "11ROOFj0wfiPjZMVTXKaIF_KwwTT1-EYw"
sudo /home/docker/.local/bin/gdown --id "1SXBzeBqP6QKKE3e3I0mzc-e-7DvLq5rT"
sudo /home/docker/.local/bin/gdown --id "1SXD6X1v0FbKf0Uvo3g3HQF8KGchf7j3e"
# p225/p226/p227 target wav
sudo /home/docker/.local/bin/gdown --id "1kj_X411hmSkQb0CUvcsXwK8BMTXhywmo"
sudo /home/docker/.local/bin/gdown --id "13h-khtJe2x1Yi6zskxtidUJSRNcsVgti"
sudo /home/docker/.local/bin/gdown --id "1ZT5wqdavaDZk6wiQ5FVJi3gPJAIdzABY"
sudo /home/docker/.local/bin/gdown --id "1Yr9ytkXHAWIhCjEvJczYelHSh1LpT1Iz"
sudo /home/docker/.local/bin/gdown --id "1080PECgaBsZLKgg5GuDmRxKhdNhSMpPL" 
sudo /home/docker/.local/bin/gdown --id "10mtmmZuTkIGe0QGF0hj1owrfqiR0bfe_"
sudo /home/docker/.local/bin/gdown --id "1CVrjNFFUfkPcfl38I8jJf2czRvSR-h2N"
sudo /home/docker/.local/bin/gdown --id "1CgQxYeIisp6uA_9p5aapS7tNPgUdwOQ0"
sudo /home/docker/.local/bin/gdown --id "12gfNYnQYpbgfK961TO23RS5bZDBPuPAc"
# p376
:<' mark due to p376 is India rather than English
sudo /home/docker/.local/bin/gdown --id "1LwMAEx-PF0dOf3nhqMLkQHwun5Y2CG3r"
sudo /home/docker/.local/bin/gdown --id "1488XpJKVRMfk8xCI4u5GeEKImRAU755f"
sudo /home/docker/.local/bin/gdown --id "123rQ93Xt_K0Jhpj4ROQhj3eVIOnnfVde"
'
sudo chown -R docker:docker ~/cei_pytorch_vc/utils_thisbuild
tar -xvf preset.tar.gz
# merge all id_rsa.pub into this container
#sh -c 'cat /home/docker/utils_thisbuild/*pub > /home/docker/.ssh/authorized_keys'
sh -c 'git config --global user.name alvinyc'
sh -c 'git config --global user.email chen.yongcheng@gmail.com'

#rsync -av --exclude='*sh' ~/utils_thisbuild ~/cei_pytorch_vc/ 
#basically, this folder need the following file
# 1.[file] 20171222_deepvoice3_vctk108_checkpoint_step000300000.pth
# 2 [file] 20210412_cei_ljspeech_checkpoint_step001630000.pth
# 3.[folder] presets
#        - [file] deepvoice3_ljspeech.json
#        - [file] deepvoice3_niklm.json
#        - [file] deepvoice3_nikls.json
#        - [file] deepvoice3_vctk.json
#        - [file] nyanko_ljspeech.json

# move ljspeech inference model to checkpoints path 
mkdir -p ~/cei_pytorch_vc/deepvoice3_0421749/checkpoints
:< 'moving model downloaded from google drive to checkpoint folder
mv 20210412_cei_ljspeech_checkpoint_step001630000.pth ~/cei_pytorch_vc/deepvoice3_0421749/checkpoints
# full dataset adapation model
mv 20210426_cei_ljspeech_vctk_p225_checkpoint_step000020000.pth ~/cei_pytorch_vc/deepvoice3_0421749/checkpoints
mv 20210426_cei_ljspeech_vctk_p226_checkpoint_step000040000.pth ~/cei_pytorch_vc/deepvoice3_0421749/checkpoints
mv 20210426_cei_ljspeech_vctk_p227_checkpoint_step000040000.pth ~/cei_pytorch_vc/deepvoice3_0421749/checkpoints
'
mv *pth ~/cei_pytorch/deepvoice3_0421749/checkpoints
mkdir -p ~/cei_pytorch_vc/deepvoice3_0421749/eval_output
mv p225*wav ~/cei_pytroch_vc/deepvocie_0421749/eval_output
mv p226*wav ~/cei_pytorch_vc/deepvoice_0421749/eval_output
mv p376*wav ~/cei_pytorch_vc/deepvoice_0421749/eval_output

# necessary nltk data for train/synthesis
python -c "import nltk; nltk.download('cmudict')"
python -c "import nltk; nltk.download('punkt')"

