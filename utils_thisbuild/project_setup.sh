cd ~/
#sh -c 'mv utils_thisbuild content'
cd ~/cei_pytorch_vc/utils_thisbuild

# download necessary model file and map file 
# TODO: check file existed first
sudo /home/docker/.local/bin/gdown --id "11dHZ_Fl2DLkfxlg0_4ayimRgjowLOkgs"
sudo /home/docker/.local/bin/gdown --id "1uFBbk0iqyCDAAnrs8jqNgz3T3I9ewuB5"
sudo /home/docker/.local/bin/gdown --id "16clV1dTpIBjdZossSDXGnjroe8BjKTzL"
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
mv 20210412_cei_ljspeech_checkpoint_step001630000.pth ~/cei_pytorch_vc/deepvoice3_0421749/checkpoints

# necessary nltk data for train/synthesis
python -c "import nltk; nltk.download('cmudict')"
python -c "import nltk; nltk.download('punkt')"

